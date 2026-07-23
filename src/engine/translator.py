"""Translation engine — continuous-text translation, backend-agnostic."""

import re
from typing import List, Dict, Tuple, Deque
from collections import deque
from src.engine.llm_gateway import create_backend, LLMBackend

try:
    from config import TRANSLATE_TIMEOUT, TRANSLATE_TEMPERATURE
except ImportError:
    TRANSLATE_TIMEOUT = 20
    TRANSLATE_TEMPERATURE = 0.1

try:
    from config import CONTEXT_MAX_CHARS
except ImportError:
    CONTEXT_MAX_CHARS = 500

LANG_NAMES = {
    "zh-CN": "简体中文", "zh-TW": "繁体中文", "en": "English",
    "ja": "日本語", "ko": "한국어", "fr": "Français", "de": "Deutsch",
    "es": "Español", "pt": "Português", "ru": "Русский",
    "it": "Italiano", "ar": "العربية", "vi": "Tiếng Việt", "th": "ไทย",
}

SYSTEM_PROMPT = (
    "你是一个专业翻译引擎，内嵌于屏幕实时翻译工具中。"
    "你的译文会以浮动覆盖层显示在用户屏幕上，直接替换原文区域。\n\n"

    "## 输入\n"
    "你要翻译的文本来自屏幕 OCR 识别，可能源自任何场景——"
    "网页、文档、软件界面、游戏、视频字幕、聊天记录等。\n"
    "OCR 可能因物理位置分段而语义断裂，"
    "请你理解全文的整体含义后翻译，输出通顺连贯的译文。\n\n"

    "## 核心原则\n"
    "1. **意译优先** — 传达真实含义和意图，不逐字死译。\n"
    "2. **极简** — 用最少的字把话说清楚，不加解释、不啰嗦。\n"
    "3. **自然** — 读起来像目标语言原生写就的，不是翻译腔。\n"
    "4. **专有名词保留** — 人名、地名、品牌、网址、代码、文件名照抄。\n"
    "5. **界面文字 ≤ 8 字** — 按钮/菜单/标签类尽量控制。\n"
    "6. **格式跟随原文** — 原文是标题就译成标题，是列表就保持列表，是代码就不动。\n"
    "7. **语气跟随上文** — 上文给出的翻译历史反映了内容的语域和风格，保持一致。\n\n"

    "## 格式\n"
    "只输出翻译结果本身。不要加任何解释、注释、标记或引号包裹。"
)

# ── Lightweight context analysis ──────────────────────────────

def _analyze_context(history: Deque[Tuple[str, str]]) -> str:
    """Return a one-line hint describing what kind of content this appears to be.

    Heuristic based on line-length distribution and punctuation patterns.
    Generic enough to work across web pages, documents, games, UIs, etc.
    """
    if not history:
        return ""

    originals = [orig for orig, _ in history]
    if not originals:
        return ""

    avg_len = sum(len(o) for o in originals) / len(originals)

    # Detect quoted content (any language / quotation style)
    quote_count = 0
    for o in originals:
        if re.search(r'[「『」』""""''‘’“”]', o):
            quote_count += 1

    ratio_quoted = quote_count / len(originals)

    # Classify by text shape, not by assumed application
    if avg_len <= 3:
        return "【短文本 — 可能是界面标签、按钮或单个词汇】\n"
    elif ratio_quoted >= 0.4:
        return "【对话 — 请从上文判断说话人的语气和风格，保持一致】\n"
    elif avg_len <= 18:
        return "【短句 — 注意上下文连贯，可能来自对话、聊天或标题】\n"
    elif avg_len >= 50:
        return "【长文 — 注意语域、节奏和段落逻辑】\n"
    else:
        return ""


def _build_context_window(history: Deque[Tuple[str, str]],
                          max_chars: int) -> str:
    """Build a context string from recent translation pairs, respecting max_chars.

    Newest entries come last; the window is trimmed to fit max_chars
    by dropping the oldest entries first.
    """
    if not history or max_chars <= 0:
        return ""

    parts: List[str] = []
    used = 0
    for orig, trans in reversed(history):
        entry = f"{orig} → {trans}"
        if used + len(entry) > max_chars:
            if used == 0 and len(entry) > max_chars:
                entry = entry[:max_chars - 3] + "..."
            else:
                break
        parts.append(entry)
        used += len(entry)
        if used >= max_chars:
            break

    if not parts:
        return ""

    parts.reverse()
    context = "\n".join(parts)
    return (
        "【以下是前几轮翻译，用于理解上下文、保持术语和语气一致】\n"
        f"{context}\n\n"
    )


def _build_prompt(text: str, target_lang: str, context: str,
                  content_hint: str) -> str:
    lang_name = LANG_NAMES.get(target_lang, target_lang)
    parts = [content_hint, context, f"翻译为{lang_name}：\n\n{text}"]
    return "".join(p for p in parts if p)


class TranslationEngine:
    """Continuous-text translation via LLM gateway with sliding context window.

    Instead of only remembering the previous round, this keeps a
    sliding window of recent (original → translated) pairs whose total
    character count fits within CONTEXT_MAX_CHARS.  The LLM sees all of
    them as reference, which dramatically improves consistency across
    multi-line dialogue in visual novels.
    """

    def __init__(self):
        self._cache: Dict[Tuple[str, str], str] = {}
        self._context_history: Deque[Tuple[str, str]] = deque()
        self._backend: LLMBackend | None = None

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            self._backend = create_backend()
        return self._backend

    def translate(self, texts: List[str], target_lang: str) -> str:
        """Join OCR texts into continuous text, translate as a whole, return one string."""
        original = " ".join(t.strip() for t in texts if t.strip())
        if not original:
            return ""

        cache_key = (original, target_lang)
        if cache_key in self._cache:
            print("[翻译] 命中缓存", flush=True)
            return self._cache[cache_key]

        lang_name = LANG_NAMES.get(target_lang, target_lang)
        content_hint = _analyze_context(self._context_history)
        context = _build_context_window(self._context_history, CONTEXT_MAX_CHARS)
        prompt = _build_prompt(original, target_lang, context, content_hint)

        ctx_chars = len(context)
        hint_label = content_hint.strip() if content_hint else ""
        print(f"[翻译] {len(texts)} 段 → {lang_name}"
              f"{f' ({hint_label})' if hint_label else ''}"
              f"{f' 上文 {ctx_chars} 字' if ctx_chars else ''}",
              flush=True)
        print(f"  原文: {original[:120]}{'...' if len(original) > 120 else ''}", flush=True)

        backend = self._get_backend()
        response = backend.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        if response is None:
            self._cache[cache_key] = original
            return original

        translated = response.strip()
        self._cache[cache_key] = translated
        self._context_history.append((original, translated))

        print(f"  译文: {translated[:120]}{'...' if len(translated) > 120 else ''}"
              f" (历史 {len(self._context_history)} 轮)",
              flush=True)
        return translated

    def clear_cache(self):
        self._cache.clear()
        self._context_history.clear()
