"""OCR engine — pluggable OCR gateway.

Configuration via config.py:
  OCR_BACKEND = "src.engine.ocr_florence.FlorenceOCREngine"

Users can point to their own engine by changing the dotted class path.
"""

import importlib
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

# ── Config (with fallback for existing installs) ──────────
try:
    from config import OCR_BACKEND
except ImportError:
    OCR_BACKEND = "src.engine.ocr_florence.FlorenceOCREngine"


# ═══════════════════════════════════════════════════════════════
# Data class — backend-agnostic
# ═══════════════════════════════════════════════════════════════

@dataclass
class OcrResult:
    bbox: Tuple[int, int, int, int]
    original_text: str
    translated_text: str = ""
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Abstract backend
# ═══════════════════════════════════════════════════════════════

class OcrBackend(ABC):
    """Abstract OCR backend. Implement this to plug in a new OCR engine."""

    @abstractmethod
    def recognize(self, image_bgr: np.ndarray) -> List[OcrResult]:
        """Run OCR on a BGR image (H, W, 3), return results with bbox + text."""
        ...


# ═══════════════════════════════════════════════════════════════
# Dynamic loader
# ═══════════════════════════════════════════════════════════════

def _create_ocr_backend() -> OcrBackend:
    """Dynamically import and instantiate the OCR backend from config path."""
    path = OCR_BACKEND.strip()
    if not path:
        raise ValueError("OCR_BACKEND 未配置，请在 config.py 中设置")

    # Split "a.b.c.ClassName" into module path and class name
    parts = path.split(".")
    if len(parts) < 2:
        raise ValueError(
            f"OCR_BACKEND 格式错误: {path!r}，"
            f"应为 'module.path.ClassName'"
        )

    module_path = ".".join(parts[:-1])
    class_name = parts[-1]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"无法导入 OCR 模块 {module_path!r}: {e}\n"
            f"请检查 config.py 中的 OCR_BACKEND = {path!r}"
        )

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"模块 {module_path!r} 中没有 {class_name!r} 类\n"
            f"请检查 config.py 中的 OCR_BACKEND = {path!r}"
        )

    instance = cls()
    print(f"[OCR] 引擎: {class_name} (from {module_path})", flush=True)
    return instance


# ═══════════════════════════════════════════════════════════════
# Facade
# ═══════════════════════════════════════════════════════════════

class OCREngine:
    """OCR engine facade — delegates to the configured backend."""

    def __init__(self):
        self._engine: OcrBackend | None = None

    def _get_engine(self) -> OcrBackend:
        """Lazy-init the configured OCR backend."""
        if self._engine is not None:
            return self._engine
        self._engine = _create_ocr_backend()
        return self._engine

    def recognize(self, image_bgr: np.ndarray) -> List[OcrResult]:
        """Run OCR via the configured backend."""
        return self._get_engine().recognize(image_bgr)

    # ── Paragraph merge (static utility, backend-agnostic) ────

    @staticmethod
    def merge_paragraphs(results: List[OcrResult]) -> List[OcrResult]:
        """Merge OCR fragments into reading paragraphs by spatial proximity."""
        if len(results) <= 1:
            return results

        # 1. Group into reading lines (vertical overlap)
        results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))
        lines: List[List[OcrResult]] = []
        current = [results[0]]
        for r in results[1:]:
            cy1, cy2 = current[0].bbox[1], current[0].bbox[3]
            ry1, ry2 = r.bbox[1], r.bbox[3]
            overlap = max(0, min(cy2, ry2) - max(cy1, ry1))
            min_h = min(cy2 - cy1, ry2 - ry1)
            if overlap > min_h * 0.35:
                current.append(r)
            else:
                lines.append(current)
                current = [r]
        if current:
            lines.append(current)

        # 2. Within each line, merge adjacent fragments
        merged_lines: List[OcrResult] = []
        for line in lines:
            line.sort(key=lambda r: r.bbox[0])
            merged = [line[0]]
            for r in line[1:]:
                prev = merged[-1]
                gap = r.bbox[0] - prev.bbox[2]
                char_w = (prev.bbox[3] - prev.bbox[1]) * 0.5
                prev_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in prev.original_text)
                cur_cjk = any(0x4E00 <= ord(c) <= 0x9FFF for c in r.original_text)
                if gap < char_w * 2.5:
                    sep = "" if (prev_cjk and cur_cjk) else " "
                    merged[-1] = OcrResult(
                        bbox=(min(prev.bbox[0], r.bbox[0]),
                              min(prev.bbox[1], r.bbox[1]),
                              max(prev.bbox[2], r.bbox[2]),
                              max(prev.bbox[3], r.bbox[3])),
                        original_text=prev.original_text + sep + r.original_text,
                        confidence=max(prev.confidence, r.confidence),
                    )
                else:
                    merged.append(r)
            merged_lines.extend(merged)

        # 3. Group lines into paragraphs
        if len(merged_lines) <= 1:
            return merged_lines

        paragraphs: List[OcrResult] = []
        current_para = [merged_lines[0]]
        for r in merged_lines[1:]:
            prev = current_para[-1]
            prev_h = prev.bbox[3] - prev.bbox[1]
            cur_h = r.bbox[3] - r.bbox[1]
            left_diff = abs(r.bbox[0] - prev.bbox[0])
            height_ratio = min(prev_h, cur_h) / max(prev_h, cur_h) if max(prev_h, cur_h) > 0 else 0
            vert_gap = r.bbox[1] - prev.bbox[3]
            if left_diff < prev_h * 0.6 and height_ratio > 0.55 and vert_gap < prev_h * 0.8:
                current_para.append(r)
            else:
                paragraphs.append(_join_group(current_para))
                current_para = [r]
        if current_para:
            paragraphs.append(_join_group(current_para))

        print(f"[OCR] 合并: {len(results)} → {len(merged_lines)} 行 → {len(paragraphs)} 段",
              flush=True)
        return paragraphs


def _join_group(group: List[OcrResult]) -> OcrResult:
    """Join a group of OcrResults into one."""
    if len(group) == 1:
        return group[0]
    x1 = min(r.bbox[0] for r in group)
    y1 = min(r.bbox[1] for r in group)
    x2 = max(r.bbox[2] for r in group)
    y2 = max(r.bbox[3] for r in group)
    texts = []
    for i, r in enumerate(group):
        if i == 0:
            texts.append(r.original_text)
        else:
            prev_cjk = any(0x4E00 <= ord(c) <= 0x9FFF
                           for c in group[i - 1].original_text[-1:])
            cur_cjk = any(0x4E00 <= ord(c) <= 0x9FFF
                          for c in r.original_text[:1])
            sep = "" if (prev_cjk or cur_cjk) else " "
            texts.append(sep + r.original_text)
    return OcrResult(
        bbox=(x1, y1, x2, y2),
        original_text="".join(texts),
        confidence=sum(r.confidence for r in group) / len(group),
    )
