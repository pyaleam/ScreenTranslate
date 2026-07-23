"""配置文件 — 填写你的 DeepSeek API Key 后即可使用。

获取 Key: https://platform.deepseek.com/api_keys
"""

# ── LLM 后端选择 ──────────────────────────────────────────
# deepseek  = DeepSeek API（推荐，便宜且效果好）
# openai    = OpenAI API
# ollama    = 本地 Ollama（需先运行 ollama serve）
# 也可填任意 OpenAI 兼容的 API 地址
LLM_BACKEND = "deepseek"
LLM_API_KEY = "sk-"
LLM_BASE_URL = ""          # 留空用默认地址，或填自定义地址
LLM_MODEL = ""             # 留空用默认模型，或指定模型名

# ── OCR 后端选择 ──────────────────────────────────────────
# 填写 OCR 引擎的完整类路径，工厂会动态加载
# 默认: src.engine.ocr_florence.FlorenceOCREngine
OCR_BACKEND = "src.engine.ocr_florence.FlorenceOCREngine"

# ── 翻译参数 ──────────────────────────────────────────────
TRANSLATE_TIMEOUT = 20     # API 超时（秒）
TRANSLATE_TEMPERATURE = 0.1  # 低温度保证一致性
STABLE_POLLS_REQUIRED = 2  # 连续 N 轮 poll 无变化才翻译（1=最快，2=更稳）
POLL_INTERVAL_MS = 800     # 内容变化检测间隔（毫秒）
CONTEXT_MAX_CHARS = 500    # 翻译上文参考的最大字数，用于术语和语境一致性
