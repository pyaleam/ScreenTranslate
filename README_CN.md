# 框选翻译

一个浮动翻译窗口，实时 OCR 识别屏幕文字并翻译。拖到任意位置即可翻译框内文字，无需截图、无需复制。

适合各种无汉化的视觉小说、游戏、软件界面及任何屏幕文本。

## 功能

- 🪟 无边框置顶窗口，拖拽 `⋮⋮` 移动，右下角缩放
- 可替换 OCR 引擎，适配不同语言
- 可替换 LLM 后端（DeepSeek / OpenAI / Ollama / 自定义兼容接口）
- 自定义目标语言
- 快捷键：`Ctrl+L` 开关翻译，`Escape` 清除，`F5` 强制刷新

## 环境要求

- Windows 10+
- Python 3.10+
- CUDA（可选，有 GPU 则 Florence-2 自动用 GPU）

## 安装

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 下载 OCR 模型

默认使用 **Florence-2-base**（微软轻量 VLM，0.23B 参数），约 0.5GB。

首次运行会自动从 HuggingFace 下载，或手动下载到项目目录：

```powershell
pip install huggingface_hub
huggingface-cli download microsoft/Florence-2-base --local-dir Florence-2-base
```

> **OCR 效果因语言而异。** Florence-2 对英文效果较好。翻译其他语言（日文、韩文、阿拉伯文等），建议找针对该语言优化的 OCR 模型。

## 配置

编辑项目根目录下的 `config.py`：

```python
# ── OCR 引擎 ──
OCR_BACKEND = "src.engine.ocr_florence.FlorenceOCREngine"

# ── LLM 翻译后端 ──
LLM_BACKEND = "deepseek"     # deepseek / openai / ollama / 自定义地址
LLM_API_KEY = "sk-xxxxx"     # API Key（Ollama 不需要）
LLM_BASE_URL = ""            # 留空用默认，或填自定义地址
LLM_MODEL = ""               # 留空用默认，或指定模型名
```

### DeepSeek（默认）

```python
LLM_BACKEND = "deepseek"
LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

获取 Key：[platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)

### OpenAI

```python
LLM_BACKEND = "openai"
LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"
LLM_MODEL = "gpt-4o-mini"
```

### 本地模型（Ollama）

无需联网，完全免费。

```powershell
ollama pull qwen2.5:7b
```

```python
LLM_BACKEND = "ollama"
LLM_MODEL = "qwen2.5:7b"
```

### 自定义 OpenAI 兼容接口

vLLM、LocalAI、LiteLLM、one-api 等：

```python
LLM_BACKEND = "http://localhost:8000/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "qwen2.5-7b-instruct"
```

## 替换 OCR 引擎

OCR 引擎只需实现一个方法：

```python
class MyEngine(OcrBackend):
    def recognize(self, image_bgr: np.ndarray) -> List[OcrResult]:
        ...
```

返回结果包含 `bbox`（坐标）、`original_text`（原文）、`confidence`（置信度）。

然后在 `config.py` 中指向你的引擎：

```python
OCR_BACKEND = "my_package.my_ocr.MyEngine"
```

系统用 `importlib` 动态加载，不需要改项目源码。

## 启动

```powershell
python main.py
```

## 使用

1. 拖拽窗口到需要翻译的文字上方
2. 点击 `▶` 或按 `Ctrl+L` 开启翻译
3. 点击语言按钮选择目标语言
4. 窗口停稳 1 秒后自动开始识别翻译，每 1.5 秒检测变化

## 快捷键

| 快捷键 | 功能 |
|---|---|
| `Ctrl+L` | 开关翻译 |
| `Escape` | 清除翻译，1 秒后重新识别 |
| `F5` | 强制刷新 |

## 注意事项

- 翻译 API 需要联网（Ollama 本地模型除外）
- 默认 OCR 模型文件（`Florence-2-base/`）需自行下载
- **OCR 效果因语言而异**：Florence-2 对英文较好，其他语言需自行配置对应引擎
- 关闭按钮立即终止进程（`os._exit(0)`）
