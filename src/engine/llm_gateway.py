"""LLM Gateway — unified interface for cloud APIs and local models."""

import json
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Dict

# Import config
try:
    from config import (
        LLM_BACKEND, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL,
        TRANSLATE_TIMEOUT, TRANSLATE_TEMPERATURE,
    )
except ImportError:
    LLM_BACKEND = "deepseek"
    LLM_API_KEY = ""
    LLM_BASE_URL = ""
    LLM_MODEL = "deepseek-chat"
    TRANSLATE_TIMEOUT = 20
    TRANSLATE_TEMPERATURE = 0.1


# ═══════════════════════════════════════════════════════════════
# Abstract backend
# ═══════════════════════════════════════════════════════════════

class LLMBackend(ABC):
    """Abstract LLM backend. Subclass to add new providers."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = TRANSLATE_TEMPERATURE,
             max_tokens: int = 4096) -> str | None:
        """Send messages, return response text or None on failure."""
        ...


# ═══════════════════════════════════════════════════════════════
# OpenAI-compatible backend (DeepSeek, OpenAI, vLLM, etc.)
# ═══════════════════════════════════════════════════════════════

class OpenAICompatBackend(LLMBackend):
    """Backend for any OpenAI-compatible API.

    Works with:
      - DeepSeek:    https://api.deepseek.com/v1
      - OpenAI:      https://api.openai.com/v1
      - vLLM:        http://localhost:8000/v1
      - LocalAI:     http://localhost:8080/v1
      - LiteLLM:     any configured endpoint
    """

    def __init__(self, base_url: str, api_key: str, model: str):
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model

    def chat(self, messages, temperature=TRANSLATE_TEMPERATURE,
             max_tokens=4096):
        url = f"{self._base}/chat/completions"
        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")

        try:
            with urllib.request.urlopen(req, timeout=TRANSLATE_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] 调用失败 ({self._model}): {e}", flush=True)
            return None


# ═══════════════════════════════════════════════════════════════
# Ollama backend (local)
# ═══════════════════════════════════════════════════════════════

class OllamaBackend(LLMBackend):
    """Backend for locally running Ollama models.

    Requires: ollama serve (default: http://localhost:11434)
    """

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self._host = host.rstrip("/")
        self._model = model

    def chat(self, messages, temperature=TRANSLATE_TEMPERATURE,
             max_tokens=4096):
        url = f"{self._host}/api/chat"
        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body)
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=TRANSLATE_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            return data["message"]["content"]
        except Exception as e:
            print(f"[LLM] Ollama 调用失败 ({self._model}): {e}", flush=True)
            return None


# ═══════════════════════════════════════════════════════════════
# Gateway — creates the right backend from config
# ═══════════════════════════════════════════════════════════════

BACKEND_ALIASES = {
    "deepseek": ("openai", "https://api.deepseek.com/v1"),
    "openai":   ("openai", "https://api.openai.com/v1"),
    "ollama":   ("ollama", "http://localhost:11434"),
}


def create_backend() -> LLMBackend:
    """Factory: build LLM backend from config."""
    backend_type = LLM_BACKEND.lower()
    alias = BACKEND_ALIASES.get(backend_type)
    if alias is None:
        # Treat as custom OpenAI-compatible: LLM_BACKEND is the URL
        model = LLM_MODEL or "default"
        url = LLM_BASE_URL or backend_type
        print(f"[LLM] 自定义 OpenAI 兼容: {url} 模型={model}", flush=True)
        return OpenAICompatBackend(url, LLM_API_KEY, model)

    kind, default_url = alias
    url = LLM_BASE_URL or default_url
    model = LLM_MODEL or (
        "deepseek-chat" if backend_type == "deepseek" else
        "gpt-4o-mini" if backend_type == "openai" else
        "qwen2.5:7b"
    )

    if kind == "openai":
        print(f"[LLM] OpenAI 兼容: {url} 模型={model}", flush=True)
        return OpenAICompatBackend(url, LLM_API_KEY, model)
    elif kind == "ollama":
        print(f"[LLM] Ollama 本地: {url} 模型={model}", flush=True)
        return OllamaBackend(url, model)

    raise ValueError(f"Unknown backend: {backend_type}")
