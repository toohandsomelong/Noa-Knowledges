"""LLM provider clients for embeddings and chat."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_CHAT_MODEL = "qwen2.5-rag:latest"


class Embedder(ABC):
    """Protocol for embedding providers."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class ChatModel(ABC):
    """Protocol for chat completion providers."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 512,
        seed: int = 42,
        format: dict[str, Any] | str | None = None,
    ) -> str:
        """Return the model's response content."""


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _with_retry(
    fn,
    *,
    max_retries: int,
    base_delay: float = 1.0,
    on_retry_message: str = "Request failed, retrying",
):
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except (requests.RequestException, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise RuntimeError(
        f"{on_retry_message} after {max_retries} attempts: {last_error}"
    ) from last_error


class OllamaEmbedder(Embedder):
    """Embedding client for a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        texts = [t if t else "" for t in texts]

        def _call() -> list[list[float]]:
            response = requests.post(
                _join_url(self.base_url, "/api/embed"),
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            if response.status_code == 404:
                return self._embed_single(texts)
            response.raise_for_status()
            payload = response.json()
            if "embeddings" not in payload:
                raise RuntimeError(f"Unexpected embed response: {payload.keys()}")
            return payload["embeddings"]

        try:
            return _with_retry(
                _call,
                max_retries=self.max_retries,
                on_retry_message="Ollama embedding request failed",
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Start the server with `ollama serve` and ensure the model is pulled."
            ) from exc

    def _embed_single(self, texts: Sequence[str]) -> list[list[float]]:
        """Fallback for older Ollama versions that only support /api/embeddings."""
        embeddings: list[list[float]] = []
        for text in texts:
            response = requests.post(
                _join_url(self.base_url, "/api/embeddings"),
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            embeddings.append(payload["embedding"])
        return embeddings


class OllamaChat(ChatModel):
    """Chat client for a local Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int = 512,
        seed: int = 42,
        format: dict[str, Any] | str | None = None,
    ) -> str:
        def _call() -> str:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "seed": seed,
                },
            }
            if format is not None:
                payload["format"] = format
            response = requests.post(
                _join_url(self.base_url, "/api/chat"),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        try:
            return _with_retry(
                _call,
                max_retries=self.max_retries,
                on_retry_message="Ollama chat request failed",
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Start the server with `ollama serve` and ensure the model is pulled."
            ) from exc



class _RateLimiter:
    """Very simple fixed-interval rate limiter."""

    def __init__(self, max_per_minute: int):
        self.interval = 60.0 / max_per_minute if max_per_minute > 0 else 0.0
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last_call = time.monotonic()


class CloudflareChat(ChatModel):
    """Chat client for Cloudflare Workers AI (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str,
        account_id: str,
        api_token: str,
        base_url: str,
        timeout: float = 120.0,
        max_retries: int = 6,
        max_requests_per_minute: int = 30,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.account_id = account_id
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = _RateLimiter(max_requests_per_minute)
        self.chat_template_kwargs = chat_template_kwargs
        self.usage_log: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int = 512,
        seed: int = 42,
        format: dict[str, Any] | str | None = None,
    ) -> str:
        if not self.account_id or not self.api_token:
            raise RuntimeError(
                "Cloudflare credentials missing. "
                "Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN."
            )

        self._rate_limiter.wait()

        def _call() -> str:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": num_predict,
                "seed": seed,
            }
            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = self.chat_template_kwargs
            response = requests.post(
                _join_url(self.base_url, "/chat/completions"),
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.usage_log.append(data.get("usage", {}))
            content = data["choices"][0]["message"].get("content")
            if content is None:
                raise RuntimeError(
                    f"Cloudflare model returned no content. Response: {json.dumps(data, ensure_ascii=False)[:500]}"
                )
            return content

        try:
            return _with_retry(
                _call,
                max_retries=self.max_retries,
                base_delay=5.0,
                on_retry_message="Cloudflare chat request failed",
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Cloudflare Workers AI. "
                "Check your network and credentials."
            ) from exc


class FakeEmbedder(Embedder):
    """Deterministic embedder for tests."""

    def __init__(self, dim: int = 8, **kwargs: Any):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import numpy as np

        vectors: list[list[float]] = []
        for text in texts:
            rng = np.random.default_rng(hash(text) % (2**32))
            vec = rng.normal(size=self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec) or 1.0
            vectors.append(vec.tolist())
        return vectors


class FakeChat(ChatModel):
    """Deterministic chat model for tests."""

    def __init__(self, response: str = "Fake answer"):
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 512,
        seed: int = 42,
        format: dict[str, Any] | str | None = None,
    ) -> str:
        self.calls.append(messages)
        return self.response
