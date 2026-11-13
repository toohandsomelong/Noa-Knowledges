"""Load project configuration from environment and .env file."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any


def load_env(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Load environment variables from .env if dotenv is available."""
    if dotenv_path is None:
        dotenv_path = Path(".env")
    else:
        dotenv_path = Path(dotenv_path)

    try:
        from dotenv import load_dotenv as _load_dotenv

        if dotenv_path.exists():
            _load_dotenv(dotenv_path, override=False)
    except ImportError:
        pass

    return dict(os.environ)


def _normalize_url(url: str) -> str:
    return url.replace("host.docker.internal", "127.0.0.1").rstrip("/")


def _env(env: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return env if env is not None else os.environ


def ollama_url(env: Mapping[str, str] | None = None) -> str:
    env = _env(env)
    return _normalize_url(
        env.get("OLLAMA_QUERY_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434"
    )


def ollama_embed_url(env: Mapping[str, str] | None = None) -> str:
    env = _env(env)
    return _normalize_url(
        env.get("OLLAMA_EMBED_URL", ollama_url(env)) or ollama_url(env)
    )


def ollama_chat_model(env: Mapping[str, str] | None = None) -> str:
    env = _env(env)
    return env.get("OLLAMA_QUERY_MODEL", "qwen2.5-rag:latest") or "qwen2.5-rag:latest"


def ollama_embed_model(env: Mapping[str, str] | None = None) -> str:
    env = _env(env)
    return env.get("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b") or "qwen3-embedding:0.6b"


def cloudflare_account_id(env: Mapping[str, str] | None = None) -> str | None:
    return _env(env).get("CLOUDFLARE_ACCOUNT_ID")


def cloudflare_api_token(env: Mapping[str, str] | None = None) -> str | None:
    return _env(env).get("CLOUDFLARE_API_TOKEN")


def cloudflare_base_url(env: Mapping[str, str] | None = None) -> str:
    account_id = cloudflare_account_id(env)
    if not account_id:
        return ""
    return f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"


def cloudflare_chat_model(env: Mapping[str, str] | None = None) -> str:
    return (
        _env(env).get("CF_INDEX_MODEL", "@cf/zai-org/glm-4.7-flash")
        or "@cf/zai-org/glm-4.7-flash"
    )
