"""Small shared utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from kgrag.providers import Embedder


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_batch(
    texts: Sequence[str],
    embedder: Embedder,
    batch_size: int = 16,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        batch_vectors = embedder.embed(batch)
        vectors.append(np.asarray(batch_vectors, dtype=np.float32))
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def token_count(text: str) -> int:
    return len(text.split())


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])
