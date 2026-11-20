"""Tests for shared utilities."""

from __future__ import annotations

import numpy as np

from kgrag.providers import FakeEmbedder
from kgrag.utils import (
    embed_batch,
    l2_normalize,
    sha256_file,
    token_count,
    truncate_to_tokens,
)


def test_sha256_file_empty(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert sha256_file(path) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_content(tmp_path):
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    assert sha256_file(path) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_l2_normalize_rows():
    vectors = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    np.testing.assert_allclose(np.linalg.norm(normalized[0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(normalized[1], [0.0, 0.0])


def test_embed_batch_shapes():
    embedder = FakeEmbedder(dim=8)
    vectors = embed_batch(["a", "b"], embedder, batch_size=1)
    assert vectors.shape == (2, 8)
    direct = np.asarray(embedder.embed(["a", "b"]), dtype=np.float32)
    np.testing.assert_allclose(vectors, direct)


def test_embed_batch_empty():
    vectors = embed_batch([], FakeEmbedder(dim=4))
    assert vectors.shape == (0, 0)


def test_token_helpers():
    assert token_count("a b  c") == 3
    assert truncate_to_tokens("a b c", 2) == "a b"
    assert truncate_to_tokens("a b", 5) == "a b"
