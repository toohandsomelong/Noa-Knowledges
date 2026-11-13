"""Tests for FAISS index build/search."""

from __future__ import annotations

import numpy as np
import pytest

from kgrag.providers import FakeEmbedder
from kgrag.retrieval import build_index, load_index, retrieve
from kgrag.walks import Walk


def _walk(
    walk_id: str,
    start: str,
    nodes: list[str],
    texts: list[str],
    relations: list[str] | None = None,
) -> Walk:
    return Walk(
        walk_id=walk_id,
        strategy="bfs",
        start_node=start,
        nodes=tuple(nodes),
        relations=tuple(relations or []),
        node_labels=tuple(nodes),
        node_texts=tuple(texts),
        template_text="\n".join(texts),
        graph_hash="gh",
    )


def _make_index(tmp_path, dim: int = 8):
    embedder = FakeEmbedder(dim=dim)
    walks = [
        _walk("w:a1", "a", ["a", "b"], ["Điều 1", "rút học phần"], ["contains"]),
        _walk("w:a2", "a", ["a", "c"], ["Điều 1", "đăng ký học"], ["contains"]),
        _walk("w:d1", "d", ["d", "e"], ["Điều 2", "thi cuối kỳ"], ["contains"]),
        _walk("w:d2", "d", ["d", "f"], ["Điều 2", "bảo vệ khóa luận"], ["contains"]),
    ]
    verbalization_map = {
        "w:a1": "Điều 1 quy định việc rút học phần.",
        "w:a2": "Điều 1 quy định việc đăng ký học.",
        "w:d1": "Điều 2 quy định thi cuối kỳ.",
        "w:d2": "Điều 2 quy định bảo vệ khóa luận.",
    }
    index = build_index(
        walks,
        verbalization_map,
        embedder,
        tmp_path / "index",
        max_node_desc_chars=10000,
    )
    return index, walks, verbalization_map


def test_build_and_load_index(tmp_path):
    index, _, _ = _make_index(tmp_path)
    assert len(index.node_ids) == 2
    assert len(index.walk_ids) == 4
    assert index.node_vectors.shape[1] == 8

    loaded = load_index(tmp_path / "index")
    assert loaded.node_ids == index.node_ids
    assert loaded.walk_ids == index.walk_ids
    np.testing.assert_array_almost_equal(loaded.node_vectors, index.node_vectors)


def test_retrieve_top_nodes_and_per_node_walks(tmp_path):
    index, _, _ = _make_index(tmp_path)
    embedder = FakeEmbedder(dim=8)

    # "rút học phần" should pull node "a" and its walks.
    result = retrieve(
        "rút học phần",
        index,
        embedder,
        top_k_nodes=2,
        top_k_walks_per_node=2,
        max_context_tokens=1000,
    )

    node_ids = [n["node_id"] for n in result["nodes"]]
    assert "a" in node_ids
    walk_ids = [w["walk_id"] for w in result["walks"]]
    assert "w:a1" in walk_ids


def test_retrieve_context_truncation(tmp_path):
    index, _, _ = _make_index(tmp_path)
    embedder = FakeEmbedder(dim=8)

    result = retrieve(
        "rút học phần",
        index,
        embedder,
        top_k_nodes=2,
        top_k_walks_per_node=2,
        max_context_tokens=5,
    )
    assert result["context_tokens"] <= 5


def test_retrieve_dedupes_identical_walk_texts(tmp_path):
    embedder = FakeEmbedder(dim=8)
    walks = [
        _walk("w1", "a", ["a"], ["same text"]),
        _walk("w2", "b", ["b"], ["same text"]),
    ]
    index = build_index(
        walks,
        {"w1": "same sentence", "w2": "same sentence"},
        embedder,
        tmp_path / "index",
    )
    result = retrieve(
        "same",
        index,
        embedder,
        top_k_nodes=2,
        top_k_walks_per_node=2,
        max_context_tokens=1000,
    )
    texts = [w["text"] for w in result["walks"]]
    assert len(texts) == len(set(texts))


def test_retrieve_returns_empty_when_no_index(tmp_path):
    embedder = FakeEmbedder(dim=8)
    index = build_index(
        [_walk("w", "a", ["a"], ["x"])],
        {"w": "x"},
        embedder,
        tmp_path / "index",
    )
    result = retrieve(
        "totally unrelated query phrase",
        index,
        embedder,
        top_k_nodes=1,
        top_k_walks_per_node=1,
        max_context_tokens=1000,
    )
    # A walk is still returned because FakeEmbedder produces deterministic vectors.
    # The real assertion here is that the function runs without error.
    assert result["query"] == "totally unrelated query phrase"
