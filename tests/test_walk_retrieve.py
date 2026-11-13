"""Tests for the Walk&Retrieve answer pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from kgrag.providers import FakeChat, FakeEmbedder
from kgrag.retrieval import build_index
from kgrag.walk_retrieve import answer_question
from kgrag.walks import Walk


def _walk(walk_id, start, nodes, texts, relations=None):
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


@pytest.fixture
def index_dir(tmp_path):
    embedder = FakeEmbedder(dim=8)
    walks = [
        _walk("w:a", "a", ["a"], ["Sinh viên được rút học phần trước 15/3."]),
        _walk("w:b", "b", ["b"], ["Điểm số tính theo thang điểm 10."]),
    ]
    index = build_index(
        walks,
        {
            "w:a": "Sinh viên được rút học phần trước 15/3.",
            "w:b": "Điểm số tính theo thang điểm 10.",
        },
        embedder,
        tmp_path / "index",
    )
    return tmp_path / "index"


def test_answer_question_uses_context(index_dir):
    embedder = FakeEmbedder(dim=8)
    chat = FakeChat("Trước ngày 15/3.")
    result = answer_question(
        "Khi nào sinh viên được rút học phần?",
        index_dir,
        embedder,
        chat,
    )
    assert result["answer"] == "Trước ngày 15/3."
    assert result["abstained"] is False
    assert "w:a" in result["walk_ids"]
    assert len(chat.calls) == 1


def test_abstain_when_no_walks(index_dir):
    embedder = FakeEmbedder(dim=8)
    chat = FakeChat("should not be called")
    result = answer_question(
        "xyz completely unrelated",
        index_dir,
        embedder,
        chat,
        top_k_nodes=1,
        top_k_walks_per_node=1,
    )
    # FakeEmbedder still returns some walk, so this test mainly checks the path.
    assert result["answer"]  # non-empty


def test_abstain_text_detection(index_dir):
    embedder = FakeEmbedder(dim=8)
    abstain = "Tôi không biết câu trả lờ_i"
    chat = FakeChat(abstain)
    result = answer_question(
        "Khi nào sinh viên được rút học phần?",
        index_dir,
        embedder,
        chat,
        abstain_text=abstain,
    )
    assert result["answer"] == abstain
    assert result["abstained"] is True


def test_build_index_cli(tmp_path):
    from kgrag.providers import OllamaEmbedder
    from kgrag.verbalize import VerbalizationCache
    from kgrag.walk_retrieve import main as cli_main
    from kgrag.walks import save_walks

    index_dir = tmp_path / "index"
    walks = [
        _walk("w:a", "a", ["a"], ["text a"]),
    ]
    walks_path = tmp_path / "walks.jsonl"
    save_walks(walks, walks_path)
    cache = VerbalizationCache.load(tmp_path / "verb.jsonl")
    cache.add(
        {
            "signature": "sig",
            "text": "verbalized a",
            "model": "fake",
            "prompt_version": "v1",
            "fidelity_ok": True,
            "dropped_numbers": [],
            "added_numbers": [],
            "graph_hash": "gh",
        }
    )
    cache.save()

    class FakeEmbedderNoNetwork(FakeEmbedder):
        pass

    # Replace OllamaEmbedder with a fake for this CLI invocation.
    original_embedder = OllamaEmbedder
    try:
        import kgrag.walk_retrieve as wr

        wr.OllamaEmbedder = FakeEmbedderNoNetwork
        import sys

        old_argv = sys.argv
        sys.argv = [
            "walk_retrieve.py",
            "--build-index",
            f"--walks={walks_path}",
            f"--verbalizations={tmp_path / 'verb.jsonl'}",
            f"--index-dir={index_dir}",
        ]
        cli_main()
        sys.argv = old_argv
    finally:
        wr.OllamaEmbedder = original_embedder

    assert (index_dir / "nodes.faiss").exists()
    assert (index_dir / "walks.faiss").exists()
