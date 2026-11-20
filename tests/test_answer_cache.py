"""Tests for the Walk&Retrieve answer cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kgrag.answer_cache import AnswerCache
from kgrag.providers import FakeChat, FakeEmbedder
from kgrag.retrieval import build_index
from kgrag.walk_retrieve import answer_question
from kgrag.walks import Walk


class CountingEmbedder(FakeEmbedder):
    def __init__(self, dim: int = 8):
        super().__init__(dim=dim)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


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
    build_index(
        walks,
        {
            "w:a": "Sinh viên được rút học phần trước 15/3.",
            "w:b": "Điểm số tính theo thang điểm 10.",
        },
        embedder,
        tmp_path / "index",
    )
    return tmp_path / "index"


def test_cache_hit_skips_chat_and_embedding(index_dir, tmp_path):
    cache = AnswerCache.load(tmp_path / "answers.jsonl")
    embedder = CountingEmbedder()
    chat = FakeChat("Trước ngày 15/3.")

    first = answer_question(
        "Khi nào sinh viên được rút học phần?", index_dir, embedder, chat, cache=cache
    )
    calls_after_first = embedder.calls
    assert len(chat.calls) == 1
    assert calls_after_first > 0

    second = answer_question(
        "Khi nào sinh viên được rút học phần?", index_dir, embedder, chat, cache=cache
    )
    assert second == first
    assert len(chat.calls) == 1
    assert embedder.calls == calls_after_first


def test_different_question_misses(index_dir, tmp_path):
    cache = AnswerCache.load(tmp_path / "answers.jsonl")
    embedder = FakeEmbedder()
    chat = FakeChat("answer")

    answer_question("q1", index_dir, embedder, chat, cache=cache)
    answer_question("q2", index_dir, embedder, chat, cache=cache)

    assert len(chat.calls) == 2
    assert cache.get("q1") is not None
    assert cache.get("q2") is not None


def test_no_cache_always_calls_chat(index_dir):
    embedder = FakeEmbedder()
    chat = FakeChat("answer")

    answer_question("q", index_dir, embedder, chat, cache=None)
    answer_question("q", index_dir, embedder, chat, cache=None)

    assert len(chat.calls) == 2


def test_re_cache_bypasses_lookup_and_appends(index_dir, tmp_path):
    path = tmp_path / "answers.jsonl"
    cache = AnswerCache.load(path)
    embedder = FakeEmbedder()
    chat = FakeChat("answer v1")

    answer_question("q", index_dir, embedder, chat, cache=cache)
    chat.response = "answer v2"
    result = answer_question("q", index_dir, embedder, chat, cache=cache, re_cache=True)

    assert result["answer"] == "answer v2"
    assert len(chat.calls) == 2

    with path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert [line["answer"] for line in lines] == ["answer v1", "answer v2"]

    reloaded = AnswerCache.load(path)
    assert reloaded.get("q")["answer"] == "answer v2"


def test_cache_roundtrip_and_tolerates_partial_line(tmp_path):
    path = tmp_path / "answers.jsonl"
    AnswerCache.load(path).add({"question": "q", "answer": "a"})

    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"question": "broken"')

    with pytest.warns(UserWarning, match="malformed"):
        loaded = AnswerCache.load(path)
    assert loaded.get("q") == {"question": "q", "answer": "a"}


def test_last_record_wins(tmp_path):
    path = tmp_path / "answers.jsonl"
    AnswerCache.load(path).add({"question": "q", "answer": "old"})
    AnswerCache.load(path).add({"question": "q", "answer": "new"})

    assert AnswerCache.load(path).get("q")["answer"] == "new"


def _write_questions(path: Path, question: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"question": question}, ensure_ascii=False))
        handle.write("\n")


def test_cli_no_cache_writes_no_cache_file(index_dir, tmp_path, monkeypatch):
    import kgrag.walk_retrieve as wr

    monkeypatch.setattr(wr, "OllamaEmbedder", lambda **kwargs: FakeEmbedder())
    monkeypatch.setattr(wr, "OllamaChat", lambda **kwargs: FakeChat("answer"))

    questions = tmp_path / "questions.jsonl"
    _write_questions(questions, "q")
    cache_path = tmp_path / "cache.jsonl"
    out_path = tmp_path / "answers.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "walk_retrieve",
            f"--index-dir={index_dir}",
            f"--questions-file={questions}",
            f"--out={out_path}",
            f"--answer-cache={cache_path}",
            "--no-cache",
        ],
    )
    wr.main()

    assert out_path.exists()
    assert not cache_path.exists()


def test_cli_re_cache_appends(index_dir, tmp_path, monkeypatch):
    import kgrag.walk_retrieve as wr

    monkeypatch.setattr(wr, "OllamaEmbedder", lambda **kwargs: FakeEmbedder())
    monkeypatch.setattr(wr, "OllamaChat", lambda **kwargs: FakeChat("answer"))

    questions = tmp_path / "questions.jsonl"
    _write_questions(questions, "q")
    cache_path = tmp_path / "cache.jsonl"
    out_path = tmp_path / "answers.jsonl"

    argv = [
        "walk_retrieve",
        f"--index-dir={index_dir}",
        f"--questions-file={questions}",
        f"--out={out_path}",
        f"--answer-cache={cache_path}",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    wr.main()
    assert len(AnswerCache.load(cache_path).records) == 1

    monkeypatch.setattr(sys, "argv", argv + ["--re-cache"])
    wr.main()

    with cache_path.open("r", encoding="utf-8") as handle:
        assert len([line for line in handle if line.strip()]) == 2
