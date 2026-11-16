"""Tests for the Traditional RAG baseline."""

from __future__ import annotations

import numpy as np
import pytest

from kgrag.ingest import load_source_text, parse_regulation
from kgrag.providers import FakeChat, FakeEmbedder
from kgrag.traditional_rag import (
    CHUNKER_VERSION,
    Chunk,
    _SYSTEM_PROMPT,
    answer_question,
    build_chunk_index,
    build_chunks,
    load_chunk_index,
    load_chunks,
    retrieve_chunks,
    save_chunks,
)


@pytest.fixture(scope="module")
def units():
    return parse_regulation(load_source_text()).units


@pytest.fixture(scope="module")
def chunks(units):
    return build_chunks(units)


def test_build_chunks_counts(chunks):
    levels = [chunk.level for chunk in chunks]
    assert len(chunks) == 132
    assert levels.count("clause") == 127
    assert levels.count("article") == 5


def test_chunks_are_non_empty(chunks):
    for chunk in chunks:
        assert chunk.text.strip()


def test_clause_less_articles_become_chunks(chunks):
    chunk_ids = {chunk.chunk_id for chunk in chunks}
    assert "decision:dieu-1" in chunk_ids


def test_table_chunks_include_header_row(chunks):
    chunk = next(
        chunk
        for chunk in chunks
        if chunk.chunk_id == "regulation:chuong-ii:dieu-10:khoan-1"
    )
    assert "| Loại học phần | Tối thiểu | Chuẩn | Tối đa |" in chunk.text
    assert chunk.text.index("| Loại học phần |") < chunk.text.index("| Lý thuyết")


def test_all_leaf_units_are_covered(units, chunks):
    texts = [chunk.text for chunk in chunks]
    leaves = [
        unit
        for unit in units
        if unit.level in {"point", "subpoint", "table_row"} and unit.text.strip()
    ]
    missing = [
        unit.id
        for unit in leaves
        if not any(unit.text.strip() in text for text in texts)
    ]
    assert missing == []


def test_chunk_roundtrip(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    save_chunks(chunks, path)
    loaded = load_chunks(path)
    assert loaded == chunks


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        level="clause",
        citation_label=chunk_id,
        text=text,
    )


def test_chunk_index_build_and_load(tmp_path):
    chunks = [
        _chunk("c1", "Sinh viên được rút học phần trước 15/3."),
        _chunk("c2", "Điểm số tính theo thang điểm 10."),
    ]
    index = build_chunk_index(
        chunks,
        FakeEmbedder(dim=8),
        tmp_path / "index",
        embed_model="fake",
        source_hash="abc",
    )
    assert index.metadata["dim"] == 8
    assert index.metadata["chunk_count"] == 2
    assert index.metadata["chunker_version"] == CHUNKER_VERSION
    assert index.metadata["embed_model"] == "fake"
    assert index.metadata["source_hash"] == "abc"

    loaded = load_chunk_index(tmp_path / "index")
    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.texts == index.texts
    np.testing.assert_array_almost_equal(loaded.vectors, index.vectors)


def _small_index(tmp_path):
    chunks = [
        _chunk("c1", "Sinh viên được rút học phần trước 15/3."),
        _chunk("c2", "Điểm số tính theo thang điểm 10."),
        _chunk("c3", "Sinh viên bị cảnh báo học vụ khi điểm trung bình dưới 1.0."),
    ]
    index = build_chunk_index(chunks, FakeEmbedder(dim=8), tmp_path / "index")
    return index, FakeEmbedder(dim=8)


def test_retrieve_chunks_ranks_exact_match_first(tmp_path):
    index, embedder = _small_index(tmp_path)
    result = retrieve_chunks(
        "Điểm số tính theo thang điểm 10.",
        index,
        embedder,
        top_k=2,
    )
    assert result["chunks"][0]["chunk_id"] == "c2"
    assert result["chunks"][0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert "thang điểm 10" in result["context"]


def test_retrieve_chunks_honors_token_budget(tmp_path):
    index, embedder = _small_index(tmp_path)
    result = retrieve_chunks(
        "Sinh viên được rút học phần trước 15/3.",
        index,
        embedder,
        top_k=3,
        max_context_tokens=5,
    )
    assert result["context_tokens"] <= 5
    assert result["retrieved_chunk_count"] >= 1


def test_retrieve_chunks_empty_index(tmp_path):
    index = build_chunk_index([], FakeEmbedder(dim=8), tmp_path / "index")
    result = retrieve_chunks("anything", index, FakeEmbedder(dim=8))
    assert result["chunks"] == []
    assert result["context"] == ""


def test_answer_question_uses_single_chat_call(tmp_path):
    _small_index(tmp_path)
    embedder = FakeEmbedder(dim=8)
    chat = FakeChat("Trước ngày 15/3.")
    result = answer_question(
        "Khi nào sinh viên được rút học phần?",
        tmp_path / "index",
        embedder,
        chat,
    )
    assert result["answer"] == "Trước ngày 15/3."
    assert result["abstained"] is False
    assert "c1" in result["chunk_ids"]
    assert len(chat.calls) == 1
    assert "NGỮ CẢNH:" in chat.calls[0][1]["content"]


def test_answer_question_abstains_without_chunks(tmp_path):
    build_chunk_index([], FakeEmbedder(dim=8), tmp_path / "index")
    chat = FakeChat("should not be called")
    result = answer_question(
        "anything",
        tmp_path / "index",
        FakeEmbedder(dim=8),
        chat,
    )
    assert result["abstained"] is True
    assert result["chunk_ids"] == []
    assert chat.calls == []


def test_answer_question_detects_abstain_text(tmp_path):
    _small_index(tmp_path)
    abstain = "Tôi không biết câu trả lời"
    result = answer_question(
        "Khi nào sinh viên được rút học phần?",
        tmp_path / "index",
        FakeEmbedder(dim=8),
        FakeChat(abstain),
        abstain_text=abstain,
    )
    assert result["answer"] == abstain
    assert result["abstained"] is True


def test_prompt_matches_walk_retrieve():
    from kgrag.walk_retrieve import _SYSTEM_PROMPT as walk_prompt

    assert _SYSTEM_PROMPT == walk_prompt


class _UnexpectedChat:
    def __init__(self, *args, **kwargs):
        raise AssertionError("chat must not be constructed")


def test_build_index_cli(tmp_path, monkeypatch):
    import sys

    import kgrag.traditional_rag as tr

    monkeypatch.setattr(tr, "OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr(tr, "OllamaChat", _UnexpectedChat)
    chunks_path = tmp_path / "chunks.jsonl"
    index_dir = tmp_path / "index"
    argv = [
        "traditional_rag.py",
        "--build-index",
        f"--chunks={chunks_path}",
        f"--index-dir={index_dir}",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    tr.main()

    assert chunks_path.exists()
    assert (index_dir / "chunks.faiss").exists()
    loaded = load_chunk_index(index_dir)
    assert len(loaded.chunk_ids) == 132
    assert loaded.metadata["chunker_version"] == CHUNKER_VERSION


def test_questions_file_rejects_missing_question(tmp_path, monkeypatch):
    import sys

    import kgrag.traditional_rag as tr

    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text('{"foo": "bar"}\n', encoding="utf-8")
    out_path = tmp_path / "answers.jsonl"

    monkeypatch.setattr(tr, "OllamaEmbedder", FakeEmbedder)
    monkeypatch.setattr(tr, "OllamaChat", _UnexpectedChat)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "traditional_rag.py",
            f"--questions-file={questions_path}",
            f"--out={out_path}",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        tr.main()

    assert "questions.jsonl:1" in str(excinfo.value)
    assert not out_path.exists()
