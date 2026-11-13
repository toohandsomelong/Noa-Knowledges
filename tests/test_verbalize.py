"""Tests for walk verbalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from kgrag.providers import FakeChat
from kgrag.verbalize import (
    VerbalizationCache,
    _check_fidelity,
    _walk_signature,
    build_walk_verbalization_map,
    verbalize_walks,
)
from kgrag.walks import Walk


def _walk(
    nodes,
    relations=None,
    labels=None,
    texts=None,
    strategy="bfs",
    walk_id="w1",
    start="a",
    graph_hash="gh",
):
    relations = relations or []
    labels = labels or [f"L{n}" for n in nodes]
    texts = texts or [f"text {n}" for n in nodes]
    return Walk(
        walk_id=walk_id,
        strategy=strategy,
        start_node=start,
        nodes=tuple(nodes),
        relations=tuple(relations),
        node_labels=tuple(labels),
        node_texts=tuple(texts),
        template_text="template",
        graph_hash=graph_hash,
    )


def test_trivial_length_zero_walks_skip_llm():
    chat = FakeChat("unexpected")
    walks = [_walk(["a"], texts=["Điều 1 nội dung"])]
    records, stats = verbalize_walks(walks, chat)
    assert stats["trivial"] == 1
    assert stats["llm_calls"] == 0
    assert "Điều 1" in records[0]["text"]
    assert chat.calls == []


def test_verbalization_cache_hit_avoids_call(tmp_path):
    chat = FakeChat("Sentence one.")
    walks = [_walk(["a", "b"], relations=["contains"], texts=["Alpha", "Beta"])]
    cache = VerbalizationCache.load(tmp_path / "cache.jsonl")
    records1, _ = verbalize_walks(walks, chat, cache=cache, progress=False)
    assert len(chat.calls) == 1
    cache.save()

    cache2 = VerbalizationCache.load(tmp_path / "cache.jsonl")
    records2, stats = verbalize_walks(walks, chat, cache=cache2, progress=False)
    assert stats["cached"] == 1
    assert len(chat.calls) == 1  # no new call
    assert records1 == records2


def test_fidelity_flags_dropped_number():
    walk = _walk(
        ["a"],
        texts=["Điểm số phải đạt 8.5"],
    )
    ok, dropped, added = _check_fidelity(walk, "Điểm số phải đạt.", 400)
    assert ok is False
    assert "8.5" in dropped
    assert added == []


def test_fidelity_ok_when_numbers_present():
    walk = _walk(
        ["a"],
        texts=["Điểm số phải đạt 8.5"],
    )
    ok, dropped, added = _check_fidelity(walk, "Sinh viên cần đạt 8.5 điểm.", 400)
    assert ok is True
    assert dropped == []
    assert added == []


def test_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "verbalizations.jsonl"
    cache = VerbalizationCache.load(cache_path)
    cache.add(
        {
            "signature": "sig1",
            "text": "cached sentence",
            "model": "fake",
            "prompt_version": "v1",
            "fidelity_ok": True,
            "dropped_numbers": [],
            "added_numbers": [],
            "graph_hash": "gh",
        }
    )
    cache.save()

    loaded = VerbalizationCache.load(cache_path)
    assert loaded.get("sig1")["text"] == "cached sentence"


def test_build_walk_verbalization_map(tmp_path):
    walk_a = _walk(["a"], walk_id="w:a", texts=["A"])
    walk_b = _walk(["b"], walk_id="w:b", texts=["B"])
    cache = VerbalizationCache.load(tmp_path / "v.jsonl")
    for walk in (walk_a, walk_b):
        cache.add(
            {
                "signature": _walk_signature(walk, "fake"),
                "text": f"verbalized {walk.nodes[0]}",
                "model": "fake",
                "prompt_version": "v1",
                "fidelity_ok": True,
                "dropped_numbers": [],
                "added_numbers": [],
                "graph_hash": walk.graph_hash,
            }
        )
    mapping = build_walk_verbalization_map([walk_a, walk_b], cache, model="fake")
    assert mapping == {"w:a": "verbalized a", "w:b": "verbalized b"}


def test_batch_verbalization_parses_json():
    chat = FakeChat('{\"sentences\": [\"s1\", \"s2\"]}')
    walks = [
        _walk(["a", "b"], relations=["contains"], walk_id="w1", texts=["A", "B"]),
        _walk(["a", "c"], relations=["contains"], walk_id="w2", texts=["A", "C"]),
    ]
    records, stats = verbalize_walks(
        walks, chat, batch_size=2, workers=1, progress=False
    )
    assert stats["llm_calls"] == 1
    assert len(records) == 2
    assert records[0]["content"] == "s1"
    assert records[1]["content"] == "s2"


def test_batch_fallback_to_individual_on_bad_json():
    chat = FakeChat("not json")
    walks = [
        _walk(["a", "b"], relations=["contains"], walk_id="w1", texts=["A", "B"]),
    ]
    records, stats = verbalize_walks(
        walks, chat, batch_size=2, workers=1, progress=False
    )
    # Falls back to single walk, FakeChat returns same string each time.
    assert records[0]["content"] == "not json"


def test_verbalize_skips_duplicate_signatures():
    chat = FakeChat("one sentence")
    walk1 = _walk(["a", "b"], relations=["contains"], walk_id="w1", texts=["A", "B"])
    walk2 = _walk(["a", "b"], relations=["contains"], walk_id="w2", texts=["A", "B"])
    records, stats = verbalize_walks(
        [walk1, walk2], chat, batch_size=1, workers=1, progress=False
    )
    assert stats["total_unique"] == 1
    assert len(records) == 1
    assert len(chat.calls) == 1
