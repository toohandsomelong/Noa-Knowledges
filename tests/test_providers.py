"""Tests for LLM provider clients."""

from __future__ import annotations

import pytest

from kgrag.providers import FakeChat, FakeEmbedder, OllamaChat, OllamaEmbedder


def test_fake_embedder_returns_normalized_vectors():
    embedder = FakeEmbedder(dim=8)
    vectors = embedder.embed(["a", "b", "a"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)
    import numpy as np

    norms = [np.linalg.norm(v) for v in vectors]
    assert all(abs(n - 1.0) < 1e-6 for n in norms)
    assert vectors[0] == vectors[2]


def test_fake_chat_records_calls():
    chat = FakeChat("test answer")
    messages = [{"role": "user", "content": "hi"}]
    answer = chat.chat(messages)
    assert answer == "test answer"
    assert chat.calls == [messages]


def test_ollama_embedder_payload(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, *, json, timeout):
        calls.append({"url": url, "json": json, "timeout": timeout})

        class Response:
            status_code = 200

            def json(self):
                return {"embeddings": [[0.1, 0.2, 0.3]]}

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr("kgrag.providers.requests.post", fake_post)
    embedder = OllamaEmbedder(model="m", base_url="http://x")
    result = embedder.embed(["hello"])
    assert result == [[0.1, 0.2, 0.3]]
    assert calls[0]["url"] == "http://x/api/embed"
    assert calls[0]["json"]["model"] == "m"
    assert calls[0]["json"]["input"] == ["hello"]


def test_ollama_embedder_404_falls_back(monkeypatch):
    calls: list[str] = []

    def fake_post(url, *, json, timeout):
        calls.append(url)

        class Response:
            def __init__(self, code):
                self.status_code = code

            def json(self):
                return {"embedding": [0.1, 0.2, 0.3]}

            def raise_for_status(self):
                pass

        if url.endswith("/api/embed"):
            return Response(404)
        return Response(200)

    monkeypatch.setattr("kgrag.providers.requests.post", fake_post)
    embedder = OllamaEmbedder(model="m", base_url="http://x")
    result = embedder.embed(["hello"])
    assert result == [[0.1, 0.2, 0.3]]
    assert any(c.endswith("/api/embeddings") for c in calls)


def test_ollama_chat_payload(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, *, json, timeout):
        calls.append({"url": url, "json": json})

        class Response:
            status_code = 200

            def json(self):
                return {"message": {"content": "answer"}}

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr("kgrag.providers.requests.post", fake_post)
    chat = OllamaChat(model="m", base_url="http://x")
    messages = [{"role": "user", "content": "q"}]
    answer = chat.chat(messages, temperature=0.5, num_ctx=2048, num_predict=100, seed=7)
    assert answer == "answer"
    payload = calls[0]["json"]
    assert payload["model"] == "m"
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.5
    assert payload["options"]["seed"] == 7


def test_ollama_chat_format_parameter(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, *, json, timeout):
        calls.append({"json": json})

        class Response:
            status_code = 200

            def json(self):
                return {"message": {"content": '{"sentences": ["s1"]}'}}

            def raise_for_status(self):
                pass

        return Response()

    monkeypatch.setattr("kgrag.providers.requests.post", fake_post)
    chat = OllamaChat(model="m", base_url="http://x")
    schema = {"type": "object", "properties": {"sentences": {"type": "array"}}}
    chat.chat([{"role": "user", "content": "q"}], format=schema)
    assert calls[0]["json"]["format"] == schema


def test_ollama_chat_retry_then_raise(monkeypatch):
    import requests

    attempts = {"count": 0}

    def fake_post(url, *, json, timeout):
        attempts["count"] += 1
        raise requests.ConnectionError("boom")

    monkeypatch.setattr("kgrag.providers.requests.post", fake_post)
    chat = OllamaChat(model="m", base_url="http://x", max_retries=2)
    with pytest.raises(RuntimeError, match="Ollama"):
        chat.chat([{"role": "user", "content": "q"}])
    assert attempts["count"] == 2
