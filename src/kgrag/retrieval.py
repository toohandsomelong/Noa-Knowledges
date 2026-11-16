"""Build and search a FAISS index over verbalized walks."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from kgrag.providers import Embedder
from kgrag.walks import Walk


@dataclass
class Index:
    """In-memory representation of a built retrieval index."""

    node_ids: list[str]
    walk_ids: list[str]
    walk_start_nodes: list[str]
    node_texts: list[str]
    walk_texts: list[str]
    node_index: faiss.IndexFlatIP
    walk_index: faiss.IndexFlatIP
    node_vectors: np.ndarray
    walk_vectors: np.ndarray
    metadata: dict[str, Any]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embed_batch(
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


def build_index(
    walks: list[Walk],
    verbalization_map: dict[str, str],
    embedder: Embedder,
    out_dir: str | Path,
    *,
    max_node_desc_chars: int = 6000,
    embed_batch_size: int = 16,
) -> Index:
    """Build a FAISS index of node and walk embeddings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_ids = sorted({w.start_node for w in walks})
    walks_by_node: dict[str, list[Walk]] = defaultdict(list)
    for walk in walks:
        walks_by_node[walk.start_node].append(walk)

    node_texts: list[str] = []
    for node_id in node_ids:
        node_walks = walks_by_node[node_id]
        parts: list[str] = []
        total = 0
        for walk in node_walks:
            text = verbalization_map.get(walk.walk_id, walk.template_text)
            if total + len(text) > max_node_desc_chars and parts:
                break
            parts.append(text)
            total += len(text)
        node_texts.append("\n".join(parts))

    walk_ids = [w.walk_id for w in walks]
    walk_start_nodes = [w.start_node for w in walks]
    walk_texts = [
        verbalization_map.get(w.walk_id, w.template_text) for w in walks
    ]

    node_vectors = _l2_normalize(
        _embed_batch(node_texts, embedder, batch_size=embed_batch_size)
    )
    walk_vectors = _l2_normalize(
        _embed_batch(walk_texts, embedder, batch_size=embed_batch_size)
    )

    dim = node_vectors.shape[1]
    node_index = faiss.IndexFlatIP(dim)
    walk_index = faiss.IndexFlatIP(dim)
    node_index.add(node_vectors)
    walk_index.add(walk_vectors)

    metadata = {
        "dim": dim,
        "node_count": len(node_ids),
        "walk_count": len(walk_ids),
        "max_node_desc_chars": max_node_desc_chars,
    }

    faiss.write_index(node_index, str(out_dir / "nodes.faiss"))
    faiss.write_index(walk_index, str(out_dir / "walks.faiss"))
    np.save(out_dir / "node_vectors.npy", node_vectors)
    np.save(out_dir / "walk_vectors.npy", walk_vectors)

    with (out_dir / "node_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(node_ids, handle, ensure_ascii=False, indent=2)
    with (out_dir / "walk_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(walk_ids, handle, ensure_ascii=False, indent=2)
    with (out_dir / "walk_meta.jsonl").open("w", encoding="utf-8") as handle:
        for wid, start in zip(walk_ids, walk_start_nodes):
            handle.write(
                json.dumps({"walk_id": wid, "start_node": start}, ensure_ascii=False)
            )
            handle.write("\n")
    with (out_dir / "node_texts.json").open("w", encoding="utf-8") as handle:
        json.dump(node_texts, handle, ensure_ascii=False, indent=2)
    with (out_dir / "walk_texts.json").open("w", encoding="utf-8") as handle:
        json.dump(walk_texts, handle, ensure_ascii=False, indent=2)
    with (out_dir / "index_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return Index(
        node_ids=node_ids,
        walk_ids=walk_ids,
        walk_start_nodes=walk_start_nodes,
        node_texts=node_texts,
        walk_texts=walk_texts,
        node_index=node_index,
        walk_index=walk_index,
        node_vectors=node_vectors,
        walk_vectors=walk_vectors,
        metadata=metadata,
    )


def load_index(index_dir: str | Path) -> Index:
    """Load a previously built FAISS index."""
    index_dir = Path(index_dir)

    node_index = faiss.read_index(str(index_dir / "nodes.faiss"))
    walk_index = faiss.read_index(str(index_dir / "walks.faiss"))
    node_vectors = np.load(index_dir / "node_vectors.npy")
    walk_vectors = np.load(index_dir / "walk_vectors.npy")

    with (index_dir / "node_ids.json").open("r", encoding="utf-8") as handle:
        node_ids = json.load(handle)
    with (index_dir / "walk_ids.json").open("r", encoding="utf-8") as handle:
        walk_ids = json.load(handle)
    with (index_dir / "node_texts.json").open("r", encoding="utf-8") as handle:
        node_texts = json.load(handle)
    with (index_dir / "walk_texts.json").open("r", encoding="utf-8") as handle:
        walk_texts = json.load(handle)
    with (index_dir / "index_meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    walk_start_nodes: list[str] = []
    with (index_dir / "walk_meta.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                walk_start_nodes.append(json.loads(line)["start_node"])

    return Index(
        node_ids=node_ids,
        walk_ids=walk_ids,
        walk_start_nodes=walk_start_nodes,
        node_texts=node_texts,
        walk_texts=walk_texts,
        node_index=node_index,
        walk_index=walk_index,
        node_vectors=node_vectors,
        walk_vectors=walk_vectors,
        metadata=metadata,
    )


def _token_count(text: str) -> int:
    return len(text.split())


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


def retrieve(
    query: str,
    index: Index,
    embedder: Embedder,
    *,
    top_k_nodes: int = 3,
    top_k_walks_per_node: int = 3,
    max_context_tokens: int = 4096,
) -> dict[str, Any]:
    """Retrieve top nodes and their walks, assemble context."""
    query_vector = _l2_normalize(
        np.asarray(embedder.embed([query]), dtype=np.float32)
    )

    node_scores, node_positions = index.node_index.search(
        query_vector, min(top_k_nodes, len(index.node_ids))
    )
    node_positions = node_positions[0]
    node_scores = node_scores[0]

    nodes = [
        {
            "node_id": index.node_ids[pos],
            "node_text": index.node_texts[pos],
            "score": float(node_scores[idx]),
        }
        for idx, pos in enumerate(node_positions)
    ]

    selected_node_ids = {node["node_id"] for node in nodes}
    selected_walks: list[dict[str, Any]] = []
    if index.walk_ids:
        walk_scores, walk_positions = index.walk_index.search(
            query_vector, len(index.walk_ids)
        )
        walk_scores = walk_scores[0]
        walk_positions = walk_positions[0]
        kept_per_node: dict[str, int] = defaultdict(int)
        for score, pos in zip(walk_scores, walk_positions):
            if pos < 0:
                continue
            start_node = index.walk_start_nodes[pos]
            if start_node not in selected_node_ids:
                continue
            if kept_per_node[start_node] >= top_k_walks_per_node:
                continue
            kept_per_node[start_node] += 1
            selected_walks.append(
                {
                    "walk_id": index.walk_ids[pos],
                    "start_node": start_node,
                    "text": index.walk_texts[pos],
                    "score": float(score),
                }
            )

    selected_walks.sort(key=lambda w: w["score"], reverse=True)

    # Exact-dedupe by text, keeping the higher-scoring occurrence.
    seen_texts: set[str] = set()
    deduped_walks: list[dict[str, Any]] = []
    for walk in selected_walks:
        if walk["text"] in seen_texts:
            continue
        seen_texts.add(walk["text"])
        deduped_walks.append(walk)

    context_parts: list[str] = []
    context_tokens = 0
    kept_walks: list[dict[str, Any]] = []
    for walk in deduped_walks:
        tokens = _token_count(walk["text"])
        if context_tokens + tokens > max_context_tokens and context_parts:
            break
        available = max_context_tokens - context_tokens
        part = _truncate_to_tokens(walk["text"], available)
        context_parts.append(part)
        context_tokens += _token_count(part)
        kept_walks.append(walk)

    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "nodes": nodes,
        "walks": kept_walks,
        "context": context,
        "context_tokens": context_tokens,
        "context_chars": len(context),
        "retrieved_walk_count": len(kept_walks),
    }
