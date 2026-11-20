"""Traditional RAG baseline: chunk regulations, index chunks, answer questions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from kgrag import config
from kgrag.ingest import load_source, parse_regulation
from kgrag.providers import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    ChatModel,
    Embedder,
    OllamaChat,
    OllamaEmbedder,
)
from kgrag.schemas import DocumentUnit
from kgrag.utils import embed_batch, l2_normalize, token_count, truncate_to_tokens
from kgrag.walk_retrieve import DEFAULT_ABSTAIN_TEXT

CHUNKER_VERSION = "clause-subtree-v1"

_INCLUDED_DESCENDANT_LEVELS = {"point", "subpoint", "table", "table_row"}


@dataclass
class Chunk:
    """A retrieval chunk built from one clause (or a clause-less article)."""

    chunk_id: str
    document_id: str
    level: str
    citation_label: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _children_map(units: list[DocumentUnit]) -> dict[str, list[DocumentUnit]]:
    children: dict[str, list[DocumentUnit]] = defaultdict(list)
    for unit in units:
        if unit.parent_id:
            children[unit.parent_id].append(unit)
    return children


def _render_unit(unit: DocumentUnit) -> str:
    body = unit.text.strip()
    if not body and unit.level == "table":
        body = str(unit.metadata.get("header", "")).strip()
    if not body:
        return ""
    label = unit.citation_label.strip()
    if label and not body.startswith(label):
        return f"{label}\n{body}"
    return body


def _descendant_parts(
    unit_id: str,
    children: dict[str, list[DocumentUnit]],
) -> list[str]:
    parts: list[str] = []
    for child in children.get(unit_id, []):
        if child.level in _INCLUDED_DESCENDANT_LEVELS:
            rendered = _render_unit(child)
            if rendered:
                parts.append(rendered)
        parts.extend(_descendant_parts(child.id, children))
    return parts


def _chunk_text(unit: DocumentUnit, children: dict[str, list[DocumentUnit]]) -> str:
    parts: list[str] = []
    label = unit.citation_label.strip()
    body = unit.context_text.strip() or unit.text.strip()
    if label:
        parts.append(label)
    if body:
        parts.append(body)
    parts.extend(_descendant_parts(unit.id, children))
    return "\n".join(parts)


def build_chunks(units: list[DocumentUnit]) -> list[Chunk]:
    """Build clause-subtree chunks, with article fallback for clause-less articles."""
    children = _children_map(units)
    chunks: list[Chunk] = []

    for unit in units:
        if unit.level not in ("clause", "article"):
            continue
        if unit.level == "article" and any(
            child.level == "clause" for child in children.get(unit.id, [])
        ):
            continue

        text = _chunk_text(unit, children)
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=unit.id,
                document_id=unit.document_id,
                level=unit.level,
                citation_label=unit.citation_label,
                text=text,
            )
        )

    return chunks


def load_units(
    source: str | Path | None = None,
    *,
    document_id: str = "quyche-2026",
) -> tuple[list[DocumentUnit], dict[str, Any]]:
    """Parse the regulation source and return units plus source metadata."""
    text, path, source_format, source_hash = load_source(source)
    result = parse_regulation(
        text,
        document_id=document_id,
        source_file=path.as_posix(),
        source_format=source_format,
        source_hash=source_hash,
    )
    meta = {
        "source_file": path.as_posix(),
        "source_hash": source_hash,
        "chunker_version": CHUNKER_VERSION,
        "unit_count": len(result.units),
    }
    return result.units, meta


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    """Persist chunks as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunks from JSONL."""
    path = Path(path)
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


@dataclass
class ChunkIndex:
    """In-memory representation of a built chunk index."""

    chunk_ids: list[str]
    texts: list[str]
    index: faiss.IndexFlatIP
    vectors: np.ndarray
    metadata: dict[str, Any]


def build_chunk_index(
    chunks: list[Chunk],
    embedder: Embedder,
    out_dir: str | Path,
    *,
    embed_model: str = "",
    source_hash: str = "",
    embed_batch_size: int = 16,
) -> ChunkIndex:
    """Embed chunks and persist a FAISS cosine index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    vectors = l2_normalize(
        embed_batch(texts, embedder, batch_size=embed_batch_size)
    )

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = {
        "dim": dim,
        "chunk_count": len(chunk_ids),
        "chunker_version": CHUNKER_VERSION,
        "embed_model": embed_model,
        "source_hash": source_hash,
    }

    faiss.write_index(index, str(out_dir / "chunks.faiss"))
    np.save(out_dir / "chunk_vectors.npy", vectors)
    with (out_dir / "chunk_ids.json").open("w", encoding="utf-8") as handle:
        json.dump(chunk_ids, handle, ensure_ascii=False, indent=2)
    with (out_dir / "chunk_texts.json").open("w", encoding="utf-8") as handle:
        json.dump(texts, handle, ensure_ascii=False, indent=2)
    with (out_dir / "index_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return ChunkIndex(
        chunk_ids=chunk_ids,
        texts=texts,
        index=index,
        vectors=vectors,
        metadata=metadata,
    )


def load_chunk_index(index_dir: str | Path) -> ChunkIndex:
    """Load a previously built chunk index."""
    index_dir = Path(index_dir)

    index = faiss.read_index(str(index_dir / "chunks.faiss"))
    vectors = np.load(index_dir / "chunk_vectors.npy")
    with (index_dir / "chunk_ids.json").open("r", encoding="utf-8") as handle:
        chunk_ids = json.load(handle)
    with (index_dir / "chunk_texts.json").open("r", encoding="utf-8") as handle:
        texts = json.load(handle)
    with (index_dir / "index_meta.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return ChunkIndex(
        chunk_ids=chunk_ids,
        texts=texts,
        index=index,
        vectors=vectors,
        metadata=metadata,
    )


def retrieve_chunks(
    query: str,
    index: ChunkIndex,
    embedder: Embedder,
    *,
    top_k: int = 5,
    max_context_tokens: int = 4096,
) -> dict[str, Any]:
    """Retrieve top-k chunks and assemble their context."""
    if not index.chunk_ids:
        return {
            "query": query,
            "chunks": [],
            "context": "",
            "context_tokens": 0,
            "context_chars": 0,
            "retrieved_chunk_count": 0,
        }

    query_vector = l2_normalize(
        np.asarray(embedder.embed([query]), dtype=np.float32)
    )
    scores, positions = index.index.search(
        query_vector, min(top_k, len(index.chunk_ids))
    )
    scores = scores[0]
    positions = positions[0]

    chunks: list[dict[str, Any]] = []
    for score, pos in zip(scores, positions):
        if pos < 0:
            continue
        chunks.append(
            {
                "chunk_id": index.chunk_ids[pos],
                "text": index.texts[pos],
                "score": float(score),
            }
        )

    context_parts: list[str] = []
    context_tokens = 0
    kept_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        tokens = token_count(chunk["text"])
        if context_tokens + tokens > max_context_tokens and context_parts:
            break
        available = max_context_tokens - context_tokens
        part = truncate_to_tokens(chunk["text"], available)
        context_parts.append(part)
        context_tokens += token_count(part)
        kept_chunks.append(chunk)

    context = "\n\n".join(context_parts)

    return {
        "query": query,
        "chunks": kept_chunks,
        "context": context,
        "context_tokens": context_tokens,
        "context_chars": len(context),
        "retrieved_chunk_count": len(kept_chunks),
    }


_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi về quy chế đào tạo đại học. "
    "Chỉ sử dụng thông tin trong NGỮ CẢNH dưới đây. "
    "Nếu ngữ cảnh không đủ để trả lời, hãy trả lời đúng: {abstain_text}. "
    "Trả lời ngắn gọn, nêu rõ điều/khoản/điểm nếu có."
)


def _build_prompt(
    question: str, context: str, abstain_text: str
) -> list[dict[str, str]]:
    system = _SYSTEM_PROMPT.format(abstain_text=abstain_text)
    user = f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}\n\nTrả lời:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _answer(
    question: str,
    context: str,
    chat: ChatModel,
    abstain_text: str,
) -> str:
    messages = _build_prompt(question, context, abstain_text)
    return chat.chat(messages, temperature=0.0, num_predict=512, seed=42)


def answer_question(
    question: str,
    index_dir: str | Path,
    embedder: Embedder,
    chat: ChatModel,
    *,
    top_k: int = 5,
    max_context_tokens: int = 4096,
    abstain_text: str = DEFAULT_ABSTAIN_TEXT,
) -> dict[str, Any]:
    """Answer a single question with one chat call over top-k chunks."""
    index = load_chunk_index(index_dir)
    result = retrieve_chunks(
        question,
        index,
        embedder,
        top_k=top_k,
        max_context_tokens=max_context_tokens,
    )

    if not result["chunks"]:
        return {
            "question": question,
            "answer": abstain_text,
            "abstained": True,
            "chunk_ids": [],
            "context": "",
            "context_tokens": 0,
            "context_chars": 0,
            "retrieved_chunk_scores": {},
        }

    answer = _answer(question, result["context"], chat, abstain_text)
    abstained = answer.strip() == abstain_text.strip()

    return {
        "question": question,
        "answer": answer.strip(),
        "abstained": abstained,
        "chunk_ids": [chunk["chunk_id"] for chunk in result["chunks"]],
        "context": result["context"],
        "context_tokens": result["context_tokens"],
        "context_chars": result["context_chars"],
        "retrieved_chunk_scores": {
            chunk["chunk_id"]: round(chunk["score"], 6)
            for chunk in result["chunks"]
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional RAG QA CLI")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build chunks and the chunk index from the regulation source",
    )
    parser.add_argument("--source", default=None, help="Path to source text")
    parser.add_argument("--document-id", default="quyche-2026", help="Document ID")
    parser.add_argument("--question", default=None, help="Single question to answer")
    parser.add_argument(
        "--questions-file",
        default=None,
        help="JSONL file with questions (one per line, field 'question')",
    )
    parser.add_argument(
        "--out",
        default="artifacts/traditional_rag/answers.jsonl",
        help="Output path for batch answers",
    )
    parser.add_argument(
        "--index-dir",
        default="artifacts/traditional_rag/index",
        help="Directory containing the chunk index",
    )
    parser.add_argument(
        "--chunks",
        default="artifacts/traditional_rag/chunks.jsonl",
        help="Path to the chunks JSONL",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k chunks")
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
        help="Maximum context tokens passed to the answer LLM",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Ollama embedding model",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help="Ollama chat model",
    )
    parser.add_argument(
        "--ollama-url",
        default=config.ollama_url(),
        help="Ollama chat base URL",
    )
    parser.add_argument(
        "--embed-url",
        default=config.ollama_embed_url(),
        help="Ollama embedding base URL",
    )
    parser.add_argument(
        "--abstain-text",
        default=DEFAULT_ABSTAIN_TEXT,
        help="Exact abstention string",
    )
    return parser.parse_args()


def main() -> None:
    config.load_env()
    args = _parse_args()

    embedder = OllamaEmbedder(
        model=args.embed_model,
        base_url=args.embed_url,
    )

    if args.build_index:
        units, meta = load_units(args.source, document_id=args.document_id)
        chunks = build_chunks(units)
        meta["chunk_count"] = len(chunks)
        meta["by_level"] = {
            level: sum(1 for chunk in chunks if chunk.level == level)
            for level in {chunk.level for chunk in chunks}
        }

        chunks_path = Path(args.chunks)
        save_chunks(chunks, chunks_path)
        with (chunks_path.parent / "chunk_meta.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)

        index = build_chunk_index(
            chunks,
            embedder,
            args.index_dir,
            embed_model=args.embed_model,
            source_hash=meta["source_hash"],
        )
        print(
            json.dumps(
                {
                    "chunks": len(index.chunk_ids),
                    "dim": index.metadata["dim"],
                    "index_dir": args.index_dir,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    questions: list[str] = []
    if args.question:
        questions.append(args.question)
    if args.questions_file:
        path = Path(args.questions_file)
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question = data.get("question") or data.get("text")
                if not isinstance(question, str) or not question.strip():
                    raise SystemExit(
                        f"{path}:{line_no}: missing non-empty 'question' or 'text'"
                    )
                questions.append(question.strip())

    if not questions:
        raise SystemExit("Provide --question or --questions-file")

    chat = OllamaChat(
        model=args.chat_model,
        base_url=args.ollama_url,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            result = answer_question(
                question,
                args.index_dir,
                embedder,
                chat,
                top_k=args.top_k,
                max_context_tokens=args.max_context_tokens,
                abstain_text=args.abstain_text,
            )
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
