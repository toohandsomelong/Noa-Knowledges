"""Walk&Retrieve CLI: build index and answer questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kgrag.providers import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, ChatModel, Embedder, OllamaChat, OllamaEmbedder
from kgrag.retrieval import build_index, load_index, retrieve
from kgrag.verbalize import VerbalizationCache, build_walk_verbalization_map
from kgrag.walks import load_walks


DEFAULT_ABSTAIN_TEXT = "Tôi không biết câu trả lời"

_SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi về quy chế đào tạo đại học. "
    "Chỉ sử dụng thông tin trong NGỮ CẢNH dưới đây. "
    "Nếu ngữ cảnh không đủ để trả lời, hãy trả lời đúng: {abstain_text}. "
    "Trả lời ngắn gọn, nêu rõ điều/khoản/điểm nếu có."
)


def _build_prompt(question: str, context: str, abstain_text: str) -> list[dict[str, str]]:
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
    top_k_nodes: int = 3,
    top_k_walks_per_node: int = 3,
    max_context_tokens: int = 4096,
    abstain_text: str = DEFAULT_ABSTAIN_TEXT,
) -> dict[str, Any]:
    """Answer a single question using Walk&Retrieve."""
    index = load_index(index_dir)
    result = retrieve(
        question,
        index,
        embedder,
        top_k_nodes=top_k_nodes,
        top_k_walks_per_node=top_k_walks_per_node,
        max_context_tokens=max_context_tokens,
    )

    if not result["walks"]:
        return {
            "question": question,
            "answer": abstain_text,
            "abstained": True,
            "node_ids": [n["node_id"] for n in result["nodes"]],
            "walk_ids": [],
            "context": "",
            "context_tokens": 0,
            "context_chars": 0,
            "retrieved_walk_scores": {},
        }

    answer = _answer(question, result["context"], chat, abstain_text)
    abstained = answer.strip() == abstain_text.strip()

    return {
        "question": question,
        "answer": answer.strip(),
        "abstained": abstained,
        "node_ids": [n["node_id"] for n in result["nodes"]],
        "walk_ids": [w["walk_id"] for w in result["walks"]],
        "context": result["context"],
        "context_tokens": result["context_tokens"],
        "context_chars": result["context_chars"],
        "retrieved_walk_scores": {
            w["walk_id"]: round(w["score"], 6) for w in result["walks"]
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk&Retrieve QA CLI")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build the retrieval index from walks + verbalizations",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Single question to answer",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="JSONL file with questions (one per line, field 'question')",
    )
    parser.add_argument(
        "--out",
        default="artifacts/kg/answers.jsonl",
        help="Output path for batch answers",
    )
    parser.add_argument(
        "--index-dir",
        default="artifacts/kg/index",
        help="Directory containing the FAISS index",
    )
    parser.add_argument(
        "--walks",
        default="artifacts/kg/walks.jsonl",
        help="Path to walks.jsonl",
    )
    parser.add_argument(
        "--verbalizations",
        default="artifacts/kg/verbalizations.jsonl",
        help="Path to verbalizations cache",
    )
    parser.add_argument(
        "--verbalization-model",
        default=DEFAULT_CHAT_MODEL,
        help="Model key used in the verbalization cache",
    )
    parser.add_argument(
        "--top-k-nodes",
        type=int,
        default=3,
        help="Top-k nodes to retrieve",
    )
    parser.add_argument(
        "--top-k-walks-per-node",
        type=int,
        default=3,
        help="Top-k walks per retrieved node",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=4096,
        help="Maximum context tokens passed to the answer LLM",
    )
    parser.add_argument(
        "--max-node-desc-chars",
        type=int,
        default=6000,
        help="Maximum characters for node descriptions",
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
        default=DEFAULT_OLLAMA_URL,
        help="Ollama base URL",
    )
    parser.add_argument(
        "--abstain-text",
        default=DEFAULT_ABSTAIN_TEXT,
        help="Exact abstention string",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    embedder = OllamaEmbedder(
        model=args.embed_model,
        base_url=args.ollama_url,
    )
    chat = OllamaChat(
        model=args.chat_model,
        base_url=args.ollama_url,
    )

    if args.build_index:
        walks = load_walks(args.walks)
        cache = VerbalizationCache.load(Path(args.verbalizations))
        verbalization_map = build_walk_verbalization_map(
            walks, cache, model=args.verbalization_model
        )
        index = build_index(
            walks,
            verbalization_map,
            embedder,
            args.index_dir,
            max_node_desc_chars=args.max_node_desc_chars,
        )
        print(
            json.dumps(
                {
                    "nodes": len(index.node_ids),
                    "walks": len(index.walk_ids),
                    "dim": index.metadata["dim"],
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
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                questions.append(data.get("question", data.get("text", "")))

    if not questions:
        raise SystemExit("Provide --question or --questions-file")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            result = answer_question(
                question,
                args.index_dir,
                embedder,
                chat,
                top_k_nodes=args.top_k_nodes,
                top_k_walks_per_node=args.top_k_walks_per_node,
                max_context_tokens=args.max_context_tokens,
                abstain_text=args.abstain_text,
            )
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
