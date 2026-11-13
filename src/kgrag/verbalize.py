"""Verbalize paper-style graph walks into natural-language sentences."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kgrag import config
from kgrag.providers import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_OLLAMA_URL,
    ChatModel,
    CloudflareChat,
    OllamaChat,
)
from kgrag.walks import Walk, load_walks


_VERBALIZATION_PROMPT_VERSION = "v4-header-content-5"

_EXAMPLE_INPUT = """1. (Điều 13 khoản 3 điểm c: Các điểm chữ I, X, R, M không tính vào điểm trung bình)
2. (references) (Điều 9: Tổ chức đăng ký học tập)
3. (contains) (Điều 9 khoản 4: Sinh viên phải đăng ký khối lượng học tập theo quy định)
4. (contains) (Điều 9 khoản 4 điểm a: Không học liên tiếp quá 03 học kỳ thì bị buộc thôi học)"""

_EXAMPLE_OUTPUT = "Các điểm chữ I, X, R, M không tính vào điểm trung bình; sinh viên phải đăng ký khối lượng học tập theo quy định, và không học liên tiếp quá 03 học kỳ thì bị buộc thôi học."

_SYSTEM_PROMPT = (
    "You are given multiple chains of nodes and relations from a Vietnamese university "
    "regulation knowledge graph. The citation path for each chain is provided separately "
    "and must not be repeated. Your task is to write ONLY the content of each chain as "
    "one natural Vietnamese sentence.\n\n"
    "Rules:\n"
    "- Do not include the citation path in your sentences.\n"
    "- Preserve all content and numbers from each chain.\n"
    "- Do not add any information not in the chains.\n"
    "- Never use the word \"chứa\".\n"
    "- Do not explain, do not reason step by step, and do not use tools.\n"
    "- Return JSON with key \"sentences\" containing exactly N sentences, in the same order as the N chains.\n\n"
    "Example:\n"
    f"Input:\n{_EXAMPLE_INPUT}\n\n"
    f'Output:\n{{"sentences": ["{_EXAMPLE_OUTPUT}"]}}'
)

_SYSTEM_PROMPT_SINGLE = (
    "You are given one chain of nodes and relations from a Vietnamese university "
    "regulation knowledge graph. The citation path is provided separately and must not "
    "be repeated. Your task is to write ONLY the content of the chain as one natural "
    "Vietnamese sentence.\n\n"
    "Rules:\n"
    "- Do not include the citation path in your sentence.\n"
    "- Preserve all content and numbers from the chain.\n"
    "- Do not add any information not in the chain.\n"
    "- Never use the word \"chứa\".\n"
    "- Do not explain, do not reason step by step, and do not use tools.\n"
    "- Return only the Vietnamese sentence.\n\n"
    "Example:\n"
    f"Input:\n{_EXAMPLE_INPUT}\n\n"
    f"Output:\n{_EXAMPLE_OUTPUT}"
)

_RELATION_LABEL = {
    "contains": "contains",
    "references": "references",
}


def _relation_label(relation: str) -> str:
    return _RELATION_LABEL.get(relation, relation)


@dataclass
class VerbalizationCache:
    """Resumable cache keyed by walk signature."""

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "VerbalizationCache":
        cache = cls(path=path)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    cache.records[record["signature"]] = record
        return cache

    def get(self, signature: str) -> dict[str, Any] | None:
        return self.records.get(signature)

    def add(self, record: dict[str, Any]) -> None:
        self.records[record["signature"]] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self.records.values():
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")


def _walk_signature(walk: Walk, model: str) -> str:
    """Stable signature for cache lookup. Model is included so swaps don't reuse old text."""
    payload = {
        "nodes": list(walk.nodes),
        "relations": list(walk.relations),
        "node_labels": list(walk.node_labels),
        "node_texts": list(walk.node_texts),
        "model": model,
        "prompt_version": _VERBALIZATION_PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _clean_node_text(text: str) -> str:
    """Make table rows readable and remove markdown pipe noise."""
    cleaned = text.strip()
    if "|" in cleaned:
        cleaned = cleaned.strip("|").strip()
        parts = [part.strip() for part in cleaned.split("|") if part.strip()]
        cleaned = ", ".join(parts)
    return cleaned


def _prepare_node_text(text: str, max_chars: int) -> str:
    """Clean and truncate a node text for the prompt."""
    return _truncate_text(_clean_node_text(text), max_chars)


def _render_walk(walk: Walk, max_node_text_chars: int) -> str:
    lines: list[str] = []
    for idx, node in enumerate(walk.nodes):
        label = walk.node_labels[idx]
        text = _prepare_node_text(walk.node_texts[idx], max_node_text_chars)
        if idx == 0:
            line = f"({label}: {text})"
        else:
            relation = _relation_label(walk.relations[idx - 1])
            line = f"({relation}) ({label}: {text})"
        lines.append(f"{idx + 1}. {line}")
    return "\n".join(lines)


def _render_batch(walks: Sequence[Walk], max_node_text_chars: int) -> str:
    parts: list[str] = []
    for idx, walk in enumerate(walks):
        parts.append(f"--- Walk {idx + 1} ---")
        parts.append(_render_walk(walk, max_node_text_chars))
    return "\n".join(parts)


def _json_schema_for_n(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n,
                "maxItems": n,
            }
        },
        "required": ["sentences"],
    }


def _extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,/]\d+)*%?", text))


def _build_citation_header(walk: Walk) -> str:
    """Build a canonical citation header from node labels, grouping by relation type."""
    if not walk.node_labels:
        return ""
    groups: list[list[str]] = []
    current: list[str] = [walk.node_labels[0]]
    for relation, label in zip(walk.relations, walk.node_labels[1:]):
        if relation == "references":
            groups.append(current)
            current = [label]
        else:
            current.append(label)
    groups.append(current)

    deduped_groups: list[list[str]] = []
    for group in groups:
        seen: list[str] = []
        for part in group:
            if part and part not in seen:
                seen.append(part)
        if seen:
            deduped_groups.append(seen)

    return "; ".join(", ".join(group) for group in deduped_groups
    )


def _check_content_fidelity(
    walk: Walk, verbalization: str, max_node_text_chars: int
) -> tuple[bool, list[str], list[str]]:
    """Check fidelity only for content numbers, ignoring structural citation numbers."""
    structural_numbers: set[str] = set()
    for label in walk.node_labels:
        structural_numbers.update(_extract_numbers(label))

    source_text = " ".join(
        _prepare_node_text(text, max_node_text_chars) for text in walk.node_texts
    )
    source_numbers = _extract_numbers(source_text) - structural_numbers
    output_numbers = _extract_numbers(verbalization) - structural_numbers
    dropped = sorted(source_numbers - output_numbers)
    added = sorted(output_numbers - source_numbers)
    return (not dropped and not added), dropped, added


def _check_fidelity(
    walk: Walk, verbalization: str, max_node_text_chars: int
) -> tuple[bool, list[str], list[str]]:
    source_parts: list[str] = []
    for label, text in zip(walk.node_labels, walk.node_texts):
        if label:
            source_parts.append(label)
        source_parts.append(_truncate_text(text, max_node_text_chars))
    source_text = " ".join(source_parts)
    source_numbers = _extract_numbers(source_text)
    output_numbers = _extract_numbers(verbalization)
    dropped = sorted(source_numbers - output_numbers)
    added = sorted(output_numbers - source_numbers)
    return (not dropped and not added), dropped, added


def _record_for_walk(
    walk: Walk,
    text: str,
    model: str,
    max_node_text_chars: int,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = text.strip().strip('"').strip("'")
    ok, dropped, added = _check_content_fidelity(
        walk, content, max_node_text_chars
    )
    header = _build_citation_header(walk)
    final_text = f"{header}\n{content}" if header else content
    record: dict[str, Any] = {
        "signature": _walk_signature(walk, model),
        "text": final_text,
        "content": content,
        "header": header,
        "model": model,
        "prompt_version": _VERBALIZATION_PROMPT_VERSION,
        "fidelity_ok": ok,
        "dropped_numbers": dropped,
        "added_numbers": added,
        "graph_hash": walk.graph_hash,
    }
    if usage:
        record["usage"] = usage
    return record


def _parse_sentences(text: str, expected: int) -> list[str] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
        sentences = parsed["sentences"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not isinstance(sentences, list) or len(sentences) != expected:
        return None
    return [str(s).strip() for s in sentences]


def _verbalize_batch(
    walks: Sequence[Walk],
    chat: ChatModel,
    model: str,
    max_node_text_chars: int,
) -> list[dict[str, Any]]:
    if not walks:
        return []

    def _last_usage() -> dict[str, Any] | None:
        log = getattr(chat, "usage_log", None)
        return log[-1] if log else None

    if len(walks) == 1:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_SINGLE},
            {
                "role": "user",
                "content": _render_walk(walks[0], max_node_text_chars)
                + "\n\nWrite the content as one Vietnamese sentence:",
            },
        ]
        text = chat.chat(messages, temperature=0.0, num_predict=256, seed=42)
        return [_record_for_walk(walks[0], text, model, max_node_text_chars, _last_usage())]

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _render_batch(walks, max_node_text_chars)
            + "\n\nWrite the content of each walk as one Vietnamese sentence and return JSON:",
        },
    ]
    schema = _json_schema_for_n(len(walks))
    try:
        text = chat.chat(
            messages,
            temperature=0.0,
            num_predict=256 * len(walks),
            seed=42,
            format=schema,
        )
    except RuntimeError:
        # Model refused/returned no content for this batch size; split and retry.
        if len(walks) > 1:
            mid = len(walks) // 2
            left = _verbalize_batch(walks[:mid], chat, model, max_node_text_chars)
            right = _verbalize_batch(walks[mid:], chat, model, max_node_text_chars)
            return left + right
        raise

    sentences = _parse_sentences(text, len(walks))

    if sentences is not None:
        usage = _last_usage()
        return [
            _record_for_walk(walk, sentence, model, max_node_text_chars, usage)
            for walk, sentence in zip(walks, sentences)
        ]

    # Fallback: split batch in half rather than individual calls to save quota.
    if len(walks) > 1:
        mid = len(walks) // 2
        left = _verbalize_batch(walks[:mid], chat, model, max_node_text_chars)
        right = _verbalize_batch(walks[mid:], chat, model, max_node_text_chars)
        return left + right

    # Should never reach here, but guard against infinite recursion.
    return [_record_for_walk(walks[0], text, model, max_node_text_chars)]


def _trivial_verbalization(walk: Walk, model: str, max_node_text_chars: int) -> dict[str, Any]:
    """For length-0 walks, use the node text as content; header is prepended separately."""
    text = _prepare_node_text(walk.node_texts[0], max_node_text_chars) if walk.node_texts else ""
    return _record_for_walk(walk, text, model, max_node_text_chars)


def verbalize_walks(
    walks: list[Walk],
    chat: ChatModel,
    *,
    cache: VerbalizationCache | None = None,
    workers: int = 1,
    batch_size: int = 8,
    limit: int | None = None,
    max_node_text_chars: int = 400,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return a verbalization record for every walk, using and updating *cache*."""
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    model = getattr(chat, "model", DEFAULT_CHAT_MODEL)

    unique_walks: list[Walk] = []
    seen: set[str] = set()
    for walk in walks:
        sig = _walk_signature(walk, model)
        if sig in seen:
            continue
        seen.add(sig)
        unique_walks.append(walk)

    if limit is not None:
        unique_walks = unique_walks[:limit]

    cache = cache or VerbalizationCache(path=Path("/dev/null"))
    results_by_signature: dict[str, dict[str, Any]] = {}
    missing: list[Walk] = []

    for walk in unique_walks:
        sig = _walk_signature(walk, model)
        cached = cache.get(sig)
        if cached is not None:
            results_by_signature[sig] = cached
        else:
            missing.append(walk)

    # Separate trivial length-0 walks; don't waste LLM calls on them.
    trivial: list[Walk] = [w for w in missing if w.depth == 0]
    llm_missing: list[Walk] = [w for w in missing if w.depth > 0]

    for walk in trivial:
        record = _trivial_verbalization(walk, model, max_node_text_chars)
        cache.add(record)
        results_by_signature[record["signature"]] = record

    batches: list[list[Walk]] = []
    current: list[Walk] = []
    for walk in llm_missing:
        current.append(walk)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    def _task(batch: list[Walk]) -> list[dict[str, Any]]:
        return _verbalize_batch(batch, chat, model, max_node_text_chars)

    if workers <= 1:
        iterator = batches
        if progress and tqdm is not None:
            iterator = tqdm(batches, desc="Verbalizing")
        for batch in iterator:
            for record in _task(batch):
                cache.add(record)
                results_by_signature[record["signature"]] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_task, b): b for b in batches}
            if progress and tqdm is not None:
                iterator = tqdm(futures.items(), total=len(futures), desc="Verbalizing")
            else:
                iterator = futures.items()
            for future, _batch in iterator:
                for record in future.result():
                    cache.add(record)
                    results_by_signature[record["signature"]] = record

    ordered_records = [
        results_by_signature[_walk_signature(walk, model)] for walk in unique_walks
    ]
    fidelity_counts = Counter(r["fidelity_ok"] for r in ordered_records)
    stats = {
        "total_unique": len(unique_walks),
        "cached": len(unique_walks) - len(missing),
        "newly_verbalized": len(missing),
        "trivial": len(trivial),
        "llm_calls": len(batches) if batch_size else len(llm_missing),
        "fidelity_ok": fidelity_counts.get(True, 0),
        "fidelity_failed": fidelity_counts.get(False, 0),
    }
    return ordered_records, stats


def build_walk_verbalization_map(
    walks: list[Walk],
    cache: VerbalizationCache,
    model: str = DEFAULT_CHAT_MODEL,
) -> dict[str, str]:
    """Return {walk_id: verbalized_text} using the cache."""
    mapping: dict[str, str] = {}
    for walk in walks:
        sig = _walk_signature(walk, model)
        record = cache.get(sig)
        if record is not None:
            mapping[walk.walk_id] = record["text"]
    return mapping


def _default_chat_template_kwargs(model: str) -> dict[str, Any] | None:
    """Return default chat_template_kwargs for known Cloudflare models."""
    lower = model.lower()
    if "glm" in lower or "gemma-4" in lower:
        return {"enable_thinking": False}
    return None


def _build_chat(args: argparse.Namespace) -> ChatModel:
    if args.provider == "ollama":
        return OllamaChat(
            model=args.chat_model,
            base_url=args.ollama_url,
        )
    if args.provider == "cloudflare":
        account_id = args.cloudflare_account_id or config.cloudflare_account_id()
        api_token = args.cloudflare_api_token or config.cloudflare_api_token()
        base_url = args.cloudflare_base_url or config.cloudflare_base_url()
        if not account_id or not api_token or not base_url:
            raise SystemExit(
                "Cloudflare provider requires CLOUDFLARE_ACCOUNT_ID, "
                "CLOUDFLARE_API_TOKEN, and a base URL."
            )
        chat_template_kwargs = args.chat_template_kwargs
        if chat_template_kwargs is None:
            chat_template_kwargs = _default_chat_template_kwargs(args.chat_model)
        return CloudflareChat(
            model=args.chat_model,
            account_id=account_id,
            api_token=api_token,
            base_url=base_url,
            timeout=args.cloudflare_timeout,
            chat_template_kwargs=chat_template_kwargs,
        )
    raise SystemExit(f"Unknown provider: {args.provider}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verbalize graph walks via a local or cloud LLM"
    )
    parser.add_argument(
        "--walks",
        default="artifacts/kg/walks.jsonl",
        help="Path to walks.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/kg",
        help="Output directory",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "cloudflare"],
        default="cloudflare",
        help="LLM provider for verbalization",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model for verbalization (default: provider-specific)",
    )
    parser.add_argument(
        "--ollama-url",
        default=config.ollama_url(),
        help="Ollama base URL",
    )
    parser.add_argument(
        "--cloudflare-account-id",
        default=None,
        help="Cloudflare account ID",
    )
    parser.add_argument(
        "--cloudflare-api-token",
        default=None,
        help="Cloudflare API token",
    )
    parser.add_argument(
        "--cloudflare-base-url",
        default=None,
        help="Cloudflare OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--cloudflare-timeout",
        type=float,
        default=120.0,
        help="Cloudflare request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Verbalize only the first N unique walks (smoke test)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel verbalization workers",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Walks per LLM prompt",
    )
    parser.add_argument(
        "--max-node-text-chars",
        type=int,
        default=400,
        help="Truncate node texts to this many chars in the prompt",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        help='JSON object passed as chat_template_kwargs (e.g. {"enable_thinking": false})',
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config.load_env()

    if args.chat_model is None:
        if args.provider == "cloudflare":
            args.chat_model = config.cloudflare_chat_model()
        else:
            args.chat_model = config.ollama_chat_model()

    walks = load_walks(args.walks)
    cache_path = Path(args.out_dir) / "verbalizations.jsonl"
    cache = VerbalizationCache.load(cache_path)

    chat = _build_chat(args)

    records, stats = verbalize_walks(
        walks,
        chat,
        cache=cache,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        max_node_text_chars=args.max_node_text_chars,
        progress=True,
    )
    cache.save()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "verbalize_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
