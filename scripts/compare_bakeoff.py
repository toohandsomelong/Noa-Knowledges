"""Compare 20-walk verbalization bake-off results and write a report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from kgrag.verbalize import _walk_signature, _check_fidelity
from kgrag.walks import Walk, load_walks


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _total_neurons(records: list[dict[str, Any]]) -> float:
    total = 0.0
    seen: set[int] = set()
    for record in records:
        usage = record.get("usage")
        if not usage:
            continue
        key = id(usage)
        if key in seen:
            continue
        seen.add(key)
        neurons = usage.get("neurons")
        if isinstance(neurons, (int, float)):
            total += float(neurons)
    return total


def _avg_text_length(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(len(r["text"]) for r in records) / len(records)


def _duplicate_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    texts = [r["text"] for r in records]
    unique = len(set(texts))
    return (len(texts) - unique) / len(texts)


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: compare_bakeoff.py <walks.jsonl> <model1:dir> <model2:dir> ...")
        raise SystemExit(1)

    walks_path = Path(sys.argv[1])
    out_path = Path("artifacts/kg/bakeoff/comparison_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_specs = []
    for arg in sys.argv[2:]:
        name, _, directory = arg.partition(":")
        model_specs.append((name, Path(directory)))

    walks = load_walks(str(walks_path))

    lines: list[str] = []
    lines.append("# 20-walk verbalization bake-off report\n")
    lines.append("| Model | Fidelity OK | Avg dropped | Avg added | Avg chars | Duplicates | Total neurons |")
    lines.append("|-------|-------------|-------------|-----------|-----------|------------|---------------|")

    all_metrics: list[tuple[str, list[dict[str, Any]]]] = []
    for name, directory in model_specs:
        records = _load_records(directory / "verbalizations.jsonl")
        all_metrics.append((name, records))
        walk_by_sig: dict[str, Walk] = {}
        for walk in walks:
            sig = _walk_signature(walk, records[0]["model"]) if records else ""
            walk_by_sig[sig] = walk

        total_dropped = 0
        total_added = 0
        fidelity_ok = 0
        for record in records:
            walk = walk_by_sig.get(record["signature"])
            if walk is None:
                continue
            ok, dropped, added = _check_fidelity(walk, record["text"], 400)
            if ok:
                fidelity_ok += 1
            total_dropped += len(dropped)
            total_added += len(added)
        n = len(records)
        avg_dropped = total_dropped / n if n else 0
        avg_added = total_added / n if n else 0
        avg_chars = _avg_text_length(records)
        dup_rate = _duplicate_rate(records)
        neurons = _total_neurons(records)
        lines.append(
            f"| {name} | {fidelity_ok}/{n} | {avg_dropped:.2f} | {avg_added:.2f} | "
            f"{avg_chars:.0f} | {dup_rate:.0%} | {neurons:.2f} |"
        )

    lines.append("\n## Sample outputs (first 3 per model)\n")
    for name, records in all_metrics:
        lines.append(f"### {name}\n")
        for i, record in enumerate(records[:3], 1):
            lines.append(f"{i}. {record['text']}\n")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
