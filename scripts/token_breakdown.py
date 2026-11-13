"""Compute token breakdown for GLM+header-v2."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("artifacts/kg/bakeoff/glm-4.7-flash-header-v2/verbalizations.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt_total = sum(r["usage"]["prompt_tokens"] for r in records if r.get("usage"))
    completion_total = sum(r["usage"]["completion_tokens"] for r in records if r.get("usage"))
    n = len(records)
    print("avg prompt tokens:", prompt_total / n)
    print("avg completion tokens:", completion_total / n)
    print("avg total tokens:", (prompt_total + completion_total) / n)


if __name__ == "__main__":
    main()
