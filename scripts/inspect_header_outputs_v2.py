"""Inspect GLM+header-v2 verbalization outputs."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("artifacts/kg/bakeoff/glm-4.7-flash-header-v2/verbalizations.jsonl")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_neurons = 0.0
    for record in records:
        usage = record.get("usage")
        if usage:
            total_neurons += float(usage.get("neurons", 0))
    print("records:", len(records))
    print("fidelity_ok:", sum(r["fidelity_ok"] for r in records))
    print("total neurons:", total_neurons)
    print("avg per walk:", total_neurons / len(records))
    print("est full run:", total_neurons / len(records) * 2038)
    print()
    for i, record in enumerate(records[:5], 1):
        print(f"--- {i} ---")
        print("header:", record["header"])
        print("content:", record["content"])
        print()


if __name__ == "__main__":
    main()
