"""Find the walk corresponding to a failing GLM+header record."""

from __future__ import annotations

import json
from pathlib import Path

from kgrag.verbalize import _walk_signature
from kgrag.walks import load_walks


def main() -> None:
    records = [
        json.loads(line)
        for line in Path("artifacts/kg/bakeoff/glm-4.7-flash-header/verbalizations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    fail = next((r for r in records if not r["fidelity_ok"]), None)
    if fail is None:
        print("no failure")
        return
    print("fail signature:", fail["signature"])
    walks = load_walks("artifacts/kg/bakeoff/sample_walks.jsonl")
    for walk in walks:
        sig = _walk_signature(walk, fail["model"])
        if sig == fail["signature"]:
            print("MATCH")
            print("nodes:", walk.nodes)
            print("labels:", walk.node_labels)
            print("relations:", walk.relations)
            print("texts:", [t[:200] for t in walk.node_texts])
            print("header:", fail["header"])
            print("content:", fail["content"])
            print("dropped:", fail["dropped_numbers"])
            break


if __name__ == "__main__":
    main()
