"""Create a deterministic sample of non-trivial walks for the bake-off."""

from __future__ import annotations

import json
import random
from pathlib import Path

from kgrag.walks import load_walks


def main() -> None:
    walks_path = Path("artifacts/kg/walks.jsonl")
    out_path = Path("artifacts/kg/bakeoff/sample_walks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    walks = load_walks(str(walks_path))
    non_trivial = [w for w in walks if w.depth > 0]

    rng = random.Random(42)
    sample = rng.sample(non_trivial, min(20, len(non_trivial)))

    with out_path.open("w", encoding="utf-8") as handle:
        for walk in sample:
            handle.write(json.dumps(walk.to_dict(), ensure_ascii=False))
            handle.write("\n")

    print(f"Wrote {len(sample)} walks to {out_path} (from {len(non_trivial)} non-trivial)")


if __name__ == "__main__":
    main()
