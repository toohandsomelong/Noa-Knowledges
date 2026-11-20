"""Resumable cache for final Walk&Retrieve answers."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnswerCache:
    """Question-keyed cache where the last record in the file wins."""

    path: Path
    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "AnswerCache":
        cache = cls(path=path)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        warnings.warn(
                            f"{path}:{line_no}: skipping malformed cache line",
                            stacklevel=2,
                        )
                        continue
                    question = record.get("question")
                    if question:
                        cache.records[question] = record
        return cache

    def get(self, question: str) -> dict[str, Any] | None:
        return self.records.get(question)

    def add(self, result: dict[str, Any]) -> None:
        self.records[result["question"]] = result
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
