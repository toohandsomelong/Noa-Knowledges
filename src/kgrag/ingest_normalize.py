"""Normalize source text to a canonical line model before parsing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedLine:
    """A single source line with its original and normalized forms."""

    line_no: int
    original: str
    text: str


def normalize_text(text: str) -> list[NormalizedLine]:
    """Convert raw source text into normalized lines.

    Removes Markdown heading markers, bold/italic wrappers, and normalizes
    whitespace while preserving table rows and bullet markers.
    """
    lines: list[NormalizedLine] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = unicodedata.normalize("NFC", raw)
        line = line.replace("\t", " ")
        line = re.sub(r" {2,}", " ", line)
        line = line.strip()

        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)

        line = line.strip()
        lines.append(NormalizedLine(line_no=line_no, original=raw, text=line))
    return lines
