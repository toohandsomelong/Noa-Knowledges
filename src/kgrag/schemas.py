"""Data schemas for parsed regulation units."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    """Location of a unit inside the source text."""

    start_line: int
    end_line: int
    start_char: int = 0
    end_char: int = 0


@dataclass
class DocumentUnit:
    """A single hierarchical unit from the regulation."""

    id: str
    document_id: str
    level: str
    parent_id: str | None
    title: str
    text: str
    context_text: str
    order: int
    source_span: SourceSpan
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a regulation source."""

    document_id: str
    units: list[DocumentUnit]
    metadata: dict[str, Any] = field(default_factory=dict)
