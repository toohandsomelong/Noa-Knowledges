"""Data schemas for parsed regulation units."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceSpan:
    """Location of a unit inside the source text."""

    start_line: int
    end_line: int
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class Reference:
    """A structured cross-reference inside a unit."""

    kind: str
    target: str
    text: str


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
    source_file: str = ""
    source_format: str = ""
    source_hash: str = ""
    raw_text: str = ""
    citation_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParseResult:
    """Result of parsing a regulation source."""

    document_id: str
    units: list[DocumentUnit]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "metadata": self.metadata,
            "units": [unit.to_dict() for unit in self.units],
        }
