"""Parse the Vietnamese regulation source into hierarchical DocumentUnits."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from kgrag.schemas import DocumentUnit, ParseResult, SourceSpan

_CHAPTER_RE = re.compile(r"^Chương\s+([IVXLCDM]+)\s*$")
_ARTICLE_RE = re.compile(r"^Điều\s+(\d+[a-zA-Z]*)\.\s*(.*)$")
_CLAUSE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
_POINT_RE = re.compile(r"^([a-z])\)\s+(.*)$")
_DASH_RE = re.compile(r"^[-–]\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
_TABLE_SEP_RE = re.compile(r"^\|[-:\s|]+\|$")
_CROSSREF_DIEU_RE = re.compile(r"(?:Điều|điều)\s+(\d+[a-zA-Z]*)")
_CROSSREF_KHOAN_RE = re.compile(r"(?:Khoản|khoản)\s+(\d+)")
_CROSSREF_DIEM_RE = re.compile(r"(?:Điểm|điểm)\s+([a-z])\)")

_LEVEL_RANK = {
    "document": 0,
    "chapter": 1,
    "article": 2,
    "clause": 3,
    "point": 4,
    "subpoint": 5,
    "table": 6,
    "table_row": 7,
}


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"\s+", "-", value)
    return value.strip("-")


def _find_default_source() -> Path | None:
    env_path = os.getenv("SOURCE_TEXT")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    root = Path(__file__).parents[2]
    search_dirs = [
        root / "data" / "docs",
        root / "sources" / "quyche",
    ]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        text_files = sorted(directory.glob("*.txt"))
        if len(text_files) == 1:
            return text_files[0]
    return None


def load_source_text(path: str | Path | None = None) -> str:
    """Load regulation source text from *path* or a discoverable default."""
    if path is None:
        candidate = _find_default_source()
        if candidate is None:
            raise FileNotFoundError(
                "Could not discover regulation source. "
                "Set SOURCE_TEXT or pass an explicit path."
            )
        path = candidate
    path = Path(path)
    return path.read_text(encoding="utf-8")


def _extract_crossrefs(text: str) -> dict[str, list[str]]:
    return {
        "dieu": _CROSSREF_DIEU_RE.findall(text),
        "khoan": _CROSSREF_KHOAN_RE.findall(text),
        "diem": _CROSSREF_DIEM_RE.findall(text),
    }


def parse_regulation(
    text: str,
    *,
    document_id: str = "quyche-2026",
    source_path: str = "",
    article_ids: set[str] | None = None,
    include_chapters: bool = True,
) -> ParseResult:
    """Parse *text* into a hierarchical list of DocumentUnits.

    Args:
        text: Full source text.
        document_id: Identifier for the whole document.
        source_path: Optional source file path for provenance.
        article_ids: If given, keep only units under these article numbers
            (e.g. {"13", "18"}). Chapter ancestors are kept for context.
        include_chapters: If True, include chapter units in the output.
    """
    lines = text.splitlines()
    units: list[DocumentUnit] = []
    stack: list[DocumentUnit] = []
    orders: dict[tuple[str | None, str], int] = defaultdict(int)

    current_prefix = "decision"
    pending_chapter: dict[str, Any] | None = None
    in_table = False
    table_unit: DocumentUnit | None = None
    preamble_lines: list[str] = []
    preamble_span: SourceSpan | None = None

    def _next_order(parent_id: str | None, level: str) -> int:
        key = (parent_id, level)
        orders[key] += 1
        return orders[key]

    def _pop_to(rank: int) -> DocumentUnit | None:
        while stack and _LEVEL_RANK[stack[-1].level] >= rank:
            stack.pop()
        return stack[-1] if stack else None

    def _close_table() -> None:
        nonlocal in_table, table_unit
        while stack and stack[-1].level in ("table", "table_row"):
            stack.pop()
        in_table = False
        table_unit = None

    def _add_unit(
        level: str,
        title: str,
        text: str,
        line_index: int,
        parent: DocumentUnit | None = None,
        extra_id_parts: Sequence[str] = (),
        level_meta: dict[str, Any] | None = None,
    ) -> DocumentUnit:
        nonlocal in_table, table_unit

        if in_table and level not in ("table", "table_row"):
            _close_table()

        if parent is None:
            parent = _pop_to(_LEVEL_RANK[level])
        parent_id = parent.id if parent else None
        order = _next_order(parent_id, level)

        id_parts = [current_prefix]
        if parent:
            id_parts = parent.id.split(":")
        id_parts = list(id_parts)
        id_parts.extend(extra_id_parts)
        unit_id = ":".join(id_parts)

        span = SourceSpan(
            start_line=line_index + 1,
            end_line=line_index + 1,
            start_char=0,
            end_char=len(lines[line_index]) if line_index < len(lines) else 0,
        )
        meta: dict[str, Any] = {}
        if level_meta:
            meta.update(level_meta)
        unit = DocumentUnit(
            id=unit_id,
            document_id=document_id,
            level=level,
            parent_id=parent_id,
            title=title,
            text=text,
            context_text="",
            order=order,
            source_span=span,
            metadata=meta,
        )
        units.append(unit)
        stack.append(unit)
        return unit

    def _append_to_top(text: str, line_index: int) -> None:
        nonlocal in_table, table_unit
        if not stack:
            preamble_lines.append(text)
            return

        if stack[-1].level in ("table", "table_row"):
            _close_table()

        if not stack:
            preamble_lines.append(text)
            return

        unit = stack[-1]
        unit.text = f"{unit.text} {text}".strip()
        unit.source_span = SourceSpan(
            start_line=unit.source_span.start_line,
            end_line=line_index + 1,
            start_char=unit.source_span.start_char,
            end_char=len(lines[line_index]) if line_index < len(lines) else 0,
        )

    def _flush_preamble() -> None:
        if not preamble_lines:
            return
        text = " ".join(preamble_lines).strip()
        span = preamble_span or SourceSpan(1, 1)
        units.append(
            DocumentUnit(
                id=f"{current_prefix}:preamble",
                document_id=document_id,
                level="preamble",
                parent_id=None,
                title="Preamble",
                text=text,
                context_text=text,
                order=_next_order(None, "preamble"),
                source_span=span,
                metadata={},
            )
        )
        preamble_lines.clear()

    for idx, raw in enumerate(lines):
        line = raw.rstrip()
        stripped = line.strip()

        if preamble_span is None and not stripped:
            continue
        if preamble_span is None and stripped:
            preamble_span = SourceSpan(idx + 1, idx + 1)

        if stripped == "QUY CHẾ":
            _flush_preamble()
            _close_table()
            stack.clear()
            current_prefix = "regulation"
            continue

        if not stripped:
            continue

        chapter_match = _CHAPTER_RE.match(stripped)
        if chapter_match:
            _flush_preamble()
            pending_chapter = {
                "number": chapter_match.group(1),
                "title_lines": [],
                "start_line": idx + 1,
            }
            continue

        if pending_chapter is not None:
            pending_chapter["title_lines"].append(stripped)
            if len(pending_chapter["title_lines"]) >= 1 and (
                _ARTICLE_RE.match(stripped)
                or _CHAPTER_RE.match(stripped)
                or idx + 1 >= len(lines)
            ):
                pending_chapter["title_lines"].pop()
            if pending_chapter["title_lines"]:
                title = " ".join(pending_chapter["title_lines"])
                chapter_id_part = f"chuong-{_slug(pending_chapter['number'])}"
                _pop_to(_LEVEL_RANK["chapter"])
                parent = stack[-1] if stack else None
                _add_unit(
                    level="chapter",
                    title=title,
                    text=title,
                    line_index=pending_chapter["start_line"] - 1,
                    parent=parent,
                    extra_id_parts=[chapter_id_part],
                    level_meta={"chapter_number": pending_chapter["number"]},
                )
                pending_chapter = None
            continue

        article_match = _ARTICLE_RE.match(stripped)
        if article_match:
            _flush_preamble()
            _close_table()
            article_number = article_match.group(1)
            title = article_match.group(2).strip()
            if current_prefix == "regulation":
                _pop_to(_LEVEL_RANK["article"])
            else:
                stack.clear()
            parent = stack[-1] if stack else None
            _add_unit(
                level="article",
                title=title,
                text=title,
                line_index=idx,
                parent=parent,
                extra_id_parts=[f"dieu-{_slug(article_number)}"],
                level_meta={"article_number": article_number},
            )
            continue

        if in_table or _TABLE_ROW_RE.match(stripped):
            if _TABLE_SEP_RE.match(stripped):
                continue
            if not _TABLE_ROW_RE.match(stripped):
                # A non-table line ends the current table; let normal parsing handle it.
                _close_table()
            else:
                if not in_table:
                    _close_table()
                    parent = _pop_to(_LEVEL_RANK["table"])
                    table_order = _next_order(parent.id if parent else None, "table")
                    table_unit = _add_unit(
                        level="table",
                        title="",
                        text="",
                        line_index=idx,
                        parent=parent,
                        extra_id_parts=[f"bang-{table_order}"],
                    )
                    table_unit.metadata["header"] = stripped
                    in_table = True
                    continue

                assert table_unit is not None
                row_text = stripped
                cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
                _pop_to(_LEVEL_RANK["table_row"])
                row = _add_unit(
                    level="table_row",
                    title="",
                    text=row_text,
                    line_index=idx,
                    parent=table_unit,
                    extra_id_parts=[f"hang-{len([u for u in units if u.parent_id == table_unit.id]) + 1}"],
                )
                row.metadata["cells"] = cells
                continue

        clause_match = _CLAUSE_RE.match(stripped)
        if clause_match:
            number = clause_match.group(1)
            text = clause_match.group(2).strip()
            _add_unit(
                level="clause",
                title="",
                text=text,
                line_index=idx,
                extra_id_parts=[f"khoan-{_slug(number)}"],
                level_meta={"clause_number": number},
            )
            continue

        point_match = _POINT_RE.match(stripped)
        if point_match:
            letter = point_match.group(1)
            text = point_match.group(2).strip()
            _add_unit(
                level="point",
                title="",
                text=text,
                line_index=idx,
                extra_id_parts=[f"diem-{letter}"],
                level_meta={"point_letter": letter},
            )
            continue

        dash_match = _DASH_RE.match(stripped)
        if dash_match:
            text = dash_match.group(1).strip()
            _add_unit(
                level="subpoint",
                title="",
                text=text,
                line_index=idx,
                extra_id_parts=[f"muc-{_slug(text[:20])}"],
            )
            continue

        _append_to_top(stripped, idx)

    _flush_preamble()
    _close_table()

    _build_context_text(units)
    _enrich_metadata(units)

    if article_ids is not None:
        keep_ids = _closure_for_articles(units, article_ids)
        units = [u for u in units if u.id in keep_ids]

    if not include_chapters:
        units = [u for u in units if u.level != "chapter"]

    metadata = {
        "source_path": source_path,
        "article_count": len([u for u in units if u.level == "article"]),
        "unit_count": len(units),
    }
    return ParseResult(document_id=document_id, units=units, metadata=metadata)


def _build_context_text(units: list[DocumentUnit]) -> None:
    by_id = {u.id: u for u in units}

    def _ancestors(unit: DocumentUnit) -> list[DocumentUnit]:
        result = []
        current = by_id.get(unit.parent_id) if unit.parent_id else None
        while current is not None:
            result.insert(0, current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return result

    for unit in units:
        ancestors = _ancestors(unit)
        context_lines = [
            u.title or u.text
            for u in ancestors
            if u.level in ("chapter", "article", "clause", "point")
        ]
        context_lines.append(unit.title or unit.text)
        unit.context_text = "\n".join(context_lines).strip()


def _enrich_metadata(units: list[DocumentUnit]) -> None:
    for unit in units:
        unit.metadata["crossrefs"] = _extract_crossrefs(unit.text)


def _closure_for_articles(units: list[DocumentUnit], article_ids: set[str]) -> set[str]:
    by_id = {u.id: u for u in units}
    keep: set[str] = set()
    article_units = [
        u for u in units if u.level == "article" and u.metadata.get("article_number") in article_ids
    ]
    for article in article_units:
        for unit in units:
            if unit.id == article.id or _is_descendant(unit, article, by_id):
                keep.add(unit.id)
        current = by_id.get(article.parent_id) if article.parent_id else None
        while current is not None:
            keep.add(current.id)
            current = by_id.get(current.parent_id) if current.parent_id else None
    return keep


def _is_descendant(unit: DocumentUnit, ancestor: DocumentUnit, by_id: dict[str, DocumentUnit]) -> bool:
    current = by_id.get(unit.parent_id) if unit.parent_id else None
    while current is not None:
        if current.id == ancestor.id:
            return True
        current = by_id.get(current.parent_id) if current.parent_id else None
    return False
