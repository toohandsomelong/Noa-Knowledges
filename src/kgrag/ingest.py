"""Parse Vietnamese regulation sources into hierarchical DocumentUnits."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from kgrag.ingest_normalize import NormalizedLine, normalize_text
from kgrag.ingest_profile import ParserProfile, vietnamese_regulation_profile
from kgrag.schemas import DocumentUnit, ParseResult, Reference, SourceSpan

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

_PARENT_RANK_GATE = {
    "section": -1,
    "chapter": 0,
    "article": 0,
    "clause": 2,
    "point": 2,
    "subpoint": 4,
    "table": 0,
}


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    text: str
    line: NormalizedLine


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_source(
    path: str | Path | None = None,
) -> tuple[str, Path, str, str]:
    """Load source text and return text, path, format, and SHA-256 hash."""
    if path is None:
        candidate = _find_default_source()
        if candidate is None:
            raise FileNotFoundError(
                "Could not discover regulation source. "
                "Set SOURCE_TEXT or pass an explicit path."
            )
        path = candidate
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    source_format = path.suffix.lower().lstrip(".") or "txt"
    source_hash = _sha256_file(path)
    return text, path, source_format, source_hash


def _allowed_here(kind: str, current_rank: int) -> bool:
    if current_rank < 0:
        return kind in ("section", "chapter", "article")
    return current_rank >= _PARENT_RANK_GATE[kind]


def _point_pattern_for_style(style: str) -> str:
    style = style.strip()
    if len(style) == 2 and style[0].isalpha() and style[1] == ")":
        return r"^([a-zA-Z])\)\s+(?P<text>.*)$"
    if len(style) == 2 and style[0].isalpha() and style[1] == ".":
        return r"^([a-zA-Z])\.\s+(?P<text>.*)$"
    if len(style) == 3 and style[0] == "(" and style[2] == ")":
        return r"^\(([a-zA-Z])\)\s+(?P<text>.*)$"
    return ""


def _point_patterns(profile: ParserProfile) -> list[re.Pattern[str]]:
    patterns = []
    for style in profile.point_styles:
        raw = _point_pattern_for_style(style)
        if raw:
            patterns.append(re.compile(raw))
    return patterns


def _match_point(stripped: str, profile: ParserProfile) -> tuple[str, str] | None:
    for pattern in _point_patterns(profile):
        match = pattern.match(stripped)
        if match:
            letter = next(g for g in match.groups() if g is not None)
            return letter, match.group("text").strip()
    return None


def _structural_appearance(line: NormalizedLine, profile: ParserProfile) -> str | None:
    stripped = line.text
    if profile.section_markers.get(stripped):
        return "section"
    if profile.chapter_re().match(stripped):
        return "chapter"
    if profile.article_re().match(stripped):
        return "article"
    if profile.clause_re().match(stripped):
        return "clause"
    if _match_point(stripped, profile):
        return "point"
    if profile.subpoint_re().match(stripped):
        return "subpoint"
    return None


def _detect_token(
    line: NormalizedLine,
    profile: ParserProfile,
    current_rank: int,
) -> _Token | None:
    stripped = line.text

    marker_ns = profile.section_markers.get(stripped)
    if marker_ns is not None:
        return _Token(kind="section", value=marker_ns, text=stripped, line=line)

    chapter_match = profile.chapter_re().match(stripped)
    if chapter_match:
        return _Token(
            kind="chapter",
            value=chapter_match.group(2),
            text=chapter_match.group(3) or "",
            line=line,
        )

    article_match = profile.article_re().match(stripped)
    if article_match and _allowed_here("article", current_rank):
        return _Token(
            kind="article",
            value=article_match.group(2),
            text=article_match.group(3).strip(),
            line=line,
        )

    clause_match = profile.clause_re().match(stripped)
    if clause_match and _allowed_here("clause", current_rank):
        return _Token(
            kind="clause",
            value=clause_match.group(1),
            text=clause_match.group(2).strip(),
            line=line,
        )

    point_text = _match_point(stripped, profile)
    if point_text is not None and _allowed_here("point", current_rank):
        letter, text = point_text
        return _Token(kind="point", value=letter, text=text, line=line)

    subpoint_match = profile.subpoint_re().match(stripped)
    if subpoint_match and _allowed_here("subpoint", current_rank):
        return _Token(
            kind="subpoint",
            value="",
            text=subpoint_match.group(1).strip(),
            line=line,
        )

    if profile.table_re().match(stripped):
        return _Token(kind="table", value="", text=stripped, line=line)

    return None


def parse_regulation(
    text: str,
    *,
    document_id: str = "quyche-2026",
    source_file: str = "",
    source_format: str = "txt",
    source_hash: str = "",
    profile: ParserProfile | None = None,
    strict: bool = True,
    article_ids: set[str] | None = None,
    include_chapters: bool = True,
) -> ParseResult:
    """Parse *text* into a hierarchical list of DocumentUnits."""
    profile = profile or vietnamese_regulation_profile()
    lines = normalize_text(text)
    units: list[DocumentUnit] = []
    stack: list[DocumentUnit] = []
    orders: dict[tuple[str | None, str], int] = defaultdict(int)

    prefix = profile.default_namespace
    pending_chapter: dict[str, Any] | None = None
    table_unit: DocumentUnit | None = None
    in_table = False

    preamble_lines: list[str] = []
    preamble_span: SourceSpan | None = None
    warnings: list[str] = []

    def _next_order(parent_id: str | None, level: str) -> int:
        key = (parent_id, level)
        orders[key] += 1
        return orders[key]

    def _current_rank() -> int:
        if not stack:
            return -1
        return _LEVEL_RANK.get(stack[-1].level, -1)

    def _pop_to(rank: int) -> DocumentUnit | None:
        while stack and _LEVEL_RANK.get(stack[-1].level, -1) >= rank:
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
        line: NormalizedLine,
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

        id_parts = [prefix]
        if parent:
            id_parts = parent.id.split(":")
        id_parts = list(id_parts)
        id_parts.extend(extra_id_parts)
        unit_id = ":".join(id_parts)

        span = SourceSpan(
            start_line=line.line_no,
            end_line=line.line_no,
            start_char=0,
            end_char=len(line.original),
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
            source_file=source_file,
            source_format=source_format,
            source_hash=source_hash,
            raw_text=line.text,
            citation_label="",
            metadata=meta,
        )
        units.append(unit)
        stack.append(unit)
        return unit

    def _append_to_top(line: NormalizedLine) -> None:
        nonlocal in_table, table_unit
        if not stack:
            preamble_lines.append(line.text)
            return
        if stack[-1].level in ("table", "table_row"):
            _close_table()
        if not stack:
            preamble_lines.append(line.text)
            return
        unit = stack[-1]
        unit.text = f"{unit.text} {line.text}".strip()
        unit.source_span = SourceSpan(
            start_line=unit.source_span.start_line,
            end_line=line.line_no,
            start_char=unit.source_span.start_char,
            end_char=len(line.original),
        )

    def _flush_preamble() -> None:
        if not preamble_lines:
            return
        text = " ".join(preamble_lines).strip()
        span = preamble_span or SourceSpan(1, 1)
        units.append(
            DocumentUnit(
                id=f"{prefix}:preamble",
                document_id=document_id,
                level="preamble",
                parent_id=None,
                title="Preamble",
                text=text,
                context_text=text,
                order=_next_order(None, "preamble"),
                source_span=span,
                source_file=source_file,
                source_format=source_format,
                source_hash=source_hash,
                raw_text=text,
                citation_label="",
                metadata={},
            )
        )
        preamble_lines.clear()

    def _create_chapter(
        number: str,
        title: str,
        start_line: NormalizedLine,
    ) -> DocumentUnit:
        return _add_unit(
            level="chapter",
            title=title,
            text=title,
            line=start_line,
            extra_id_parts=[f"chuong-{_slug(number)}"],
            level_meta={"chapter_number": number},
        )

    for line in lines:
        stripped = line.text

        if preamble_span is None and not stripped:
            continue
        if preamble_span is None and stripped:
            preamble_span = SourceSpan(line.line_no, line.line_no)

        if pending_chapter is not None and stripped:
            appearance = _structural_appearance(line, profile)
            if appearance in ("section", "chapter", "article"):
                _create_chapter(pending_chapter["number"], pending_chapter["number"], pending_chapter["line"])
            else:
                _create_chapter(pending_chapter["number"], stripped, pending_chapter["line"])
            pending_chapter = None
            if appearance in ("section", "chapter", "article"):
                pass
            else:
                continue

        token = _detect_token(line, profile, _current_rank())

        if token is None:
            appearance = _structural_appearance(line, profile)
            if appearance and strict:
                warnings.append(
                    f"line {line.line_no} looks like {appearance} "
                    f"but is not valid in the current context"
                )
            if not stripped:
                continue
            _append_to_top(line)
            continue

        if token.kind == "section":
            _flush_preamble()
            _close_table()
            stack.clear()
            prefix = token.value
            continue

        if token.kind == "chapter":
            _flush_preamble()
            if token.text:
                _create_chapter(token.value, token.text, token.line)
            else:
                pending_chapter = {"number": token.value, "line": token.line}
            continue

        if token.kind == "article":
            _flush_preamble()
            _add_unit(
                level="article",
                title=token.text,
                text=token.text,
                line=token.line,
                extra_id_parts=[f"dieu-{_slug(token.value)}"],
                level_meta={"article_number": token.value},
            )
            continue

        if token.kind == "clause":
            _add_unit(
                level="clause",
                title="",
                text=token.text,
                line=token.line,
                extra_id_parts=[f"khoan-{_slug(token.value)}"],
                level_meta={"clause_number": token.value},
            )
            continue

        if token.kind == "point":
            _add_unit(
                level="point",
                title="",
                text=token.text,
                line=token.line,
                extra_id_parts=[f"diem-{token.value.lower()}"],
                level_meta={"point_letter": token.value},
            )
            continue

        if token.kind == "subpoint":
            _add_unit(
                level="subpoint",
                title="",
                text=token.text,
                line=token.line,
                extra_id_parts=[f"muc-{_slug(token.text[:40])}-dong{token.line.line_no}"],
            )
            continue

        if token.kind == "table":
            if profile.table_separator_re().match(stripped):
                continue
            if not in_table:
                _close_table()
                table_unit = _add_unit(
                    level="table",
                    title="",
                    text="",
                    line=token.line,
                    extra_id_parts=[f"bang-{_next_order(None, 'table')}"],
                )
                table_unit.metadata["header"] = stripped
                in_table = True
            else:
                if table_unit is None:
                    continue
                row_index = len([u for u in units if u.parent_id == table_unit.id]) + 1
                row = _add_unit(
                    level="table_row",
                    title="",
                    text=stripped,
                    line=token.line,
                    parent=table_unit,
                    extra_id_parts=[f"hang-{row_index}"],
                )
                row.metadata["cells"] = [cell.strip() for cell in stripped.split("|")[1:-1]]
            continue

    _flush_preamble()
    _close_table()

    if pending_chapter is not None:
        _create_chapter(pending_chapter["number"], pending_chapter["number"], pending_chapter["line"])

    _build_context_text(units)
    _enrich_metadata(units, profile)

    if article_ids is not None:
        keep_ids = _closure_for_articles(units, article_ids)
        units = [u for u in units if u.id in keep_ids]

    if not include_chapters:
        units = [u for u in units if u.level != "chapter"]

    metadata = {
        "source_file": source_file,
        "source_format": source_format,
        "source_hash": source_hash,
        "profile": profile.name,
        "document_id": document_id,
        "article_count": len([u for u in units if u.level == "article"]),
        "unit_count": len(units),
        "warnings": warnings,
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


def _enrich_metadata(units: list[DocumentUnit], profile: ParserProfile) -> None:
    crossref_res = profile.crossref_res()
    by_id = {u.id: u for u in units}

    def _ancestors(unit: DocumentUnit) -> list[DocumentUnit]:
        result = []
        current = by_id.get(unit.parent_id) if unit.parent_id else None
        while current is not None:
            result.insert(0, current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        return result

    article_number = ""
    clause_number = ""
    point_letter = ""
    for unit in units:
        if unit.level in ("document", "preamble", "chapter"):
            article_number = ""
            clause_number = ""
            point_letter = ""
        elif unit.level == "article":
            article_number = unit.metadata.get("article_number", "")
            clause_number = ""
            point_letter = ""
        elif unit.level == "clause":
            clause_number = unit.metadata.get("clause_number", "")
            point_letter = ""
        elif unit.level == "point":
            point_letter = unit.metadata.get("point_letter", "")

        references: list[Reference] = []
        seen: set[tuple[str, str]] = set()
        for kind, pattern in crossref_res.items():
            for match in pattern.finditer(unit.text):
                target = match.group("target")
                key = (kind, target)
                if key in seen:
                    continue
                seen.add(key)
                references.append(
                    Reference(kind=kind, target=target, text=match.group(0))
                )
        unit.metadata["references"] = [asdict(ref) for ref in references]

        label_parts = []
        if article_number:
            label_parts.append(f"Điều {article_number}")
        if clause_number:
            label_parts.append(f"khoản {clause_number}")
        if point_letter:
            label_parts.append(f"điểm {point_letter}")
        unit.citation_label = " ".join(label_parts)

        if not unit.citation_label:
            ancestors = _ancestors(unit)
            nearest = next(
                (a for a in reversed(ancestors) if a.citation_label),
                None,
            )
            if nearest:
                unit.citation_label = nearest.citation_label


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


def _is_descendant(
    unit: DocumentUnit, ancestor: DocumentUnit, by_id: dict[str, DocumentUnit]
) -> bool:
    current = by_id.get(unit.parent_id) if unit.parent_id else None
    while current is not None:
        if current.id == ancestor.id:
            return True
        current = by_id.get(current.parent_id) if current.parent_id else None
    return False


def parse_source_file(
    path: str | Path,
    *,
    document_id: str = "quyche-2026",
    profile: ParserProfile | None = None,
    strict: bool = True,
    article_ids: set[str] | None = None,
    include_chapters: bool = True,
) -> ParseResult:
    """Parse a single source file, computing format and hash automatically."""
    path = Path(path)
    text, _, source_format, source_hash = load_source(path)
    return parse_regulation(
        text,
        document_id=document_id,
        source_file=path.as_posix(),
        source_format=source_format,
        source_hash=source_hash,
        profile=profile,
        strict=strict,
        article_ids=article_ids,
        include_chapters=include_chapters,
    )


def save_units_jsonl(units: list[DocumentUnit], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for unit in units:
            handle.write(json.dumps(unit.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_units_jsonl(path: str | Path) -> list[DocumentUnit]:
    path = Path(path)
    units: list[DocumentUnit] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            units.append(DocumentUnit(**data))
    return units


def save_parse_report(result: ParseResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
