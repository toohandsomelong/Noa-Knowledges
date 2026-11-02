"""Tests for the regulation parser."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from kgrag.ingest import (
    load_source_text,
    load_units_jsonl,
    parse_regulation,
    parse_source_file,
    save_parse_report,
    save_units_jsonl,
)


@pytest.fixture(scope="module")
def parsed():
    text = load_source_text()
    return parse_regulation(text)


def test_load_source_discovers_file():
    text = load_source_text()
    assert "Điều 13" in text
    assert "Điều 18" in text


def test_all_articles_parsed(parsed):
    articles = [u for u in parsed.units if u.level == "article"]
    assert len(articles) == 35  # 3 decision articles + 32 regulation articles


def test_no_duplicate_ids(parsed):
    ids = [u.id for u in parsed.units]
    assert len(ids) == len(set(ids))


def test_decision_and_regulation_namespaced(parsed):
    decision = [u for u in parsed.units if u.level == "article" and u.id.startswith("decision:")]
    regulation = [u for u in parsed.units if u.level == "article" and u.id.startswith("regulation:")]
    assert len(decision) == 3
    assert len(regulation) == 32


def test_article_13_exists(parsed):
    art13 = next(
        (u for u in parsed.units if u.level == "article" and u.metadata.get("article_number") == "13"),
        None,
    )
    assert art13 is not None
    assert "Đánh giá và tính điểm học phần" in art13.title


def test_article_13_has_clauses_points_and_table(parsed):
    art13 = next(
        u for u in parsed.units
        if u.level == "article" and u.metadata.get("article_number") == "13"
    )
    descendants = [u for u in parsed.units if u.parent_id and u.parent_id.startswith(art13.id)]
    clauses = [u for u in descendants if u.level == "clause"]
    points = [u for u in descendants if u.level == "point"]
    table_rows = [u for u in descendants if u.level == "table_row"]

    assert len(clauses) >= 3
    assert len(points) >= 4
    assert len(table_rows) == 8  # A, B+, B, C+, C, D+, D, F


def test_article_18_exists(parsed):
    art18 = next(
        (u for u in parsed.units if u.level == "article" and u.metadata.get("article_number") == "18"),
        None,
    )
    assert art18 is not None
    assert "Xử lý kết quả học tập" in art18.title


def test_article_18_has_clauses_and_points(parsed):
    art18 = next(
        u for u in parsed.units
        if u.level == "article" and u.metadata.get("article_number") == "18"
    )
    descendants = [u for u in parsed.units if u.parent_id and u.parent_id.startswith(art18.id)]
    clauses = [u for u in descendants if u.level == "clause"]
    points = [u for u in descendants if u.level == "point"]
    assert len(clauses) >= 4
    assert len(points) >= 4


def test_table_rows_preserve_cells(parsed):
    rows = [u for u in parsed.units if u.level == "table_row"]
    assert rows
    sample = next(r for r in rows if "A" in r.text)
    assert "cells" in sample.metadata
    assert any("A" in cell for cell in sample.metadata["cells"])


def test_context_text_contains_ancestors(parsed):
    art13 = next(
        u for u in parsed.units
        if u.level == "article" and u.metadata.get("article_number") == "13"
    )
    clause3 = next(
        u for u in parsed.units
        if u.parent_id == art13.id and u.level == "clause" and u.metadata.get("clause_number") == "3"
    )
    assert art13.title in clause3.context_text
    assert clause3.text in clause3.context_text


def test_multi_line_clause_span_is_not_single_line(parsed):
    multi = next(
        (u for u in parsed.units if u.level == "clause" and u.source_span.end_line > u.source_span.start_line),
        None,
    )
    assert multi is not None


def test_mvp_filter_includes_only_requested_articles():
    text = load_source_text()
    result = parse_regulation(text, article_ids={"13", "18"})
    articles = [u for u in result.units if u.level == "article"]
    numbers = {u.metadata["article_number"] for u in articles}
    assert numbers == {"13", "18"}


def test_crossrefs_extracted(parsed):
    units_with_refs = [u for u in parsed.units if u.metadata.get("references")]
    assert units_with_refs


def test_crossrefs_extract_valid_targets():
    text = "\n".join(
        [
            "Điều 1. Tiêu đề",
            "1. Theo Điều 21, khoản 3, điểm c và Mục II.",
        ]
    )
    result = parse_regulation(text, strict=False)
    refs = [
        (ref["kind"], ref["target"])
        for unit in result.units
        for ref in unit.metadata.get("references", [])
    ]
    assert ("dieu", "21") in refs
    assert ("khoan", "3") in refs
    assert ("diem", "c") in refs
    assert ("muc", "II") in refs


def test_crossrefs_ignore_false_positives():
    text = "\n".join(
        [
            "Điều 1. Tiêu đề",
            "1. Điều kiện áp dụng, Điểm học phần, điểm chữ, khoản này, mục tiêu, mục lục, điểm F, Điểm R, Điểm M, điểm D, điểm I.",
        ]
    )
    result = parse_regulation(text, strict=False)
    refs = [
        (ref["kind"], ref["target"])
        for unit in result.units
        for ref in unit.metadata.get("references", [])
    ]
    assert refs == []


def test_crossref_targets_are_well_formed(parsed):
    patterns = {
        "dieu": r"\d+[A-Za-z]?",
        "khoan": r"\d+[A-Za-z]?",
        "diem": r"[a-z]",
        "muc": r"(?:\d+|[IVXLCDMivxlcdm]+|[a-z])",
    }
    for unit in parsed.units:
        for ref in unit.metadata.get("references", []):
            assert re.fullmatch(patterns[ref["kind"]], ref["target"])


def test_crossrefs_are_deduplicated_per_unit():
    text = "\n".join(
        [
            "Điều 1. Tiêu đề",
            "1. Theo Điều 21 và Điều 21.",
        ]
    )
    result = parse_regulation(text, strict=False)
    refs = [
        (ref["kind"], ref["target"])
        for unit in result.units
        for ref in unit.metadata.get("references", [])
    ]
    assert refs == [("dieu", "21")]


def test_decision_article_points_are_parsed():
    result = parse_regulation(load_source_text())
    article = next(unit for unit in result.units if unit.id == "decision:dieu-2")
    points = [
        unit
        for unit in result.units
        if unit.parent_id == article.id and unit.level == "point"
    ]
    assert [point.metadata["point_letter"] for point in points] == ["a", "b"]
    assert not any(
        "line 29" in warning or "line 30" in warning
        for warning in result.metadata["warnings"]
    )
    assert any("line 37" in warning for warning in result.metadata["warnings"])


def test_citation_labels_reset_between_articles(parsed):
    articles = [unit for unit in parsed.units if unit.level == "article"]
    assert all("khoản" not in unit.citation_label for unit in articles)
    assert all("điểm" not in unit.citation_label for unit in articles)


def test_citation_labels_for_clause_and_point(parsed):
    article = next(unit for unit in parsed.units if unit.id == "regulation:chuong-i:dieu-2")
    clause = next(
        unit
        for unit in parsed.units
        if unit.parent_id == article.id
        and unit.level == "clause"
        and unit.metadata.get("clause_number") == "7"
    )
    point = next(
        unit
        for unit in parsed.units
        if unit.parent_id == clause.id
        and unit.level == "point"
        and unit.metadata.get("point_letter") == "a"
    )
    assert article.citation_label == "Điều 2"
    assert clause.citation_label == "Điều 2 khoản 7"
    assert point.citation_label == "Điều 2 khoản 7 điểm a"


def test_chapter_units_exist(parsed):
    chapters = [u for u in parsed.units if u.level == "chapter"]
    assert len(chapters) >= 3


def test_point_letters_parsed(parsed):
    points = [u for u in parsed.units if u.level == "point"]
    letters = {u.metadata["point_letter"] for u in points}
    assert "a" in letters
    assert "b" in letters


def test_markdown_export_parses_identically():
    md_path = Path("data/processed/docs/Toàn văn quy chế đào tạo theo tín chỉ.md")
    txt_result = parse_regulation(load_source_text())
    md_result = parse_source_file(md_path)

    txt_ids = [u.id for u in txt_result.units]
    md_ids = [u.id for u in md_result.units]
    assert txt_ids == md_ids


def test_markdown_headings_and_bold_are_normalized():
    text = "\n".join(
        [
            "## Điều 1. Phạm vi",
            "1. Quy chế này áp dụng...",
            "**Điều 2: Đối tượng**",
            "1. Sinh viên.",
            "a) Các hệ chính quy.",
            "A) Hệ thường xuyên.",
        ]
    )
    result = parse_regulation(text, strict=False)
    articles = [u for u in result.units if u.level == "article"]
    points = [u for u in result.units if u.level == "point"]
    assert len(articles) == 2
    assert len(points) == 2


def test_alternate_punctuation_styles():
    text = "\n".join(
        [
            "Điều 1. Tiêu đề",
            "1: Một",
            "2. Hai",
            "a. Ba",
            "(b) Bốn",
            "- Năm",
            "+ Sáu",
        ]
    )
    result = parse_regulation(text, strict=False)
    assert len([u for u in result.units if u.level == "clause"]) == 2
    assert len([u for u in result.units if u.level == "point"]) == 2
    assert len([u for u in result.units if u.level == "subpoint"]) == 2


def test_strict_mode_records_warnings():
    result = parse_regulation(load_source_text(), strict=True)
    assert result.metadata["warnings"]
    assert any("line 37" in w for w in result.metadata["warnings"])


def test_strict_mode_disabled_allows_continuations():
    result = parse_regulation(load_source_text(), strict=False)
    assert not result.metadata["warnings"]


def test_table_ownership(parsed):
    tables = [u for u in parsed.units if u.level == "table"]
    rows = [u for u in parsed.units if u.level == "table_row"]
    assert tables
    assert rows
    for row in rows:
        parent = next(u for u in parsed.units if u.id == row.parent_id)
        assert parent.level == "table"
    for table in tables:
        assert table.parent_id is not None


def test_citation_labels(parsed):
    labels = [u.citation_label for u in parsed.units if u.citation_label]
    assert any("Điều 13" in label for label in labels)
    assert any("khoản 3" in label for label in labels)


def test_muc_reference_extracted(parsed):
    refs = [
        ref
        for u in parsed.units
        for ref in u.metadata.get("references", [])
        if ref["kind"] == "muc"
    ]
    assert refs


def test_jsonl_roundtrip(tmp_path):
    result = parse_regulation(load_source_text())
    jsonl_path = tmp_path / "units.jsonl"
    save_units_jsonl(result.units, jsonl_path)
    loaded = load_units_jsonl(jsonl_path)
    assert [u.id for u in loaded] == [u.id for u in result.units]


def test_parse_source_file_sets_hash_and_path():
    source_path = Path("data/docs/Toàn văn quy chế đào tạo theo tín chỉ.txt")
    result = parse_source_file(source_path)
    assert result.metadata["source_hash"]
    assert result.units[0].source_file.endswith(".txt")


def test_save_parse_report(tmp_path):
    result = parse_regulation(load_source_text())
    report_path = tmp_path / "report.json"
    save_parse_report(result, report_path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["document_id"] == result.document_id
    assert data["metadata"]["unit_count"] == len(result.units)
