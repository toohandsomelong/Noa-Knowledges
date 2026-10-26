"""Tests for the regulation parser."""

from __future__ import annotations

import pytest

from kgrag.ingest import load_source_text, parse_regulation


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
    units_with_refs = [u for u in parsed.units if any(u.metadata["crossrefs"].values())]
    assert units_with_refs


def test_chapter_units_exist(parsed):
    chapters = [u for u in parsed.units if u.level == "chapter"]
    assert len(chapters) >= 3


def test_point_letters_parsed(parsed):
    points = [u for u in parsed.units if u.level == "point"]
    letters = {u.metadata["point_letter"] for u in points}
    assert "a" in letters
    assert "b" in letters
