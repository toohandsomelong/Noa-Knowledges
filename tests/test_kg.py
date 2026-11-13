"""Tests for the structural knowledge graph builder."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from kgrag.ingest import load_source_text, parse_regulation
from kgrag.kg import build_structural_graph, compute_stats, save_graph, save_stats


@pytest.fixture(scope="module")
def graph():
    text = load_source_text()
    units = parse_regulation(text).units
    return build_structural_graph(units)


@pytest.fixture(scope="module")
def parsed():
    return parse_regulation(load_source_text())


def _has_reference(graph: nx.MultiDiGraph, source: str, target: str, kind: str, target_value: str) -> bool:
    if not graph.has_edge(source, target):
        return False
    return any(
        attrs.get("relation") == "references"
        and attrs.get("ref_kind") == kind
        and attrs.get("ref_target") == target_value
        for attrs in graph[source][target].values()
    )


def test_every_unit_is_a_node(graph, parsed):
    assert set(graph.nodes()) == {u.id for u in parsed.units}


def test_node_count_matches_units(graph, parsed):
    assert graph.number_of_nodes() == len(parsed.units)


def test_node_attributes_preserved(graph):
    art = next(n for n, d in graph.nodes(data=True) if d["level"] == "article")
    data = graph.nodes[art]
    assert data["document_id"]
    assert data["level"] == "article"
    assert "text" in data
    assert "context_text" in data
    assert "metadata_json" in data
    assert "source_start_line" in data


def test_parent_child_edges(graph, parsed):
    expected_edges = {(u.parent_id, u.id) for u in parsed.units if u.parent_id}
    actual_edges = {
        (s, t)
        for s, t, d in graph.edges(data=True)
        if d.get("relation") == "contains"
    }
    assert actual_edges == expected_edges


def test_contains_edge_count(graph, parsed):
    contains = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("relation") == "contains"
    )
    parents = sum(1 for u in parsed.units if u.parent_id)
    assert contains == parents


def test_article_reference_resolves(graph):
    assert _has_reference(
        graph,
        "regulation:chuong-i:dieu-6:khoan-2:diem-a",
        "regulation:chuong-i:dieu-7",
        "dieu",
        "7",
    )


def test_cross_article_clause_reference_resolves(graph):
    assert _has_reference(
        graph,
        "regulation:chuong-ii:dieu-9:khoan-6:diem-c",
        "regulation:chuong-iii:dieu-13:khoan-3",
        "khoan",
        "3",
    )


def test_cross_article_point_reference_resolves(graph):
    assert _has_reference(
        graph,
        "regulation:chuong-iv:dieu-27:khoan-2",
        "regulation:chuong-iii:dieu-13:khoan-3:diem-a",
        "diem",
        "a",
    )


def test_reference_stays_within_namespace(graph):
    assert _has_reference(graph, "decision:dieu-3", "decision:dieu-3", "dieu", "3")


def test_dangling_muc_reference_marked(graph):
    node = "regulation:chuong-iii:dieu-13:khoan-3:diem-c"
    raw = graph.nodes[node].get("dangling_refs_json", "[]")
    dangling = json.loads(raw)
    assert any(d["kind"] == "muc" and d["target"] == "c" for d in dangling)

    ref_edges = [
        (s, t, d)
        for s, t, d in graph.edges(data=True)
        if s == node and d.get("relation") == "references" and d.get("ref_kind") == "muc"
    ]
    assert not ref_edges


def test_stats_consistent(graph, parsed):
    stats = compute_stats(graph)
    assert stats["node_count"] == len(parsed.units)
    assert stats["contains_edges"] == sum(1 for u in parsed.units if u.parent_id)
    assert stats["reference_edges"] == sum(stats["reference_edges_by_kind"].values())
    assert stats["root_count"] == sum(1 for u in parsed.units if u.parent_id is None)
    assert stats["root_count"] == 2


def test_graph_roots_are_documents(graph):
    roots = [n for n, d in graph.nodes(data=True) if not d.get("parent_id")]
    assert sorted(roots) == ["decision", "regulation"]
    assert all(graph.nodes[n]["level"] == "document" for n in roots)


def test_graph_has_two_weak_components(graph):
    assert nx.number_weakly_connected_components(graph) == 2


def test_graphml_roundtrip(graph, tmp_path):
    graphml_path = tmp_path / "structural_graph.graphml"
    save_graph(graph, graphml_path)
    loaded = nx.read_graphml(graphml_path)

    assert loaded.number_of_nodes() == graph.number_of_nodes()

    original_contains = sum(
        1 for _, _, d in graph.edges(data=True) if d.get("relation") == "contains"
    )
    loaded_contains = sum(
        1 for _, _, d in loaded.edges(data=True) if d.get("relation") == "contains"
    )
    assert loaded_contains == original_contains


def test_save_stats(graph, tmp_path):
    stats_path = tmp_path / "stats.json"
    stats = compute_stats(graph)
    save_stats(stats, stats_path)
    loaded = json.loads(stats_path.read_text(encoding="utf-8"))
    assert loaded["node_count"] == graph.number_of_nodes()
    assert loaded["contains_edges"] == stats["contains_edges"]
