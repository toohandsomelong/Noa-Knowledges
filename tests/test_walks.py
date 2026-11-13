"""Tests for paper-style walk generation."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from kgrag.walks import (
    Walk,
    bfs_walks,
    compute_walk_stats,
    generate_walks,
    load_walks,
    random_walks,
    save_walks,
)


@pytest.fixture
def simple_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("a", title="A", text="Alpha", citation_label="")
    graph.add_node("b", title="B", text="Beta", citation_label="Điều 1")
    graph.add_node("c", title="C", text="Gamma", citation_label="")
    graph.add_edge("a", "b", relation="contains")
    graph.add_edge("b", "c", relation="contains")
    return graph


@pytest.fixture
def cyclic_graph():
    graph = nx.MultiDiGraph()
    for node in ("a", "b", "c"):
        graph.add_node(node, title=node.upper(), text=node, citation_label="")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "a")
    return graph


@pytest.fixture
def isolated_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("x", title="X", text="alone", citation_label="")
    return graph


def _graph_hash(graph):
    # Tests use a stable placeholder; module function is private.
    from kgrag.walks import _graph_hash as gh

    return gh(graph)


def test_bfs_includes_root_walk(simple_graph):
    walks = bfs_walks(simple_graph, "a", max_depth=3, graph_hash=_graph_hash(simple_graph))
    assert any(len(w.nodes) == 1 and w.nodes[0] == "a" for w in walks)


def test_bfs_spanning_tree_one_path_per_node(simple_graph):
    walks = bfs_walks(simple_graph, "a", max_depth=3, graph_hash=_graph_hash(simple_graph))
    targets = [w.nodes[-1] for w in walks]
    assert sorted(targets) == sorted(set(targets))
    assert set(targets) == {"a", "b", "c"}


def test_bfs_depth_limit(simple_graph):
    walks = bfs_walks(simple_graph, "a", max_depth=1, graph_hash=_graph_hash(simple_graph))
    assert all(len(w.nodes) <= 2 for w in walks)
    assert {w.nodes[-1] for w in walks} == {"a", "b"}


def test_bfs_no_duplicate_targets(simple_graph):
    walks = bfs_walks(simple_graph, "a", max_depth=3, graph_hash=_graph_hash(simple_graph))
    ends = [w.nodes[-1] for w in walks]
    assert len(ends) == len(set(ends))


def test_bfs_records_relation_labels(simple_graph):
    walks = bfs_walks(simple_graph, "a", max_depth=2, graph_hash=_graph_hash(simple_graph))
    long = next(w for w in walks if len(w.nodes) == 3)
    assert long.relations == ("contains", "contains")


def test_random_walk_exact_length_with_cycle(cyclic_graph):
    rng = random.Random("42:a")
    walks = random_walks(
        cyclic_graph, "a", num_walks=20, max_depth=3, rng=rng, graph_hash=_graph_hash(cyclic_graph)
    )
    assert all(len(w.nodes) == 4 for w in walks)
    assert all(len(w.nodes) - 1 == 3 for w in walks)


def test_random_walk_revisits_allowed(cyclic_graph):
    rng = random.Random("7:a")
    walks = random_walks(
        cyclic_graph, "a", num_walks=50, max_depth=4, rng=rng, graph_hash=_graph_hash(cyclic_graph)
    )
    revisits = [w for w in walks if len(set(w.nodes)) < len(w.nodes)]
    assert revisits, "expected at least one walk with a revisit in a cycle"


def test_random_walk_seed_reproducible(cyclic_graph):
    rng_a = random.Random("42:a")
    rng_b = random.Random("42:a")
    walks_a = random_walks(
        cyclic_graph, "a", 10, 3, rng_a, graph_hash=_graph_hash(cyclic_graph)
    )
    walks_b = random_walks(
        cyclic_graph, "a", 10, 3, rng_b, graph_hash=_graph_hash(cyclic_graph)
    )
    assert walks_a == walks_b


def test_random_walk_dead_end_stops_early(simple_graph):
    rng = random.Random(0)
    walks = random_walks(
        simple_graph, "b", num_walks=5, max_depth=10, rng=rng, graph_hash=_graph_hash(simple_graph)
    )
    for walk in walks:
        assert len(walk.nodes) <= 3
        assert walk.nodes[-1] == "c"


def test_isolated_node_yields_length_zero_walks(isolated_graph):
    gh = _graph_hash(isolated_graph)
    bfs = bfs_walks(isolated_graph, "x", max_depth=3, graph_hash=gh)
    assert all(len(w.nodes) == 1 for w in bfs)

    rng = random.Random(0)
    randoms = random_walks(isolated_graph, "x", 5, 3, rng, graph_hash=gh)
    assert all(len(w.nodes) == 1 for w in randoms)


def test_generate_walks_counts_and_strategies(simple_graph):
    walks = generate_walks(
        simple_graph,
        bfs_depth=2,
        random_walks_per_node=3,
        random_depth=2,
        seed=7,
        strategy="both",
    )
    strategies = {"bfs": 0, "random": 0}
    for walk in walks:
        strategies[walk.strategy] += 1
    # a,b,c each get 1 bfs root walk; spanning tree adds b from a, c from b.
    assert strategies["bfs"] == 6
    assert strategies["random"] == 9


def test_compute_walk_stats(simple_graph):
    walks = generate_walks(
        simple_graph,
        bfs_depth=2,
        random_walks_per_node=2,
        random_depth=2,
        seed=1,
        strategy="both",
    )
    stats = compute_walk_stats(walks)
    assert stats["total_walks"] == len(walks)
    assert stats["by_strategy"]["bfs"] == 6
    assert stats["by_strategy"]["random"] == 6
    assert stats["min_length"] == 1
    assert stats["max_length"] == 3


def test_save_and_load_walks_roundtrip(tmp_path, simple_graph):
    walks = generate_walks(
        simple_graph,
        bfs_depth=1,
        random_walks_per_node=1,
        random_depth=1,
        seed=1,
        strategy="both",
    )
    path = tmp_path / "walks.jsonl"
    save_walks(walks, path)
    loaded = load_walks(path)
    assert [w.to_dict() for w in loaded] == [w.to_dict() for w in walks]


def test_generate_walks_on_real_graph(graph):
    walks = generate_walks(
        graph,
        bfs_depth=2,
        random_walks_per_node=2,
        random_depth=2,
        seed=1,
        strategy="bfs",
    )
    assert walks
    assert all(isinstance(w, Walk) for w in walks)
    # BFS spanning tree: no duplicate end nodes per root.
    by_root: dict[str, list[str]] = {}
    for w in walks:
        by_root.setdefault(w.start_node, []).append(w.nodes[-1])
    for ends in by_root.values():
        assert len(ends) == len(set(ends))


@pytest.fixture(scope="module")
def graph():
    from kgrag.ingest import load_source_text, parse_regulation
    from kgrag.kg import build_structural_graph

    text = load_source_text()
    result = parse_regulation(text)
    return build_structural_graph(result.units)
