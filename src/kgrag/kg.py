"""Build a structural knowledge graph from parsed regulation units."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from kgrag.ingest import load_source, parse_regulation, parse_source_file
from kgrag.ingest_profile import ParserProfile, vietnamese_regulation_profile
from kgrag.schemas import DocumentUnit, Reference


LEVEL_ORDER = ("document", "chapter", "article", "clause", "point", "subpoint", "table", "table_row")


def _namespace(unit: DocumentUnit) -> str:
    return unit.id.split(":")[0]


def _ancestor(units_by_id: dict[str, DocumentUnit], unit: DocumentUnit, level: str) -> DocumentUnit | None:
    current = units_by_id.get(unit.parent_id) if unit.parent_id else None
    while current is not None:
        if current.level == level:
            return current
        current = units_by_id.get(current.parent_id) if current.parent_id else None
    return None


def _span_distance(a: Reference, b: Reference) -> int:
    if a.end_char <= b.start_char:
        return b.start_char - a.end_char
    if b.end_char <= a.start_char:
        return a.start_char - b.end_char
    return 0


def _nearest(ref: Reference, pool: list[Reference]) -> Reference | None:
    if not pool:
        return None
    return min(pool, key=lambda other: _span_distance(ref, other))


def _node_attrs(unit: DocumentUnit) -> dict[str, Any]:
    return {
        "document_id": unit.document_id,
        "level": unit.level,
        "parent_id": unit.parent_id or "",
        "title": unit.title,
        "text": unit.text,
        "context_text": unit.context_text,
        "citation_label": unit.citation_label,
        "order": unit.order,
        "source_file": unit.source_file,
        "source_format": unit.source_format,
        "source_hash": unit.source_hash,
        "source_start_line": unit.source_span.start_line,
        "source_end_line": unit.source_span.end_line,
        "source_start_char": unit.source_span.start_char,
        "source_end_char": unit.source_span.end_char,
        "metadata_json": json.dumps(unit.metadata, ensure_ascii=False),
    }


def _resolve_references(
    units: list[DocumentUnit],
    profile: ParserProfile,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[dict[str, Any]]]:
    units_by_id = {u.id: u for u in units}
    articles: dict[tuple[str, str], list[DocumentUnit]] = defaultdict(list)
    clauses_by_article: dict[str, list[DocumentUnit]] = defaultdict(list)
    points_by_clause: dict[str, list[DocumentUnit]] = defaultdict(list)

    for unit in units:
        if unit.level == "article":
            articles[(_namespace(unit), unit.metadata["article_number"])].append(unit)

    for unit in units:
        if unit.level == "clause":
            article = _ancestor(units_by_id, unit, "article")
            if article is not None:
                clauses_by_article[article.id].append(unit)
        elif unit.level == "point":
            clause = _ancestor(units_by_id, unit, "clause")
            if clause is not None:
                points_by_clause[clause.id].append(unit)

    ref_edges: list[tuple[str, str, dict[str, Any]]] = []
    dangling: list[dict[str, Any]] = []

    for unit in units:
        raw_refs = unit.metadata.get("references", [])
        if not raw_refs:
            continue
        refs = [Reference(**raw_ref) for raw_ref in raw_refs]
        dieu_refs = [r for r in refs if r.kind == "dieu"]
        khoan_refs = [r for r in refs if r.kind == "khoan"]

        for ref in refs:
            target_unit: DocumentUnit | None = None
            reason: str | None = None

            if ref.kind == "dieu":
                candidates = articles.get((_namespace(unit), ref.target), [])
                target_unit = candidates[0] if len(candidates) == 1 else None
                reason = None if target_unit else "missing"

            elif ref.kind == "khoan":
                anchor_article = _ancestor(units_by_id, unit, "article")
                nearest_dieu = _nearest(ref, dieu_refs)
                if nearest_dieu is not None:
                    candidates = articles.get((_namespace(unit), nearest_dieu.target), [])
                    anchor_article = candidates[0] if len(candidates) == 1 else None
                if anchor_article is not None:
                    candidates = [
                        c
                        for c in clauses_by_article[anchor_article.id]
                        if c.metadata.get("clause_number") == ref.target
                    ]
                    target_unit = candidates[0] if len(candidates) == 1 else None
                    reason = None if target_unit else ("missing" if not candidates else "ambiguous")
                else:
                    reason = "no_article"

            elif ref.kind == "diem":
                base_clause = unit if unit.level == "clause" else _ancestor(units_by_id, unit, "clause")
                nearest_khoan = _nearest(ref, khoan_refs)
                if nearest_khoan is not None:
                    anchor_article = _ancestor(units_by_id, unit, "article")
                    nearest_dieu = _nearest(nearest_khoan, dieu_refs)
                    if nearest_dieu is not None:
                        candidates = articles.get((_namespace(unit), nearest_dieu.target), [])
                        anchor_article = candidates[0] if len(candidates) == 1 else None
                    if anchor_article is not None:
                        candidates = [
                            c
                            for c in clauses_by_article[anchor_article.id]
                            if c.metadata.get("clause_number") == nearest_khoan.target
                        ]
                        base_clause = candidates[0] if len(candidates) == 1 else None
                    else:
                        base_clause = None
                if base_clause is not None:
                    candidates = [
                        p
                        for p in points_by_clause[base_clause.id]
                        if p.metadata.get("point_letter", "").lower() == ref.target.lower()
                    ]
                    target_unit = candidates[0] if len(candidates) == 1 else None
                    reason = None if target_unit else ("missing" if not candidates else "ambiguous")
                else:
                    reason = "no_clause"

            else:
                reason = "unsupported_kind"

            if target_unit is not None:
                ref_edges.append(
                    (
                        unit.id,
                        target_unit.id,
                        {
                            "relation": "references",
                            "ref_kind": ref.kind,
                            "ref_target": ref.target,
                            "ref_text": ref.text,
                        },
                    )
                )
            else:
                dangling.append(
                    {
                        "source_id": unit.id,
                        "kind": ref.kind,
                        "target": ref.target,
                        "text": ref.text,
                        "reason": reason,
                    }
                )

    seen: set[tuple[str, str, str, str]] = set()
    deduped_edges: list[tuple[str, str, dict[str, Any]]] = []
    for source, target, attrs in ref_edges:
        key = (source, target, attrs["ref_kind"], attrs["ref_target"])
        if key in seen:
            continue
        seen.add(key)
        deduped_edges.append((source, target, attrs))

    return deduped_edges, dangling


def build_structural_graph(
    units: list[DocumentUnit],
    *,
    profile: ParserProfile | None = None,
) -> nx.MultiDiGraph:
    """Build a structural graph from regulation units."""
    profile = profile or vietnamese_regulation_profile()

    ids = [u.id for u in units]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate unit IDs found")

    graph = nx.MultiDiGraph()
    for unit in units:
        graph.add_node(unit.id, **_node_attrs(unit))

    contains_count = 0
    for unit in units:
        if unit.parent_id:
            if unit.parent_id not in graph:
                raise ValueError(f"Unit {unit.id} has unknown parent {unit.parent_id}")
            graph.add_edge(unit.parent_id, unit.id, relation="contains")
            contains_count += 1

    ref_edges, dangling = _resolve_references(units, profile)
    for source, target, attrs in ref_edges:
        graph.add_edge(source, target, **attrs)

    dangling_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in dangling:
        dangling_by_source[entry["source_id"]].append(entry)
    for source_id, entries in dangling_by_source.items():
        graph.nodes[source_id]["dangling_refs_json"] = json.dumps(entries, ensure_ascii=False)

    graph.graph["dangling_references_json"] = json.dumps(dangling, ensure_ascii=False)
    graph.graph["contains_edges"] = contains_count

    return graph


def compute_stats(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Compute statistics for the structural graph."""
    nodes_by_level = Counter(data["level"] for _, data in graph.nodes(data=True))
    relation_counts = Counter(data["relation"] for _, _, data in graph.edges(data=True))
    reference_kind_counts = Counter(
        data["ref_kind"]
        for _, _, data in graph.edges(data=True)
        if data.get("relation") == "references"
    )
    dangling = json.loads(graph.graph.get("dangling_references_json", "[]"))

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "contains_edges": relation_counts.get("contains", 0),
        "reference_edges": relation_counts.get("references", 0),
        "reference_edges_by_kind": dict(reference_kind_counts),
        "nodes_by_level": dict(nodes_by_level),
        "dangling_count": len(dangling),
        "dangling_references": dangling,
        "root_count": sum(
            1 for _, data in graph.nodes(data=True) if not data.get("parent_id")
        ),
        "leaf_count": sum(1 for node in graph if graph.out_degree(node) == 0),
        "self_loop_count": sum(1 for source, target in graph.edges() if source == target),
        "weak_component_count": nx.number_weakly_connected_components(graph),
    }


def save_graph(graph: nx.MultiDiGraph, path: str | Path) -> None:
    """Persist a structural graph to GraphML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def save_stats(stats: dict[str, Any], path: str | Path) -> None:
    """Persist graph statistics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the structural regulation KG")
    parser.add_argument("--source", default=None, help="Path to source text")
    parser.add_argument("--document-id", default="quyche-2026", help="Document ID")
    parser.add_argument(
        "--article-ids",
        default=None,
        help="Comma-separated article numbers to scope the graph",
    )
    parser.add_argument("--out-dir", default="artifacts/kg", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    article_ids: set[str] | None = None
    if args.article_ids:
        article_ids = {part.strip() for part in args.article_ids.split(",") if part.strip()}

    if args.source:
        result = parse_source_file(
            args.source,
            document_id=args.document_id,
            article_ids=article_ids,
        )
    else:
        text, path, source_format, source_hash = load_source()
        result = parse_regulation(
            text,
            document_id=args.document_id,
            source_file=path.as_posix(),
            source_format=source_format,
            source_hash=source_hash,
            article_ids=article_ids,
        )

    graph = build_structural_graph(result.units)
    stats = compute_stats(graph)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph(graph, out_dir / "structural_graph.graphml")
    save_stats(stats, out_dir / "stats.json")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
