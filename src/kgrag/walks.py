"""Generate paper-style walks over the structural regulation graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx

from kgrag.kg import compute_stats, load_source, parse_regulation


@dataclass(frozen=True)
class Walk:
    """A paper-style graph walk: an ordered path with relation labels."""

    walk_id: str
    strategy: str
    start_node: str
    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    node_labels: tuple[str, ...]
    node_texts: tuple[str, ...]
    template_text: str
    graph_hash: str

    @property
    def depth(self) -> int:
        return len(self.nodes) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "walk_id": self.walk_id,
            "strategy": self.strategy,
            "start_node": self.start_node,
            "depth": self.depth,
            "nodes": list(self.nodes),
            "relations": list(self.relations),
            "node_labels": list(self.node_labels),
            "node_texts": list(self.node_texts),
            "text": self.template_text,
            "graph_hash": self.graph_hash,
        }


def _graph_hash(graph: nx.MultiDiGraph) -> str:
    """A stable hash of the graph topology and node labels."""
    digest = hashlib.sha256()
    for node in sorted(graph.nodes()):
        data = graph.nodes[node]
        label = data.get("citation_label") or data.get("title") or node
        text = data.get("text") or ""
        digest.update(f"{node}|{label}|{text}\n".encode("utf-8"))
    for source, target, key, data in sorted(graph.edges(data=True, keys=True)):
        digest.update(
            f"{source}|{target}|{key}|{data.get('relation', '')}\n".encode("utf-8")
        )
    return digest.hexdigest()[:16]


def _out_edges(graph: nx.MultiDiGraph, node: str) -> list[tuple[str, str]]:
    """Return outgoing (relation, neighbor) pairs, including parallel edges."""
    return sorted(
        (data.get("relation", ""), target)
        for _, target, data in graph.out_edges(node, data=True)
    )


def _node_label(graph: nx.MultiDiGraph, node: str) -> str:
    data = graph.nodes[node]
    return data.get("citation_label") or data.get("title") or node


def _node_text(graph: nx.MultiDiGraph, node: str) -> str:
    return graph.nodes[node].get("text") or ""


def _template_text(
    nodes: Sequence[str],
    relations: Sequence[str],
    labels: Sequence[str],
    texts: Sequence[str],
) -> str:
    """Fallback plain-text rendering of a walk."""
    parts: list[str] = []
    for idx, node in enumerate(nodes):
        label = labels[idx]
        text = texts[idx]
        line = f"{label}. {text}" if label else text
        if idx > 0:
            rel = relations[idx - 1] if idx - 1 < len(relations) else ""
            if rel:
                line = f"({rel}) {line}"
        parts.append(line)
    return "\n".join(parts)


def bfs_walks(
    graph: nx.MultiDiGraph,
    start: str,
    max_depth: int,
    graph_hash: str,
) -> list[Walk]:
    """Generate spanning-tree BFS walks up to *max_depth* edges from *start*."""
    walks: list[Walk] = []

    parent: dict[str, tuple[str, str] | None] = {start: None}
    queue: list[str] = [start]

    for _ in range(max_depth):
        next_queue: list[str] = []
        for node in queue:
            for relation, neighbor in _out_edges(graph, node):
                if neighbor in parent:
                    continue
                parent[neighbor] = (node, relation)
                next_queue.append(neighbor)
        queue = next_queue

    for idx, target in enumerate(sorted(parent.keys())):
        nodes: list[str] = []
        relations: list[str] = []
        current: str | None = target
        while current is not None:
            nodes.append(current)
            prev = parent[current]
            if prev is not None:
                relations.append(prev[1])
                current = prev[0]
            else:
                current = None
        nodes.reverse()
        relations.reverse()

        labels = [_node_label(graph, n) for n in nodes]
        texts = [_node_text(graph, n) for n in nodes]
        walks.append(
            Walk(
                walk_id=f"bfs:{start}:{idx}",
                strategy="bfs",
                start_node=start,
                nodes=tuple(nodes),
                relations=tuple(relations),
                node_labels=tuple(labels),
                node_texts=tuple(texts),
                template_text=_template_text(nodes, relations, labels, texts),
                graph_hash=graph_hash,
            )
        )

    return walks


def random_walks(
    graph: nx.MultiDiGraph,
    start: str,
    num_walks: int,
    max_depth: int,
    rng: random.Random,
    graph_hash: str,
) -> list[Walk]:
    """Generate *num_walks* standard random walks of up to *max_depth* edges."""
    walks: list[Walk] = []

    for walk_idx in range(num_walks):
        nodes = [start]
        relations: list[str] = []

        for _ in range(max_depth):
            current = nodes[-1]
            edges = _out_edges(graph, current)
            if not edges:
                break
            relation, neighbor = rng.choice(edges)
            relations.append(relation)
            nodes.append(neighbor)

        labels = [_node_label(graph, n) for n in nodes]
        texts = [_node_text(graph, n) for n in nodes]
        walks.append(
            Walk(
                walk_id=f"random:{start}:{walk_idx}",
                strategy="random",
                start_node=start,
                nodes=tuple(nodes),
                relations=tuple(relations),
                node_labels=tuple(labels),
                node_texts=tuple(texts),
                template_text=_template_text(nodes, relations, labels, texts),
                graph_hash=graph_hash,
            )
        )

    return walks


def generate_walks(
    graph: nx.MultiDiGraph,
    *,
    bfs_depth: int = 3,
    random_walks_per_node: int = 10,
    random_depth: int = 3,
    seed: int = 42,
    strategy: str = "bfs",
) -> list[Walk]:
    """Generate BFS and/or random walks for every node in *graph*."""
    graph_hash = _graph_hash(graph)
    nodes = sorted(graph.nodes())
    walks: list[Walk] = []

    for start in nodes:
        if strategy in ("bfs", "both"):
            walks.extend(bfs_walks(graph, start, bfs_depth, graph_hash))

        if strategy in ("random", "both", "rw"):
            rng = random.Random(f"{seed}:{start}")
            walks.extend(
                random_walks(
                    graph,
                    start,
                    num_walks=random_walks_per_node,
                    max_depth=random_depth,
                    rng=rng,
                    graph_hash=graph_hash,
                )
            )

    return walks


def compute_walk_stats(walks: list[Walk]) -> dict[str, Any]:
    """Compute summary statistics for generated walks."""
    lengths = [len(w.nodes) for w in walks]
    strategies = Counter(w.strategy for w in walks)
    depths = Counter(w.depth for w in walks)

    return {
        "total_walks": len(walks),
        "by_strategy": dict(strategies),
        "by_depth": dict(depths),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "unique_nodes": sorted({w.start_node for w in walks}),
    }


def save_walks(walks: list[Walk], path: str | Path) -> None:
    """Persist walks as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for walk in walks:
            handle.write(json.dumps(walk.to_dict(), ensure_ascii=False))
            handle.write("\n")


def load_walks(path: str | Path) -> list[Walk]:
    """Load walks from JSONL."""
    path = Path(path)
    walks: list[Walk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            walks.append(
                Walk(
                    walk_id=data["walk_id"],
                    strategy=data["strategy"],
                    start_node=data["start_node"],
                    nodes=tuple(data["nodes"]),
                    relations=tuple(data["relations"]),
                    node_labels=tuple(data["node_labels"]),
                    node_texts=tuple(data["node_texts"]),
                    template_text=data["text"],
                    graph_hash=data["graph_hash"],
                )
            )
    return walks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style graph walks")
    parser.add_argument(
        "--graph",
        default="artifacts/kg/structural_graph.graphml",
        help="Path to the structural graph",
    )
    parser.add_argument(
        "--bfs-depth",
        type=int,
        default=3,
        help="Maximum BFS walk depth (edges)",
    )
    parser.add_argument(
        "--random-walks",
        type=int,
        default=10,
        help="Number of random walks per node",
    )
    parser.add_argument(
        "--random-depth",
        type=int,
        default=3,
        help="Maximum random walk depth (edges)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed base",
    )
    parser.add_argument(
        "--strategy",
        choices=["bfs", "random", "rw", "both"],
        default="bfs",
        help="Walk strategy to generate",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/kg",
        help="Output directory",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Rebuild the structural graph from source before walking",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graph_path = Path(args.graph)

    if args.build_graph or not graph_path.exists():
        text, path, source_format, source_hash = load_source()
        result = parse_regulation(
            text,
            document_id="quyche-2026",
            source_file=path.as_posix(),
            source_format=source_format,
            source_hash=source_hash,
        )
        from kgrag.kg import build_structural_graph, save_graph, save_stats

        graph = build_structural_graph(result.units)
        stats = compute_stats(graph)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_graph(graph, out_dir / "structural_graph.graphml")
        save_stats(stats, out_dir / "stats.json")
    else:
        graph = nx.read_graphml(graph_path)
        if not isinstance(graph, nx.MultiDiGraph):
            graph = nx.MultiDiGraph(graph)

    strategy = args.strategy
    if strategy == "rw":
        strategy = "random"

    walks = generate_walks(
        graph,
        bfs_depth=args.bfs_depth,
        random_walks_per_node=args.random_walks,
        random_depth=args.random_depth,
        seed=args.seed,
        strategy=strategy,
    )
    stats = compute_walk_stats(walks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_walks(walks, out_dir / "walks.jsonl")
    with (out_dir / "walk_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
