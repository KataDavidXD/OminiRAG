"""Convert real LightRAG KG data into the format expected by the simplified stores.

Reads from a LightRAG graph directory (e.g. /data1/ragworkspace/train/fullwiki/graph/)
and produces the JSON files expected by the simplified LightRAG stores:
  - chunks.json    (from kv_store_text_chunks.json)
  - kv.json        (from kv_store_full_entities + kv_store_full_relations)
  - graph.json     (from graph_chunk_entity_relation.graphml)

The vdb_*.json files are already in the correct format (no conversion needed).

Usage:
    python scripts/prepare_lightrag_stores.py /data1/ragworkspace/train/fullwiki/graph
    python scripts/prepare_lightrag_stores.py /data1/ragworkspace/train/ultradomain/graph
    python scripts/prepare_lightrag_stores.py --all   # both fullwiki + ultradomain
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

FULLWIKI_GRAPH = Path("/data1/ragworkspace/train/fullwiki/graph")
ULTRADOMAIN_GRAPH = Path("/data1/ragworkspace/train/ultradomain/graph")


def convert_chunks(graph_dir: Path) -> int:
    """Convert kv_store_text_chunks.json -> chunks.json."""
    src = graph_dir / "kv_store_text_chunks.json"
    dst = graph_dir / "chunks.json"
    if not src.exists():
        print(f"  SKIP chunks: {src} not found")
        return 0

    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks = {}
    for chunk_id, info in raw.items():
        doc_id = info.get("full_doc_id", "")
        chunks[chunk_id] = {
            "content": info.get("content", ""),
            "doc_ids": [doc_id] if doc_id else [],
            "tokens": info.get("tokens", 0),
        }

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  chunks.json: {len(chunks)} chunks ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    return len(chunks)


def convert_kv(graph_dir: Path) -> int:
    """Build kv.json from kv_store_full_entities + kv_store_full_relations + entity/relation chunks."""
    entity_path = graph_dir / "kv_store_full_entities.json"
    relation_path = graph_dir / "kv_store_full_relations.json"
    entity_chunks_path = graph_dir / "kv_store_entity_chunks.json"
    relation_chunks_path = graph_dir / "kv_store_relation_chunks.json"
    dst = graph_dir / "kv.json"

    kv_items = []

    entity_chunks = {}
    if entity_chunks_path.exists():
        with open(entity_chunks_path, "r", encoding="utf-8") as f:
            entity_chunks = json.load(f)

    relation_chunks = {}
    if relation_chunks_path.exists():
        with open(relation_chunks_path, "r", encoding="utf-8") as f:
            relation_chunks = json.load(f)

    if entity_path.exists():
        with open(entity_path, "r", encoding="utf-8") as f:
            entities = json.load(f)
        for doc_id, info in entities.items():
            entity_names = info.get("entity_names", [])
            for name in entity_names:
                ec = entity_chunks.get(name, {})
                kv_items.append({
                    "key": [name],
                    "value": f"Entity: {name}",
                    "source_chunk_ids": ec.get("chunk_ids", []),
                    "source_doc_ids": [doc_id],
                })

    if relation_path.exists():
        with open(relation_path, "r", encoding="utf-8") as f:
            relations = json.load(f)
        for doc_id, info in relations.items():
            for pair in info.get("relation_pairs", []):
                if isinstance(pair, list) and len(pair) >= 2:
                    key_str = f"{pair[0]}<SEP>{pair[1]}"
                    rc = relation_chunks.get(key_str, {})
                    kv_items.append({
                        "key": pair[:2],
                        "value": f"Relation: {pair[0]} -> {pair[1]}",
                        "source_chunk_ids": rc.get("chunk_ids", []),
                        "source_doc_ids": [doc_id],
                    })

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(kv_items, f, ensure_ascii=False)
    print(f"  kv.json: {len(kv_items)} items ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    return len(kv_items)


def convert_graphml(graph_dir: Path) -> int:
    """Convert graph_chunk_entity_relation.graphml -> graph.json."""
    graphml_path = graph_dir / "graph_chunk_entity_relation.graphml"
    dst = graph_dir / "graph.json"
    if not graphml_path.exists():
        print(f"  SKIP graph: {graphml_path} not found")
        return 0

    print(f"  Parsing GraphML ({graphml_path.stat().st_size / 1024 / 1024:.1f} MB)...")
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    tree = ET.parse(str(graphml_path))
    root = tree.getroot()

    key_defs = {}
    for key_elem in root.findall("g:key", ns):
        kid = key_elem.get("id")
        key_defs[kid] = {
            "name": key_elem.get("attr.name"),
            "type": key_elem.get("attr.type"),
            "for": key_elem.get("for"),
        }

    graph_elem = root.find("g:graph", ns)
    if graph_elem is None:
        print("  ERROR: no <graph> element found")
        return 0

    nodes = []
    for node_elem in graph_elem.findall("g:node", ns):
        node_id = node_elem.get("id")
        node = {"name": node_id}
        for data_elem in node_elem.findall("g:data", ns):
            key_id = data_elem.get("key")
            attr_name = key_defs.get(key_id, {}).get("name", key_id)
            text = data_elem.text or ""
            if attr_name == "entity_type":
                node["type"] = text
            elif attr_name == "description":
                node["description"] = text
            elif attr_name == "source_id":
                node["source_doc_ids"] = [s.strip() for s in text.split(",") if s.strip()]
            elif attr_name == "entity_id":
                pass
            else:
                node[attr_name] = text
        nodes.append(node)

    edges = []
    for edge_elem in graph_elem.findall("g:edge", ns):
        source = edge_elem.get("source")
        target = edge_elem.get("target")
        edge = {"source": source, "target": target}
        for data_elem in edge_elem.findall("g:data", ns):
            key_id = data_elem.get("key")
            attr_name = key_defs.get(key_id, {}).get("name", key_id)
            text = data_elem.text or ""
            if attr_name == "keywords":
                edge["keywords"] = [k.strip() for k in text.split(",") if k.strip()]
            elif attr_name == "description":
                edge["description"] = text
            elif attr_name == "weight":
                try:
                    edge["weight"] = float(text)
                except ValueError:
                    edge["weight"] = 1.0
            elif attr_name == "source_id":
                edge["source_doc_ids"] = [s.strip() for s in text.split(",") if s.strip()]
            else:
                edge[attr_name] = text
        edges.append(edge)

    graph_data = {"nodes": nodes, "edges": edges}
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False)
    print(f"  graph.json: {len(nodes)} nodes, {len(edges)} edges ({dst.stat().st_size / 1024 / 1024:.1f} MB)")
    return len(nodes) + len(edges)


def prepare_dir(graph_dir: Path):
    print(f"\n{'='*60}")
    print(f"  Preparing LightRAG stores: {graph_dir}")
    print(f"{'='*60}")

    if not graph_dir.exists():
        print(f"  ERROR: directory not found: {graph_dir}")
        return

    t0 = time.time()

    n_chunks = convert_chunks(graph_dir)
    n_kv = convert_kv(graph_dir)
    n_graph = convert_graphml(graph_dir)

    print(f"\n  Done in {time.time()-t0:.1f}s: {n_chunks} chunks, {n_kv} kv items, {n_graph} graph elements")
    print(f"  VDB files (no conversion needed):")
    for vdb in sorted(graph_dir.glob("vdb_*.json")):
        print(f"    {vdb.name}: {vdb.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Prepare LightRAG store files from real KG data")
    parser.add_argument("graph_dir", nargs="?", default=None, help="Path to graph directory")
    parser.add_argument("--all", action="store_true", help="Process both fullwiki and ultradomain")
    parser.add_argument("--fullwiki-only", action="store_true", help="Process fullwiki only")
    args = parser.parse_args()

    if args.all:
        prepare_dir(FULLWIKI_GRAPH)
        prepare_dir(ULTRADOMAIN_GRAPH)
    elif args.graph_dir:
        prepare_dir(Path(args.graph_dir))
    elif args.fullwiki_only:
        prepare_dir(FULLWIKI_GRAPH)
    else:
        prepare_dir(FULLWIKI_GRAPH)


if __name__ == "__main__":
    main()
