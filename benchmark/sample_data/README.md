# KG-Format Sample Data for Benchmarks

This directory contains **small, representative KG-indexed samples** for each of the 3 benchmarks used in the ominirag project. These are the **only** dummy/sample data files in the repository.

## Purpose

- Enable integration tests and demos without requiring full corpus access
- Demonstrate the LightRAG KG store format for each benchmark
- Serve as structural reference for the full index produced by the indexing pipeline

## Directory Structure

```
sample_data/
├── hotpotqa_kg_sample/     # 3 multi-hop QA questions from HotpotQA
├── ultradomain_kg_sample/  # 3 domain-specific questions (agriculture, CS, legal)
├── alce_kg_sample/         # 3 ALCE items (ASQA, QAMPARI, ELI5 subtasks)
└── README.md               # This file
```

## File Format (per benchmark)

Each benchmark sample contains:

| File | Contents | Format |
|------|----------|--------|
| `queries.jsonl` | Sample questions with ground truth answers | JSONL, one item per line |
| `chunks.json` | Text chunks from source documents | Dict mapping chunk_id -> {content, doc_ids, chunk_order_index} |
| `graph.json` | Knowledge graph with entities and relations | {nodes: [...], edges: [...]} |
| `kv.json` | Entity key-value descriptions | List of {key, value, source_chunk_ids, source_doc_ids} |
| `vdb_chunks.json` | Chunk embedding vectors | Dict mapping chunk_id -> [float, ...] |
| `vdb_entities.json` | Entity name embedding vectors | Dict mapping entity_name -> [float, ...] |
| `vdb_relations.json` | Relation keyword embedding vectors | Dict mapping relation_key -> [float, ...] |
| `alce_docs.json` | *(ALCE only)* Pre-retrieved documents per query | Dict mapping query_id -> [{title, text}, ...] |

### Notes on Embeddings

The `vdb_*.json` files contain **placeholder 8-dimensional vectors** for structural validation. In production, these would be 768-dim (Contriever) or 1536-dim (ada-002) dense vectors. The short placeholders let tests verify the store format without requiring an embedding model.

## How These Were Generated

1. **Source**: Subsampled from the full KG index built by colleague's indexing pipeline
2. **Process**: Selected 3-5 representative items per benchmark, extracted relevant chunks/entities/relations from the full index, truncated embeddings to 8-dim placeholders
3. **Validation**: Each sample was verified to contain correct multi-hop connections (HotpotQA), cross-domain entities (UltraDomain), and per-document segmentation (ALCE)

## How to Regenerate from Scratch

```python
# 1. Run the full LightRAG indexing pipeline on the benchmark corpus:
from lightrag_langgraph.main_pipeline import build_index_graph
graph = build_index_graph(config)
# Feed corpus documents through the graph

# 2. Extract a subsample from the resulting stores:
# - Pick 3-5 queries
# - Collect chunks referenced by those queries
# - Extract the subgraph of entities/relations connected to those chunks
# - Export embeddings for the selected chunks/entities/relations
```

## How to Use in Tests and Demos

```python
import json
from pathlib import Path

sample_dir = Path("benchmark/sample_data/hotpotqa_kg_sample")

# Load sample data
with open(sample_dir / "queries.jsonl", encoding="utf-8") as f:
    queries = [json.loads(line) for line in f if line.strip()]

with open(sample_dir / "chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

with open(sample_dir / "graph.json", encoding="utf-8") as f:
    kg = json.load(f)

# Use in tests -- convert chunks to RetrievalResult for pipeline testing
from rag_contracts import RetrievalResult
results = [
    RetrievalResult(
        source_id=cid,
        content=info["content"],
        score=1.0,
        title=info["doc_ids"][0] if info["doc_ids"] else cid,
    )
    for cid, info in chunks.items()
]

# For ALCE -- use ALCEDocRetrieval to wrap pre-retrieved docs:
from rag_contracts import ALCEDocRetrieval
with open("benchmark/sample_data/alce_kg_sample/alce_docs.json", encoding="utf-8") as f:
    alce_docs = json.load(f)
retrieval = ALCEDocRetrieval(docs=alce_docs["alce_asqa_1"])
```

## Benchmark-Specific Notes

### HotpotQA
- 3 questions requiring multi-hop reasoning (e.g., comparing nationalities of two people)
- KG shows bridge entities (e.g., "United States" connecting Scott Derrickson and Ed Wood)
- Tests EM/F1 metrics on short factual answers

### UltraDomain
- 3 questions spanning agriculture, computer science, and legal domains
- KG shows domain-specific concept hierarchies and cross-concept relations
- Tests LLM-judge evaluation (Comprehensiveness, Diversity, Empowerment)

### ALCE
- 3 items from the three ALCE subtasks: ASQA, QAMPARI, ELI5
- `alce_docs.json` contains pre-retrieved documents in ALCE's native format
- Optional mini KG built from ALCE source documents for LightRAG retrieval testing
- Tests F1, STR-EM, citation metrics, AutoAIS
