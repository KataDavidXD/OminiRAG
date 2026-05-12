"""Build CorpusIndex JSON files from raw corpus data.

Reads fullwiki_corpus.json (or ultradomain JSONL) and produces a
{chunk_id: {content, doc_ids}} dict suitable for CorpusIndex.from_json_file().

Usage:
    python scripts/build_corpus_index.py                         # fullwiki only
    python scripts/build_corpus_index.py --ultradomain           # both
    python scripts/build_corpus_index.py --smoke-test            # build + quick retrieval test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FULLWIKI_CORPUS = Path("/data1/ragworkspace/train/fullwiki/fullwiki_corpus.json")
FULLWIKI_PARQUET = Path("/data1/ragworkspace/train/fullwiki/fullwiki_sample_500_uniform.parquet")
ULTRADOMAIN_DIR = Path("/data1/ragworkspace/train/ultradomain")

INDEX_DIR = PROJECT_ROOT / "data" / "corpus_indexes"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    words = text.split()
    if len(words) <= chunk_size:
        return [{"content": text, "doc_ids": [title]}]
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text_str = " ".join(words[start:end])
        chunks.append({"content": chunk_text_str, "doc_ids": [title], "chunk_index": idx})
        idx += 1
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_fullwiki_index() -> dict[str, dict]:
    print(f"Loading {FULLWIKI_CORPUS} ...", flush=True)
    with open(FULLWIKI_CORPUS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"  {len(docs)} documents loaded", flush=True)

    chunks_dict: dict[str, dict] = {}
    for doc in docs:
        title = doc["title"]
        text = doc["text"].strip()
        if not text:
            continue
        for chunk in chunk_text(text, title):
            cid = f"fullwiki::{title}::chunk_{len(chunks_dict)}"
            chunks_dict[cid] = chunk

    print(f"  {len(chunks_dict)} chunks produced (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})", flush=True)
    return chunks_dict


def build_ultradomain_index(domain: str = "mix") -> dict[str, dict]:
    jsonl_path = ULTRADOMAIN_DIR / f"{domain}.jsonl"
    if not jsonl_path.exists():
        print(f"  {jsonl_path} not found, skipping", flush=True)
        return {}

    print(f"Loading {jsonl_path} ...", flush=True)
    chunks_dict: dict[str, dict] = {}
    n_docs = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            context = item.get("context", "")
            if not context:
                continue
            n_docs += 1
            title = item.get("question", f"ud_{n_docs}")[:80]
            for chunk in chunk_text(context, title):
                cid = f"ultradomain::{domain}::chunk_{len(chunks_dict)}"
                chunks_dict[cid] = chunk

    print(f"  {n_docs} docs -> {len(chunks_dict)} chunks", flush=True)
    return chunks_dict


def save_index(chunks_dict: dict[str, dict], name: str) -> Path:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INDEX_DIR / f"{name}_corpus_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, ensure_ascii=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)", flush=True)
    return out_path


def smoke_test(index_path: Path):
    print(f"\n--- Smoke Test: BM25 + Dense + Hybrid on {index_path.name} ---", flush=True)
    from rag_contracts.retrieval_methods import BM25Retrieval, DenseRetrieval, HybridRetrieval, CorpusIndex

    t0 = time.time()
    corpus = CorpusIndex.from_json_file(index_path)
    print(f"  CorpusIndex loaded: {len(corpus)} chunks ({time.time()-t0:.1f}s)", flush=True)

    test_queries = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "Which state is home to the Kickapoo tribe?",
    ]
    if FULLWIKI_PARQUET.exists():
        import pandas as pd
        df = pd.read_parquet(FULLWIKI_PARQUET)
        test_queries = df["question"].head(3).tolist()
        print(f"  Using real queries from parquet:", flush=True)
        for q in test_queries:
            print(f"    {q[:90]}", flush=True)

    print(f"\n  BM25 Retrieval:", flush=True)
    t0 = time.time()
    bm25 = BM25Retrieval(corpus=corpus)
    print(f"    Index built in {time.time()-t0:.1f}s", flush=True)
    for q in test_queries:
        results = bm25.retrieve([q], top_k=3)
        print(f"    Q: {q[:70]}...")
        for r in results[:2]:
            print(f"      [{r.score:.2f}] {r.title}: {r.content[:80]}...")
        if not results:
            print(f"      (no results)")

    print(f"\n  Dense Retrieval (e5-small):", flush=True)
    t0 = time.time()
    dense = DenseRetrieval(corpus=corpus)
    print(f"    Index built in {time.time()-t0:.1f}s", flush=True)
    for q in test_queries:
        results = dense.retrieve([q], top_k=3)
        print(f"    Q: {q[:70]}...")
        for r in results[:2]:
            print(f"      [{r.score:.3f}] {r.title}: {r.content[:80]}...")
        if not results:
            print(f"      (no results)")

    print(f"\n  Hybrid Retrieval (BM25+Dense RRF):", flush=True)
    hybrid = HybridRetrieval(bm25=bm25, dense=dense)
    for q in test_queries:
        results = hybrid.retrieve([q], top_k=3)
        print(f"    Q: {q[:70]}...")
        for r in results[:2]:
            print(f"      [{r.score:.4f}] {r.title}: {r.content[:80]}...")
        if not results:
            print(f"      (no results)")

    print(f"\n  Smoke test PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ultradomain", action="store_true", help="Also build ultradomain index")
    parser.add_argument("--ud-domain", default="mix", help="UltraDomain domain (default: mix)")
    parser.add_argument("--smoke-test", action="store_true", help="Run retrieval smoke test after building")
    parser.add_argument("--smoke-test-only", type=str, default=None, help="Run smoke test on existing index file")
    args = parser.parse_args()

    if args.smoke_test_only:
        smoke_test(Path(args.smoke_test_only))
        return

    print("=" * 60)
    print("  Building CorpusIndex files")
    print("=" * 60)

    fw_path = None
    if FULLWIKI_CORPUS.exists():
        fw_chunks = build_fullwiki_index()
        fw_path = save_index(fw_chunks, "fullwiki")
    else:
        print(f"  {FULLWIKI_CORPUS} not found, skipping fullwiki")

    ud_path = None
    if args.ultradomain and ULTRADOMAIN_DIR.exists():
        ud_chunks = build_ultradomain_index(args.ud_domain)
        if ud_chunks:
            ud_path = save_index(ud_chunks, f"ultradomain_{args.ud_domain}")

    if args.smoke_test and fw_path:
        smoke_test(fw_path)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
