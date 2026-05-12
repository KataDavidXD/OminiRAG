"""Per-dimension ablation study: fix best config, sweep one dimension at a time.

For each dimension, fixes all other dimensions at the best config values
and evaluates every option for that dimension. Produces a table showing
which component choices matter most.

Usage:
    # Quick test (2 items per config)
    python scripts/run_ablation.py --max-items 2

    # Full ablation with specific best config
    python scripts/run_ablation.py --best-config standard_passage,identity,bm25,identity,simple_llm --max-items 50

    # Save results
    python scripts/run_ablation.py --max-items 50 --output results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "AG-UCT"))
sys.path.insert(0, str(PROJECT_ROOT / "Benchmark_Sampling"))
sys.path.insert(0, str(PROJECT_ROOT / "A-Simplified-Core-Workflow-for-Enhancing-RAG"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

SLOT_NAMES = ["chunking", "query", "retrieval", "post_retrieval", "generation"]

SLOT_OPTIONS = [
    ["standard_passage", "kg_extraction"],
    ["identity", "lightrag_keywords"],
    ["bm25", "dense_e5", "bm25_dense_hybrid", "lightrag_hybrid", "lightrag_graph"],
    ["identity", "cross_encoder", "lightrag_compress", "selfrag_critique"],
    ["simple_llm", "longrag_reader", "lightrag_answer", "selfrag_generator"],
]

DEFAULT_BEST = ("standard_passage", "identity", "bm25", "identity", "simple_llm")


def load_eval_data(max_items: int) -> list[dict]:
    json_path = Path("/data1/ragworkspace/train/fullwiki/fullwiki_sample_500.json")
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import pandas as pd
        parquet = Path("/data1/ragworkspace/train/fullwiki/fullwiki_sample_500_uniform.parquet")
        raw = pd.read_parquet(parquet).to_dict("records")
    return [{"question": e["question"], "answer": e["answer"], "query_id": e.get("id", "")}
            for e in raw[:max_items]]


def evaluate_config(choices: tuple, eval_data: list, corpus) -> dict:
    from rag_contracts.component_registry import build_pipeline_from_config
    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter

    try:
        components = build_pipeline_from_config(choices, "hotpotqa", corpus=corpus)
    except Exception as e:
        return {"config": list(choices), "error": str(e), "avg_em": 0, "avg_f1": 0}

    generation = components.get("generation")
    retrieval = components.get("retrieval")
    if generation is None:
        return {"config": list(choices), "error": "no generation", "avg_em": 0, "avg_f1": 0}

    enriched = []
    for item in eval_data:
        try:
            results = retrieval.retrieve([item["question"]], top_k=5) if retrieval else []
        except Exception:
            results = []
        enriched.append({**item, "context_results": results})

    adapter = HotpotQABenchmarkAdapter()
    t0 = time.time()
    result = adapter.evaluate_generation(enriched, generation)
    elapsed = time.time() - t0

    return {
        "config": list(choices),
        "avg_em": result.avg_em,
        "avg_f1": result.avg_f1,
        "num_items": result.num_items,
        "elapsed_s": round(elapsed, 1),
    }


def run_ablation(best_config: tuple, eval_data: list, corpus) -> list[dict]:
    all_results = []
    print(f"\n  Best config: {best_config}")

    for dim_idx in range(len(SLOT_NAMES)):
        dim_name = SLOT_NAMES[dim_idx]
        print(f"\n  --- Sweep dimension {dim_idx}: {dim_name} ---")

        for option in SLOT_OPTIONS[dim_idx]:
            config = list(best_config)
            config[dim_idx] = option
            config = tuple(config)

            label = f"{dim_name}={option}"
            print(f"    {label}...", end=" ", flush=True)

            result = evaluate_config(config, eval_data, corpus)
            result["dimension"] = dim_name
            result["swept_value"] = option
            result["is_best"] = (option == best_config[dim_idx])

            if result.get("error"):
                print(f"ERROR: {result['error']}")
            else:
                marker = " <-- best" if result["is_best"] else ""
                print(f"EM={result['avg_em']:.2f} F1={result['avg_f1']:.2f} ({result['elapsed_s']:.1f}s){marker}")

            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-config", type=str, default=",".join(DEFAULT_BEST),
                        help="Comma-separated best config (default: BM25+SimpleLLM)")
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    best = tuple(args.best_config.split(","))
    assert len(best) == 5, f"Expected 5 slots, got {len(best)}"

    print("=" * 60)
    print("  OminiRAG Per-Dimension Ablation Study")
    print("=" * 60)

    eval_data = load_eval_data(args.max_items)
    print(f"  Eval items: {len(eval_data)}")

    from rag_contracts import CorpusIndex
    corpus = CorpusIndex.from_json_file(PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json")
    print(f"  Corpus: {len(corpus)} chunks")

    results = run_ablation(best, eval_data, corpus)

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY")
    print("=" * 60)
    for dim_name in SLOT_NAMES:
        dim_results = [r for r in results if r.get("dimension") == dim_name]
        print(f"\n  {dim_name}:")
        for r in dim_results:
            marker = " ***" if r.get("is_best") else ""
            err = r.get("error", "")
            if err:
                print(f"    {r['swept_value']:<25s}  FAIL ({err[:40]})")
            else:
                print(f"    {r['swept_value']:<25s}  EM={r['avg_em']:>6.2f}  F1={r['avg_f1']:>6.2f}{marker}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
