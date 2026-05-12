"""Evaluate 3 framework presets on HotpotQA fullwiki data.

Produces EM/F1 baselines for paper Table 1.

Usage:
    # Quick test (5 items, verify pipeline works)
    python scripts/run_baseline_eval.py --max-items 5

    # Full evaluation (500 items)
    python scripts/run_baseline_eval.py --max-items 500

    # With specific preset only
    python scripts/run_baseline_eval.py --preset longrag --max-items 10

    # Save results
    python scripts/run_baseline_eval.py --max-items 500 --output results/baselines.json
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


PRESETS = {
    "longrag": {
        "name": "LongRAG (BM25 + SimpleLLM Reader)",
        "config": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
    },
    "lightrag": {
        "name": "LightRAG (KG + Hybrid + Compress + Answer)",
        "config": ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "lightrag_answer"),
    },
    "selfrag": {
        "name": "Self-RAG (Dense + Critique + Generator)",
        "config": ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator"),
    },
    "bm25_simple": {
        "name": "BM25 + SimpleLLM (minimal baseline)",
        "config": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
    },
}


def load_eval_data(max_items: int = 500) -> list[dict]:
    """Load HotpotQA eval data from fullwiki parquet/JSON."""
    json_path = Path("/data1/ragworkspace/train/fullwiki/fullwiki_sample_500.json")
    parquet_path = Path("/data1/ragworkspace/train/fullwiki/fullwiki_sample_500_uniform.parquet")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif parquet_path.exists():
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        raw = df.to_dict("records")
    else:
        raise FileNotFoundError("No HotpotQA data found")

    items = []
    for entry in raw[:max_items]:
        items.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "query_id": entry.get("id", ""),
            "type": entry.get("type", ""),
            "level": entry.get("level", ""),
        })
    return items


def evaluate_preset(preset_name: str, eval_data: list[dict], corpus) -> dict:
    """Evaluate a single preset config on the eval data."""
    from rag_contracts.component_registry import build_pipeline_from_config
    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter

    preset = PRESETS[preset_name]
    choices = preset["config"]
    print(f"\n  Evaluating: {preset['name']}")
    print(f"  Config: {choices}")

    try:
        components = build_pipeline_from_config(choices, "hotpotqa", corpus=corpus)
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {e}")
        return {"preset": preset_name, "error": str(e), "avg_em": 0, "avg_f1": 0}

    generation = components.get("generation")
    retrieval = components.get("retrieval")

    if generation is None:
        print("  No generation component")
        return {"preset": preset_name, "error": "no generation", "avg_em": 0, "avg_f1": 0}

    adapter = HotpotQABenchmarkAdapter()
    t0 = time.time()

    enriched_items = []
    for item in eval_data:
        query = item["question"]

        if retrieval is not None and hasattr(retrieval, "retrieve"):
            try:
                results = retrieval.retrieve([query], top_k=5)
            except Exception as e:
                print(f"    Retrieval error for '{query[:50]}': {e}")
                results = []
        else:
            results = []

        enriched_items.append({
            **item,
            "context_results": results,
        })

    result = adapter.evaluate_generation(enriched_items, generation)
    elapsed = time.time() - t0

    print(f"  EM: {result.avg_em:.2f}  F1: {result.avg_f1:.2f}  Items: {result.num_items}  Time: {elapsed:.1f}s")

    return {
        "preset": preset_name,
        "name": preset["name"],
        "config": list(choices),
        "avg_em": result.avg_em,
        "avg_f1": result.avg_f1,
        "num_items": result.num_items,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate framework presets on HotpotQA")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Evaluate specific preset only")
    parser.add_argument("--max-items", type=int, default=10,
                        help="Number of evaluation items (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("  OminiRAG Baseline Preset Evaluation")
    print("=" * 60)

    eval_data = load_eval_data(args.max_items)
    print(f"  Loaded {len(eval_data)} HotpotQA items")

    from rag_contracts import CorpusIndex
    index_path = PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json"
    corpus = CorpusIndex.from_json_file(index_path)
    print(f"  CorpusIndex: {len(corpus)} chunks")

    presets_to_eval = [args.preset] if args.preset else list(PRESETS.keys())
    all_results = []

    for preset_name in presets_to_eval:
        result = evaluate_preset(preset_name, eval_data, corpus)
        all_results.append(result)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Preset':<20s}  {'EM':>6s}  {'F1':>6s}  {'Items':>5s}  {'Time':>6s}")
    print("  " + "-" * 50)
    for r in all_results:
        err = r.get("error", "")
        if err:
            print(f"  {r['preset']:<20s}  {'FAIL':>6s}  {'FAIL':>6s}  {'':>5s}  {err}")
        else:
            print(f"  {r['preset']:<20s}  {r['avg_em']:>6.2f}  {r['avg_f1']:>6.2f}  {r['num_items']:>5d}  {r['elapsed_s']:>5.1f}s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
