"""Collect UCT search statistics and compare against full-dataset baselines.

Reads the --save-tree JSON produced by rag_pipeline_search.py, then
evaluates the UCT-found best config on the full 500-item test set and
compares against framework presets.

Usage:
    # After running UCT search with --save-tree:
    python scripts/collect_uct_stats.py --tree results/search_tree.json --test-set fullwiki

    # Compare against baselines on custom test set:
    python scripts/collect_uct_stats.py \\
        --tree results/search_tree.json \\
        --test-data /data1/ragworkspace/train/fullwiki/fullwiki_test_500.json \\
        --max-items 500 --output results/uct_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "AG-UCT"))
sys.path.insert(0, str(PROJECT_ROOT / "Benchmark_Sampling"))
sys.path.insert(0, str(PROJECT_ROOT / "A-Simplified-Core-Workflow-for-Enhancing-RAG"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

DATA_ROOT = Path(os.environ.get("OMINIRAG_DATA_ROOT", "/data1/ragworkspace/train"))

PRESETS = {
    "longrag": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
    "lightrag": ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "lightrag_answer"),
    "selfrag": ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator"),
    "bm25_simple": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
}


def load_test_data(path: str | None, benchmark: str, max_items: int) -> list[dict]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif benchmark == "fullwiki" or benchmark == "hotpotqa":
        p = DATA_ROOT / "fullwiki" / "fullwiki_test_500.json"
        if not p.exists():
            p = DATA_ROOT / "fullwiki" / "fullwiki_sample_500.json"
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise FileNotFoundError(f"No test data for benchmark={benchmark}")
    return raw[:max_items]


def evaluate_config_on_items(
    config_tuple: tuple[str, ...],
    items: list[dict],
    corpus,
) -> dict:
    """Evaluate a config on HotpotQA items, returning per-item and aggregate scores."""
    from rag_contracts.component_registry import build_pipeline_from_config
    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter

    try:
        components = build_pipeline_from_config(config_tuple, "hotpotqa", corpus=corpus)
    except Exception as e:
        return {"error": str(e), "avg_em": 0, "avg_f1": 0, "per_item": []}

    generation = components.get("generation")
    retrieval = components.get("retrieval")
    if generation is None:
        return {"error": "no generation", "avg_em": 0, "avg_f1": 0, "per_item": []}

    adapter = HotpotQABenchmarkAdapter()
    enriched = []
    for item in items:
        query = item["question"]
        ctx = []
        if retrieval and hasattr(retrieval, "retrieve"):
            try:
                ctx = retrieval.retrieve([query], top_k=5)
            except Exception:
                pass
        enriched.append({
            "question": query,
            "answer": item["answer"],
            "query_id": item.get("id", ""),
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "context_results": ctx,
        })

    result = adapter.evaluate_generation(enriched, generation)

    per_item = []
    if hasattr(result, "items") and result.items:
        for r in result.items:
            per_item.append({"em": r.get("em", 0), "f1": r.get("f1", 0),
                             "query_id": r.get("query_id", "")})

    return {
        "avg_em": result.avg_em,
        "avg_f1": result.avg_f1,
        "num_items": result.num_items,
        "per_item": per_item,
    }


def stratum_breakdown(items: list[dict], per_item_scores: list[dict]) -> dict:
    """Group scores by (type, level) stratum."""
    strata: dict[str, list[dict]] = defaultdict(list)
    for item, score in zip(items, per_item_scores):
        t = str(item.get("type", "nan"))
        l = str(item.get("level", "nan"))
        key = f"{t}_{l}" if t != "nan" and l != "nan" else "nan_nan"
        strata[key].append(score)

    result = {}
    for k, scores in sorted(strata.items()):
        ems = [s.get("em", 0) for s in scores]
        f1s = [s.get("f1", 0) for s in scores]
        n = len(scores)
        result[k] = {
            "count": n,
            "avg_em": sum(ems) / n if n else 0,
            "avg_f1": sum(f1s) / n if n else 0,
        }
    return result


def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: H0 is mean(A) <= mean(B)."""
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return {"delta": 0, "p_value": 1.0, "n": 0}

    rng = random.Random(seed)
    observed_delta = sum(scores_a[:n]) / n - sum(scores_b[:n]) / n

    count_ge = 0
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_a = sum(scores_a[i] for i in indices) / n
        boot_b = sum(scores_b[i] for i in indices) / n
        if (boot_a - boot_b) >= observed_delta:
            count_ge += 1

    return {
        "delta": round(observed_delta, 4),
        "p_value": round(count_ge / n_bootstrap, 4),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect UCT stats and compare against baselines")
    parser.add_argument("--tree", type=str, required=True,
                        help="Path to search tree JSON (from --save-tree)")
    parser.add_argument("--test-set", type=str, default="fullwiki",
                        help="Benchmark name: fullwiki / hotpotqa")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test data JSON (overrides --test-set)")
    parser.add_argument("--max-items", type=int, default=500)
    parser.add_argument("--baselines", type=str, nargs="*",
                        default=["bm25_simple"],
                        help="Baseline presets to compare against")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip real evaluation, only print tree stats")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  UCT Statistics Collection & Comparison", flush=True)
    print("=" * 60, flush=True)

    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)

    print("\n--- UCT Search Summary ---", flush=True)
    print(f"  Best config:       {tree.get('best_config')}", flush=True)
    print(f"  Best reward:       {tree.get('best_reward', 0):.4f}", flush=True)
    print(f"  Iterations:        {tree.get('iterations', 0)}", flush=True)
    print(f"  Evaluations:       {tree.get('total_evaluations', 0)}", flush=True)
    print(f"  Total cost:        {tree.get('total_cost', 0):.2f}", flush=True)
    print(f"  Materialized keys: {tree.get('materialized_keys_count', 0)}", flush=True)
    search_space = tree.get("search_space_size", 240)
    evals = tree.get("total_evaluations", 0)
    print(f"  Search space:      {search_space} configs", flush=True)
    print(f"  Efficiency:        {evals}/{search_space} = {evals/max(search_space,1):.1%}", flush=True)

    print("\n--- Slot Breakdown (depth 0) ---", flush=True)
    for action, stats in sorted(tree.get("slot_breakdown", {}).items()):
        print(f"  {action:20s}  visits={stats['visit_count']:4d}  "
              f"Q={stats['q_value']:.4f}  best={stats['best_value']:.4f}", flush=True)

    configs = tree.get("configs_visited", [])
    print(f"\n--- Top-10 Configs (by Q-value, {len(configs)} total visited) ---", flush=True)
    for i, c in enumerate(configs[:10]):
        cfg_str = "/".join(c["config_tuple"])
        print(f"  {i+1:2d}. Q={c['q_value']:.4f}  visits={c['visit_count']:3d}  {cfg_str}", flush=True)

    if args.skip_eval:
        print("\n  Skipping real evaluation (--skip-eval).", flush=True)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"tree_stats": tree}, f, ensure_ascii=False, indent=2)
            print(f"  Stats saved to {out}", flush=True)
        return

    best_config = tuple(tree["best_config"])
    print(f"\n--- Full-Dataset Evaluation ---", flush=True)
    test_items = load_test_data(args.test_data, args.test_set, args.max_items)
    print(f"  Test items: {len(test_items)}", flush=True)

    from rag_contracts import CorpusIndex
    index_path = PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json"
    corpus = None
    if index_path.exists():
        corpus = CorpusIndex.from_json_file(index_path)
        print(f"  CorpusIndex: {len(corpus)} chunks", flush=True)

    results = {}

    print(f"\n  Evaluating UCT-best: {'/'.join(best_config)}", flush=True)
    t0 = time.time()
    uct_result = evaluate_config_on_items(best_config, test_items, corpus)
    uct_time = time.time() - t0
    print(f"    EM={uct_result['avg_em']:.2f}  F1={uct_result['avg_f1']:.2f}  "
          f"n={uct_result.get('num_items',0)}  time={uct_time:.1f}s", flush=True)
    results["uct_best"] = {
        "config": list(best_config),
        **uct_result,
        "elapsed_s": round(uct_time, 1),
    }

    for preset_name in args.baselines:
        if preset_name not in PRESETS:
            print(f"  Unknown preset: {preset_name}, skipping", flush=True)
            continue
        config = PRESETS[preset_name]
        print(f"\n  Evaluating baseline [{preset_name}]: {'/'.join(config)}", flush=True)
        t0 = time.time()
        baseline_result = evaluate_config_on_items(config, test_items, corpus)
        baseline_time = time.time() - t0
        print(f"    EM={baseline_result['avg_em']:.2f}  F1={baseline_result['avg_f1']:.2f}  "
              f"n={baseline_result.get('num_items',0)}  time={baseline_time:.1f}s", flush=True)
        results[preset_name] = {
            "config": list(config),
            **baseline_result,
            "elapsed_s": round(baseline_time, 1),
        }

        if uct_result.get("per_item") and baseline_result.get("per_item"):
            uct_f1s = [s.get("f1", 0) for s in uct_result["per_item"]]
            base_f1s = [s.get("f1", 0) for s in baseline_result["per_item"]]
            boot = paired_bootstrap_test(uct_f1s, base_f1s)
            print(f"    Bootstrap (UCT - {preset_name}): delta={boot['delta']:.4f} p={boot['p_value']:.4f}", flush=True)
            results[f"bootstrap_uct_vs_{preset_name}"] = boot

    print("\n" + "=" * 60, flush=True)
    print("  COMPARISON SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'Config':<30s}  {'EM':>6s}  {'F1':>6s}  {'Time':>6s}", flush=True)
    print("  " + "-" * 55, flush=True)
    for name, r in results.items():
        if "config" not in r:
            continue
        label = name[:30]
        err = r.get("error", "")
        if err:
            print(f"  {label:<30s}  {'FAIL':>6s}  {'FAIL':>6s}  {err}", flush=True)
        else:
            print(f"  {label:<30s}  {r['avg_em']:>6.2f}  {r['avg_f1']:>6.2f}  {r.get('elapsed_s',0):>5.1f}s", flush=True)

    cost_saving = {
        "uct_evaluations": tree.get("total_evaluations", 0),
        "exhaustive_evaluations": search_space * len(test_items),
        "saving_ratio": 1.0 - (tree.get("total_evaluations", 0) / max(search_space * len(test_items), 1)),
    }
    print(f"\n  Cost savings: {cost_saving['uct_evaluations']} UCT evals vs "
          f"{cost_saving['exhaustive_evaluations']} exhaustive "
          f"({cost_saving['saving_ratio']:.1%} saved)", flush=True)

    if args.output:
        out_data = {
            "tree_stats": {
                "best_config": tree.get("best_config"),
                "best_reward": tree.get("best_reward"),
                "iterations": tree.get("iterations"),
                "total_evaluations": tree.get("total_evaluations"),
                "total_cost": tree.get("total_cost"),
                "materialized_keys_count": tree.get("materialized_keys_count"),
                "search_space_size": search_space,
            },
            "evaluation_results": results,
            "cost_savings": cost_saving,
            "test_set_size": len(test_items),
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Full report saved to {out}", flush=True)


if __name__ == "__main__":
    main()
