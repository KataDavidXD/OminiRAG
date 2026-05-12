"""Build stratified search/test sets using Neyman two-phase allocation.

Phase 1 (Pilot): Draw ~10% items proportionally across strata, run a
lightweight BM25+simple_llm eval to estimate per-stratum reward variance.

Phase 2 (Main): Allocate remaining budget by Neyman optimal allocation
(n_h proportional to N_h * sigma_h), draw the main sample.

Datasets:
  - HotpotQA (fullwiki): 500 items, stratified by (type x level) -> 7 strata
  - UltraDomain: 768 items, stratified by domain -> 4 strata

Usage:
    python scripts/build_stratified_search_set.py --hotpotqa-budget 400 --seed 42
    python scripts/build_stratified_search_set.py --skip-eval  # use uniform variance
"""
from __future__ import annotations

import argparse
import json
import math
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

DATA_ROOT = Path(os.environ.get("OMINIRAG_DATA_ROOT", "/data1/ragworkspace/train"))
OUTPUT_DIR = DATA_ROOT


def load_hotpotqa() -> list[dict]:
    json_path = DATA_ROOT / "fullwiki" / "fullwiki_sample_500.json"
    if not json_path.exists():
        raise FileNotFoundError(f"HotpotQA data not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ultradomain() -> list[dict]:
    ud_dir = DATA_ROOT / "ultradomain"
    items = []
    for fname in ["agriculture.jsonl", "cs.jsonl", "legal.jsonl", "mix.jsonl"]:
        fpath = ud_dir / fname
        if not fpath.exists():
            continue
        domain = fname.replace(".jsonl", "")
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_domain"] = domain
                    items.append(row)
    return items


def assign_hotpotqa_stratum(item: dict) -> str:
    t = str(item.get("type", "nan"))
    l = str(item.get("level", "nan"))
    if t == "nan" or l == "nan":
        return "nan_nan"
    return f"{t}_{l}"


def assign_ultradomain_stratum(item: dict) -> str:
    return item.get("_domain", "unknown")


def stratify_items(items: list[dict], stratum_fn) -> dict[str, list[dict]]:
    strata: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        strata[stratum_fn(item)].append(item)
    return dict(strata)


def proportional_allocation(strata: dict[str, list[dict]], budget: int) -> dict[str, int]:
    N = sum(len(v) for v in strata.values())
    if N == 0:
        return {k: 0 for k in strata}
    alloc = {}
    for k, v in strata.items():
        alloc[k] = max(1, int(budget * len(v) / N))
    total = sum(alloc.values())
    labels_by_size = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
    while total < budget:
        for k in labels_by_size:
            if alloc[k] < len(strata[k]) and total < budget:
                alloc[k] += 1
                total += 1
    while total > budget:
        for k in reversed(labels_by_size):
            if alloc[k] > 1 and total > budget:
                alloc[k] -= 1
                total -= 1
    return alloc


def neyman_allocation(
    strata: dict[str, list[dict]],
    variances: dict[str, float],
    budget: int,
) -> dict[str, int]:
    weights = {}
    for k, v in strata.items():
        sd = math.sqrt(max(variances.get(k, 0.0), 0.0))
        weights[k] = len(v) * sd

    total_w = sum(weights.values())
    if total_w < 1e-12:
        return proportional_allocation(strata, budget)

    alloc = {}
    for k in strata:
        raw = budget * weights[k] / total_w
        alloc[k] = max(1, int(raw))

    total = sum(alloc.values())
    labels_by_weight = sorted(strata.keys(), key=lambda k: weights[k], reverse=True)
    while total < budget:
        for k in labels_by_weight:
            if alloc[k] < len(strata[k]) and total < budget:
                alloc[k] += 1
                total += 1
    while total > budget:
        for k in reversed(labels_by_weight):
            if alloc[k] > 1 and total > budget:
                alloc[k] -= 1
                total -= 1
    for k in strata:
        alloc[k] = min(alloc[k], len(strata[k]))
    return alloc


def draw_sample(strata: dict[str, list[dict]], alloc: dict[str, int], rng: random.Random) -> list[dict]:
    drawn = []
    for k in sorted(strata.keys()):
        n = alloc.get(k, 0)
        pool = strata[k]
        if n >= len(pool):
            drawn.extend(pool)
        else:
            drawn.extend(rng.sample(pool, n))
    return drawn


def pilot_eval_hotpotqa(items: list[dict]) -> dict[str, float]:
    """Run lightweight BM25+simple_llm eval on pilot items, return per-stratum variance."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        from rag_contracts.component_registry import build_pipeline_from_config
        from rag_contracts import CorpusIndex

        index_path = PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json"
        corpus = None
        if index_path.exists():
            corpus = CorpusIndex.from_json_file(index_path)

        config = ("standard_passage", "identity", "bm25", "identity", "simple_llm")
        components = build_pipeline_from_config(config, "hotpotqa", corpus=corpus)
        generation = components.get("generation")
        retrieval = components.get("retrieval")

        if generation is None:
            return {}

        from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
        adapter = HotpotQABenchmarkAdapter()

        strata_rewards: dict[str, list[float]] = defaultdict(list)
        for item in items:
            query = item["question"]
            ctx = []
            if retrieval and hasattr(retrieval, "retrieve"):
                try:
                    ctx = retrieval.retrieve([query], top_k=5)
                except Exception:
                    pass
            enriched = {
                "question": query,
                "answer": item["answer"],
                "query_id": item.get("id", ""),
                "context_results": ctx,
            }
            result = adapter.evaluate_generation([enriched], generation)
            stratum = assign_hotpotqa_stratum(item)
            strata_rewards[stratum].append(result.avg_f1 / 100.0)

        variances = {}
        for k, rewards in strata_rewards.items():
            if len(rewards) >= 2:
                mean = sum(rewards) / len(rewards)
                variances[k] = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
            else:
                variances[k] = 0.1
        return variances
    except Exception as e:
        print(f"  Pilot eval failed: {e}. Using uniform variance.", flush=True)
        return {}


def build_hotpotqa_sets(budget: int, seed: int, skip_eval: bool) -> dict:
    print("=" * 60, flush=True)
    print("  HotpotQA Stratified Sampling", flush=True)
    print("=" * 60, flush=True)

    items = load_hotpotqa()
    print(f"  Total items: {len(items)}", flush=True)

    strata = stratify_items(items, assign_hotpotqa_stratum)
    print(f"  Strata: {len(strata)}", flush=True)
    for k in sorted(strata.keys()):
        print(f"    {k}: {len(strata[k])} items", flush=True)

    rng = random.Random(seed)

    pilot_budget = max(2 * len(strata), len(items) // 10)
    pilot_budget = min(pilot_budget, budget)
    pilot_alloc = proportional_allocation(strata, pilot_budget)
    pilot_items = draw_sample(strata, pilot_alloc, rng)
    print(f"\n  Pilot: {len(pilot_items)} items (allocation: {pilot_alloc})", flush=True)

    if skip_eval:
        variances = {k: 0.1 for k in strata}
        print("  Skipping pilot eval (uniform variance)", flush=True)
    else:
        print("  Running pilot evaluation (BM25 + simple_llm)...", flush=True)
        t0 = time.time()
        variances = pilot_eval_hotpotqa(pilot_items)
        elapsed = time.time() - t0
        print(f"  Pilot eval done in {elapsed:.1f}s", flush=True)
        if not variances:
            variances = {k: 0.1 for k in strata}

    print(f"  Per-stratum variances:", flush=True)
    for k in sorted(variances.keys()):
        print(f"    {k}: {variances[k]:.4f}", flush=True)

    remaining = budget - len(pilot_items)
    if remaining > 0:
        remaining_strata = {}
        for k, pool in strata.items():
            pilot_ids = {id(it) for it in pilot_items if assign_hotpotqa_stratum(it) == k}
            leftover = [it for it in pool if id(it) not in pilot_ids]
            remaining_strata[k] = leftover

        main_alloc = neyman_allocation(remaining_strata, variances, remaining)
        main_items = draw_sample(remaining_strata, main_alloc, rng)
        print(f"\n  Main draw: {len(main_items)} items (Neyman allocation: {main_alloc})", flush=True)
        search_set = pilot_items + main_items
    else:
        main_alloc = {}
        search_set = pilot_items

    print(f"\n  Final search set: {len(search_set)} items", flush=True)
    final_dist = defaultdict(int)
    for it in search_set:
        final_dist[assign_hotpotqa_stratum(it)] += 1
    for k in sorted(final_dist.keys()):
        print(f"    {k}: {final_dist[k]}", flush=True)

    out_search = OUTPUT_DIR / "fullwiki" / "fullwiki_search_400_stratified.json"
    with open(out_search, "w", encoding="utf-8") as f:
        json.dump(search_set, f, ensure_ascii=False, indent=1)
    print(f"\n  Saved search set: {out_search} ({len(search_set)} items)", flush=True)

    out_test = OUTPUT_DIR / "fullwiki" / "fullwiki_test_500.json"
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    print(f"  Saved test set: {out_test} ({len(items)} items)", flush=True)

    report = {
        "benchmark": "hotpotqa",
        "total_population": len(items),
        "search_budget": budget,
        "pilot_budget": pilot_budget,
        "pilot_allocation": pilot_alloc,
        "pilot_variances": variances,
        "main_allocation": main_alloc,
        "final_distribution": dict(final_dist),
        "search_set_size": len(search_set),
        "test_set_size": len(items),
        "seed": seed,
    }
    return report


def build_ultradomain_sets(budget: int, seed: int) -> dict:
    print("\n" + "=" * 60, flush=True)
    print("  UltraDomain Stratified Sampling", flush=True)
    print("=" * 60, flush=True)

    items = load_ultradomain()
    print(f"  Total items: {len(items)}", flush=True)

    strata = stratify_items(items, assign_ultradomain_stratum)
    print(f"  Strata: {len(strata)}", flush=True)
    for k in sorted(strata.keys()):
        print(f"    {k}: {len(strata[k])} items", flush=True)

    rng = random.Random(seed + 1000)

    pilot_budget = max(2 * len(strata), len(items) // 10)
    pilot_budget = min(pilot_budget, budget)
    pilot_alloc = proportional_allocation(strata, pilot_budget)
    pilot_items = draw_sample(strata, pilot_alloc, rng)
    print(f"\n  Pilot: {len(pilot_items)} items (allocation: {pilot_alloc})", flush=True)

    variances = {k: 0.1 for k in strata}
    print("  Using uniform variance for UltraDomain (eval is expensive)", flush=True)

    remaining = budget - len(pilot_items)
    if remaining > 0:
        remaining_strata = {}
        for k, pool in strata.items():
            pilot_ids = {id(it) for it in pilot_items if assign_ultradomain_stratum(it) == k}
            leftover = [it for it in pool if id(it) not in pilot_ids]
            remaining_strata[k] = leftover

        main_alloc = neyman_allocation(remaining_strata, variances, remaining)
        main_items = draw_sample(remaining_strata, main_alloc, rng)
        print(f"  Main draw: {len(main_items)} items (allocation: {main_alloc})", flush=True)
        search_set = pilot_items + main_items
    else:
        main_alloc = {}
        search_set = pilot_items

    print(f"\n  Final search set: {len(search_set)} items", flush=True)
    final_dist = defaultdict(int)
    for it in search_set:
        final_dist[assign_ultradomain_stratum(it)] += 1
    for k in sorted(final_dist.keys()):
        print(f"    {k}: {final_dist[k]}", flush=True)

    out_search = OUTPUT_DIR / "ultradomain" / "ultradomain_search_stratified.json"
    with open(out_search, "w", encoding="utf-8") as f:
        json.dump(search_set, f, ensure_ascii=False, indent=1)
    print(f"\n  Saved search set: {out_search} ({len(search_set)} items)", flush=True)

    out_test = OUTPUT_DIR / "ultradomain" / "ultradomain_test_all.json"
    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    print(f"  Saved test set: {out_test} ({len(items)} items)", flush=True)

    report = {
        "benchmark": "ultradomain",
        "total_population": len(items),
        "search_budget": budget,
        "pilot_budget": pilot_budget,
        "pilot_allocation": pilot_alloc,
        "pilot_variances": variances,
        "main_allocation": main_alloc,
        "final_distribution": dict(final_dist),
        "search_set_size": len(search_set),
        "test_set_size": len(items),
        "seed": seed,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Build stratified search/test sets")
    parser.add_argument("--hotpotqa-budget", type=int, default=400)
    parser.add_argument("--ultradomain-budget", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip pilot evaluation, use uniform variance")
    parser.add_argument("--output-report", type=str, default=None,
                        help="Save sampling report to JSON")
    args = parser.parse_args()

    reports = []

    hq_report = build_hotpotqa_sets(args.hotpotqa_budget, args.seed, args.skip_eval)
    reports.append(hq_report)

    try:
        ud_report = build_ultradomain_sets(args.ultradomain_budget, args.seed)
        reports.append(ud_report)
    except FileNotFoundError as e:
        print(f"\n  UltraDomain skipped: {e}", flush=True)

    if args.output_report:
        out = Path(args.output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        print(f"\n  Sampling report saved to {out}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
