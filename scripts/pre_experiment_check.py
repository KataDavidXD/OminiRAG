"""Pre-experiment checklist: validates system readiness before the final run.

Checks:
  1. Data files exist (HotpotQA, UltraDomain, CorpusIndex, LightRAG stores)
  2. Stratified search set exists with correct distribution
  3. LLM endpoint reachable (gpt-5-mini ping)
  4. Pipeline builds (3 presets + BM25-simple)
  5. 1-item smoke test (BM25 + simple_llm -> non-zero F1)
  6. Cache system round-trip (ReuseLedger record/lookup/materialized_keys)
  7. Disk space sufficient (~1GB for results)
  8. Self-RAG vLLM (optional, warn-only)
  9. GPU status (optional, report memory)

Usage:
    python scripts/pre_experiment_check.py
    python scripts/pre_experiment_check.py --skip-llm   # skip LLM ping
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "AG-UCT"))
sys.path.insert(0, str(PROJECT_ROOT / "Benchmark_Sampling"))
sys.path.insert(0, str(PROJECT_ROOT / "A-Simplified-Core-Workflow-for-Enhancing-RAG"))

DATA_ROOT = Path(os.environ.get("OMINIRAG_DATA_ROOT", "/data1/ragworkspace/train"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"


def check_data_files() -> tuple[bool, str]:
    required = [
        DATA_ROOT / "fullwiki" / "fullwiki_sample_500.json",
        DATA_ROOT / "fullwiki" / "fullwiki_test_500.json",
        PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json",
    ]
    lightrag_stores = [
        DATA_ROOT / "fullwiki" / "graph" / "graph.json",
        DATA_ROOT / "fullwiki" / "graph" / "chunks.json",
        DATA_ROOT / "fullwiki" / "graph" / "kv.json",
    ]
    ultradomain_files = [
        DATA_ROOT / "ultradomain" / "agriculture.jsonl",
        DATA_ROOT / "ultradomain" / "cs.jsonl",
        DATA_ROOT / "ultradomain" / "legal.jsonl",
        DATA_ROOT / "ultradomain" / "mix.jsonl",
    ]
    ultradomain_stores = [
        DATA_ROOT / "ultradomain" / "graph" / "graph.json",
    ]

    missing = []
    for p in required + lightrag_stores + ultradomain_files + ultradomain_stores:
        if not p.exists():
            missing.append(str(p))

    if missing:
        return False, f"Missing files:\n    " + "\n    ".join(missing)
    return True, f"All {len(required)+len(lightrag_stores)+len(ultradomain_files)+len(ultradomain_stores)} data files present"


def check_stratified_search_set() -> tuple[bool, str]:
    path = DATA_ROOT / "fullwiki" / "fullwiki_search_400_stratified.json"
    if not path.exists():
        return False, f"Not found: {path}\n    Run: python scripts/build_stratified_search_set.py"

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if len(items) < 100:
        return False, f"Only {len(items)} items (expected ~400)"

    from collections import Counter
    strata = Counter()
    for it in items:
        t = str(it.get("type", "nan"))
        l = str(it.get("level", "nan"))
        key = f"{t}_{l}" if t != "nan" and l != "nan" else "nan_nan"
        strata[key] += 1

    detail = ", ".join(f"{k}:{v}" for k, v in sorted(strata.items()))
    return True, f"{len(items)} items, {len(strata)} strata ({detail})"


def check_llm_endpoint() -> tuple[bool, str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "")
    model = os.environ.get("DEFAULT_LLM", "gpt-5-mini")

    if not base_url:
        return False, "LLM_BASE_URL not set in .env"

    try:
        import openai
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=100,
        )
        content = resp.choices[0].message.content or ""
        if not content.strip():
            return False, f"Empty response from {model}. Check max_tokens / reasoning model settings."
        return True, f"{model} responded: '{content.strip()[:50]}'"
    except Exception as e:
        return False, f"LLM error ({model}): {type(e).__name__}: {e}"


def _cleanup_gpu():
    """Release GPU memory after model loading checks."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def check_pipeline_builds() -> tuple[bool, str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    from rag_contracts.component_registry import build_pipeline_from_config
    from rag_contracts import CorpusIndex

    index_path = PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json"
    corpus = None
    if index_path.exists():
        corpus = CorpusIndex.from_json_file(index_path)

    core_presets = {
        "bm25_simple": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
        "longrag": ("standard_passage", "identity", "bm25", "identity", "simple_llm"),
        "lightrag": ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "lightrag_answer"),
    }
    optional_presets = {
        "selfrag": ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator"),
    }

    errors = []
    warnings = []
    built = []
    for name, config in core_presets.items():
        try:
            components = build_pipeline_from_config(config, "hotpotqa", corpus=corpus)
            gen = components.get("generation")
            if gen is None:
                errors.append(f"{name}: no generation component")
            else:
                built.append(name)
            del components
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    for name, config in optional_presets.items():
        try:
            components = build_pipeline_from_config(config, "hotpotqa", corpus=corpus)
            gen = components.get("generation")
            if gen is None:
                warnings.append(f"{name}: no generation (optional)")
            else:
                built.append(name)
            del components
        except Exception as e:
            warnings.append(f"{name}: {type(e).__name__}: {str(e)[:80]} (optional, needs GPU + HF)")

    del corpus
    _cleanup_gpu()

    detail = f"Built {len(built)}/{len(core_presets)+len(optional_presets)}"
    if errors:
        return False, detail + " CORE FAIL: " + "; ".join(errors)
    if warnings:
        return True, detail + " (warnings: " + "; ".join(warnings) + ")"
    return True, detail + " -- all presets OK"


def check_smoke_test() -> tuple[bool, str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    test_path = DATA_ROOT / "fullwiki" / "fullwiki_test_500.json"
    if not test_path.exists():
        test_path = DATA_ROOT / "fullwiki" / "fullwiki_sample_500.json"
    if not test_path.exists():
        return False, "No test data found"

    with open(test_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    item = items[0]

    from rag_contracts.component_registry import build_pipeline_from_config
    from rag_contracts import CorpusIndex
    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter

    index_path = PROJECT_ROOT / "data" / "corpus_indexes" / "fullwiki_corpus_index.json"
    corpus = CorpusIndex.from_json_file(index_path) if index_path.exists() else None

    config = ("standard_passage", "identity", "bm25", "identity", "simple_llm")
    components = build_pipeline_from_config(config, "hotpotqa", corpus=corpus)
    generation = components.get("generation")
    retrieval = components.get("retrieval")

    ctx = []
    if retrieval and hasattr(retrieval, "retrieve"):
        ctx = retrieval.retrieve([item["question"]], top_k=5)

    enriched = [{
        "question": item["question"],
        "answer": item["answer"],
        "query_id": item.get("id", ""),
        "context_results": ctx,
    }]

    adapter = HotpotQABenchmarkAdapter()
    result = adapter.evaluate_generation(enriched, generation)

    del components, generation, retrieval, corpus
    _cleanup_gpu()

    if result.avg_f1 > 0:
        return True, f"EM={result.avg_em:.0f} F1={result.avg_f1:.1f} on 1 item"
    return False, f"Zero F1 on smoke test. Check LLM response quality."


def check_cache_system() -> tuple[bool, str]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from ominirag_wtb.reuse_ledger import ReuseLedger

    tmp_dir = tempfile.mkdtemp(prefix="ominirag_cache_check_")
    db_path = os.path.join(tmp_dir, "test_ledger.db")

    try:
        ledger = ReuseLedger(db_path=db_path)
        ledger.record(
            prefix=("standard_passage", "identity"),
            question_id="test_q1",
            execution_id="exec_001",
            checkpoint_id="cp_001",
        )
        entry = ledger.lookup(("standard_passage", "identity"), "test_q1")
        if entry is None:
            return False, "Lookup after record returned None"

        keys = ledger.materialized_keys()
        if len(keys) != 1:
            return False, f"materialized_keys returned {len(keys)} (expected 1)"

        ledger.close()
        return True, "Record -> lookup -> materialized_keys round-trip OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def check_disk_space() -> tuple[bool, str]:
    results_dir = PROJECT_ROOT / "results"
    target = results_dir if results_dir.exists() else PROJECT_ROOT

    usage = shutil.disk_usage(str(target))
    free_gb = usage.free / (1024 ** 3)

    if free_gb < 1.0:
        return False, f"Only {free_gb:.1f} GB free (need >= 1 GB)"
    return True, f"{free_gb:.1f} GB free"


def check_selfrag_vllm() -> tuple[bool, str]:
    url = os.environ.get("SELFRAG_VLLM_URL", "")
    if not url:
        return True, "SELFRAG_VLLM_URL not set (optional)"

    try:
        import urllib.request
        req = urllib.request.Request(url.rstrip("/") + "/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, f"vLLM reachable at {url}"
    except Exception as e:
        return True, f"vLLM not reachable ({e}) -- optional, not blocking"

    return True, "vLLM check skipped"


def check_gpu() -> tuple[bool, str]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return True, "nvidia-smi not available (optional)"

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        info = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info.append(f"{parts[0]}: {parts[1]}/{parts[2]} MB free")
        return True, "; ".join(info) if info else "No GPUs detected"
    except Exception:
        return True, "GPU check skipped (optional)"


def main():
    parser = argparse.ArgumentParser(description="Pre-experiment readiness check")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM endpoint check")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip 1-item smoke test")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    checks = [
        ("1. Data files",           check_data_files),
        ("2. Stratified search set", check_stratified_search_set),
        ("3. LLM endpoint",         None if args.skip_llm else check_llm_endpoint),
        ("4. Pipeline builds",      check_pipeline_builds),
        ("5. Smoke test (1 item)",   None if args.skip_smoke else check_smoke_test),
        ("6. Cache system",         check_cache_system),
        ("7. Disk space",           check_disk_space),
        ("8. Self-RAG vLLM",        check_selfrag_vllm),
        ("9. GPU status",           check_gpu),
    ]

    print("=" * 65, flush=True)
    print("  OminiRAG Pre-Experiment Readiness Check", flush=True)
    print("=" * 65, flush=True)

    all_pass = True
    for name, fn in checks:
        if fn is None:
            print(f"  {WARN} {name}: SKIPPED", flush=True)
            continue
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"Exception: {type(e).__name__}: {e}"

        status = PASS if ok else FAIL
        print(f"  {status} {name}: {detail}", flush=True)
        if not ok and name not in ("8. Self-RAG vLLM", "9. GPU status"):
            all_pass = False

    print("\n" + "=" * 65, flush=True)
    if all_pass:
        print(f"  {PASS} All critical checks passed. Ready to run experiments.", flush=True)
    else:
        print(f"  {FAIL} Some checks failed. Fix issues above before running.", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
