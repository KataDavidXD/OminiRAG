"""Real-LLM benchmark demo across HotpotQA, ALCE, and UltraDomain.

Loads full HuggingFace datasets via the ``Benchmark_Sampling`` SDK, samples a
small budget, runs a real LLM (OpenAI-compatible) through the
``rag_contracts`` ``SimpleLLMGeneration`` component, and scores the outputs
with the standard OmniRAG benchmark adapters.

Usage:
    python demos/run_real_llm_benchmark.py --benchmark hotpotqa --budget 20
    python demos/run_real_llm_benchmark.py --benchmark alce --budget 20
    python demos/run_real_llm_benchmark.py --benchmark ultradomain --budget 20
    python demos/run_real_llm_benchmark.py --all --budget 20

Reads .env for LLM_API_KEY / LLM_BASE_URL / DEFAULT_LLM (or the OPENAI_*
equivalents). Writes a summary JSON to demos/results/benchmark_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLING_ROOT = PROJECT_ROOT / "Benchmark_Sampling"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SAMPLING_ROOT))


def _load_env() -> None:
    """Tiny .env loader so the demo runs without python-dotenv installed."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


_load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
MODEL = os.environ.get("OPENAI_MODEL") or os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

HF_HUB = Path(os.environ.get("HF_HUB_DIR", str(Path.home() / ".cache" / "huggingface" / "hub")))
HOTPOTQA_ROOT = HF_HUB / "datasets--hotpotqa--hotpot_qa" / "snapshots" / "1908d6afbbead072334abe2965f91bd2709910ab"
ALCE_ROOT = HF_HUB / "datasets--princeton-nlp--ALCE-data" / "snapshots" / "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
ULTRADOMAIN_ROOT = HF_HUB / "datasets--TommyChien--UltraDomain" / "snapshots" / "aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"

SEPARATOR = "=" * 72


def _safe_print(*args: object) -> None:
    """Print plain ASCII; replace any non-encodable chars to dodge GBK issues."""
    text = " ".join(str(a) for a in args)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


from bsamp.loader.hotpot_qa import HotpotQAAPI  # noqa: F401 -- available for manual use
from bsamp.loader.ALCE import ALCEAPI  # noqa: F401
from bsamp.loader.UltraDomain import UltraDomainAPI  # noqa: F401


# ---------------------------------------------------------------------------
# Generation component factory
# ---------------------------------------------------------------------------

def _build_generation(generator: str, llm: "LLM"):
    """Build a rag_contracts Generation component by name.

    'simple'  -> SimpleLLMGeneration (default, concise answers)
    'longrag' -> LongRAGGeneration  (extractive reader, better F1 on ALCE)
    'selfrag' -> SelfRAGGeneration  (evidence-scoring reader, synthetic shim)
    """
    if generator == "simple":
        from rag_contracts import SimpleLLMGeneration
        return SimpleLLMGeneration(llm=llm)

    if generator == "longrag":
        import os
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            from rag_contracts import SimpleLLMGeneration
            _safe_print("  [WARN] No API key for LongRAG shim; falling back to SimpleLLMGeneration")
            return SimpleLLMGeneration(llm=llm)

        class _LLMInferenceShim:
            def __init__(self):
                from openai import OpenAI
                base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
                self.client = OpenAI(api_key=api_key, base_url=base or None)
                self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

            def _ask(self, context, query, titles):
                titles_str = ", ".join(titles) if titles else "N/A"
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0.1, max_tokens=300,
                    messages=[
                        {"role": "system",
                         "content": "You are an expert reader. Extract the answer "
                         "from the provided context. Be concise and precise."},
                        {"role": "user",
                         "content": f"Titles: {titles_str}\n\nContext:\n{context[:4000]}\n\n"
                         f"Question: {query}\n\nAnswer:"},
                    ],
                )
                ans = (resp.choices[0].message.content or "").strip()
                return ans, ans

            def predict_nq(self, context, query, titles):
                return self._ask(context, query, titles)

            def predict_hotpotqa(self, context, query, titles):
                return self._ask(context, query, titles)

        from longRAG_example.longrag_langgraph.adapters import LongRAGGeneration
        return LongRAGGeneration(llm_inference=_LLMInferenceShim())

    if generator == "selfrag":
        from rag_contracts import SimpleLLMGeneration
        _safe_print("  [WARN] Self-RAG requires vLLM endpoint; falling back to SimpleLLMGeneration")
        return SimpleLLMGeneration(llm=llm)

    raise ValueError(f"Unknown generator: {generator!r}")


# ---------------------------------------------------------------------------
# Real LLM wrapper (OpenAI-compatible)
# ---------------------------------------------------------------------------

class LLM:
    """Minimal OpenAI-compatible LLM wrapper with token + call counting."""

    def __init__(self, model: str = MODEL, api_key: str = OPENAI_API_KEY,
                 base_url: str = OPENAI_BASE_URL):
        from rag_contracts import WTBCacheConfig, WTBCachedLLM

        cache_config = WTBCacheConfig.from_env()
        self._wtb_llm = None
        if cache_config.cache_active:
            self._wtb_llm = WTBCachedLLM(
                config=cache_config,
                system_name="real_llm_benchmark",
                node_path="demos.run_real_llm_benchmark.llm",
                model=model,
            )
            self._client = None
            self._model = model
            self.calls = 0
            self.total_tokens = 0
            self.cache_hits = 0
            self.cache_misses = 0
            return

        from openai import OpenAI

        kwargs = {"api_key": api_key or "sk-placeholder"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self.calls = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def complete(
        self, system: str, user: str,
        temperature: float = 0.1, max_tokens: int = 300,
    ) -> str:
        if self._wtb_llm is not None:
            text = self._wtb_llm.complete(
                system,
                user,
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            metadata = self._wtb_llm.wtb_cache_metadata()
            if metadata.get("last_cache_hit"):
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                self.calls += 1
            return text

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.calls += 1
        if resp.usage:
            self.total_tokens += resp.usage.total_tokens
        return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# SDK-backed sampling helper
# ---------------------------------------------------------------------------

def _sample_via_engine(adapter_cls, adapter_kwargs: dict, budget: int, seed: int,
                       method: str = "proportional"):
    """Use bsamp SamplingEngine to draw a stratified sample.

    Returns (items: list[BenchmarkItem], result: SamplingResult).
    """
    from bsamp.sampling.engine import SamplingEngine
    adapter = adapter_cls(**adapter_kwargs)
    engine = SamplingEngine(adapter=adapter, method=method, budget=budget, seed=seed)
    result = engine.run()
    return result.items, result


def _format_estimate(result) -> dict | None:
    """Extract estimate summary from SamplingResult for the report JSON."""
    est = result.estimate
    if est is None:
        return None
    return {
        "mean": est.mean,
        "std_error": est.std_error,
        "ci_lower": est.ci_lower,
        "ci_upper": est.ci_upper,
        "n_evaluated": est.n_evaluated,
        "n_total": est.n_total,
    }


def run_hotpotqa(llm: LLM, budget: int, seed: int = 42, generator: str = "simple") -> dict:
    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
    from bsamp.sampling.adapters.hotpotqa import HotpotQAAdapter

    _safe_print(f"\n{SEPARATOR}\n  BENCHMARK: HotpotQA  (Multi-hop QA, EM/F1)  generator={generator}\n{SEPARATOR}")
    t0 = time.time()

    sampled_items, sampling_result = _sample_via_engine(
        HotpotQAAdapter, {"root_dir": str(HOTPOTQA_ROOT)}, budget, seed,
    )
    pop_size = sum(sampling_result.strata_summary.values())
    _safe_print(f"  Population: {pop_size} items  ({time.time()-t0:.1f}s)")
    _safe_print(f"  Sampled: {len(sampled_items)} items across {len(sampling_result.strata_summary)} strata")

    eval_items: list[dict] = []
    for bitem in sampled_items:
        p = bitem.payload
        chunks = {}
        for idx, (title, sents) in enumerate(zip(
            p.get("context_titles", []), p.get("context_sentences", [])
        )):
            cid = f"c{idx}"
            chunks[cid] = {"content": f"[{title}] " + " ".join(sents), "doc_ids": [title]}
        eval_items.append({
            "question": p["question"],
            "answer": bitem.target["answer"],
            "query_id": bitem.item_id,
            "chunks": chunks,
            "_stratum": bitem.stratum,
        })

    adapter = HotpotQABenchmarkAdapter()
    gen = _build_generation(generator, llm)
    result = adapter.evaluate_generation(eval_items, gen)

    # Per-stratum F1
    per_stratum: dict[str, dict] = defaultdict(lambda: {"n": 0, "f1_sum": 0.0, "em_sum": 0})
    for src, item in zip(eval_items, result.per_item):
        s = src["_stratum"]
        per_stratum[s]["n"] += 1
        per_stratum[s]["f1_sum"] += item["f1"]
        per_stratum[s]["em_sum"] += item["em"]
    per_stratum_summary = {
        k: {"n": v["n"], "f1": 100 * v["f1_sum"] / max(v["n"], 1)}
        for k, v in per_stratum.items()
    }

    elapsed = time.time() - t0
    _safe_print(f"\n  Avg EM:  {result.avg_em:.1f}%   Avg F1:  {result.avg_f1:.1f}%")
    _safe_print(f"  Per-stratum F1: {per_stratum_summary}")
    _safe_print(f"  LLM calls: {llm.calls}   Tokens: {llm.total_tokens}   Elapsed: {elapsed:.1f}s")
    for it in result.per_item[:3]:
        _safe_print(f"    Q: {it['question'][:70]}")
        _safe_print(f"      gold={it['answer']!r}  pred={it['output'][:60]!r}  F1={it['f1']:.2f}")

    return {
        "benchmark": "hotpotqa",
        "n": len(eval_items),
        "avg_em": result.avg_em,
        "avg_f1": result.avg_f1,
        "elapsed": elapsed,
        "llm_calls": llm.calls,
        "tokens": llm.total_tokens,
        "per_stratum": per_stratum_summary,
        "sampling_estimate": _format_estimate(sampling_result),
    }


# ---------------------------------------------------------------------------
# ALCE
# ---------------------------------------------------------------------------

def run_alce(llm: LLM, budget: int, seed: int = 42, subset: str = "asqa", generator: str = "simple") -> dict:
    from benchmark.alce_adapter import ALCEBenchmarkAdapter
    from bsamp.sampling.adapters.alce import ALCEAdapter

    _safe_print(f"\n{SEPARATOR}\n  BENCHMARK: ALCE / {subset}  (long-form QA, F1/STR-EM)  generator={generator}\n{SEPARATOR}")
    t0 = time.time()

    sampled_items, sampling_result = _sample_via_engine(
        ALCEAdapter, {"root_dir": str(ALCE_ROOT), "subsets": [subset]}, budget, seed,
    )
    pop_size = sum(sampling_result.strata_summary.values())
    _safe_print(f"  Population: {pop_size} items in '{subset}'  ({time.time()-t0:.1f}s)")
    _safe_print(f"  Sampled: {len(sampled_items)} items across {len(sampling_result.strata_summary)} strata")

    eval_items = [
        {
            "question": bitem.payload["question"],
            "answer": bitem.target["answer"],
            "docs": bitem.payload.get("docs", [])[:5],
            "qa_pairs": bitem.target.get("qa_pairs", []),
            "query_id": bitem.item_id,
        }
        for bitem in sampled_items
    ]

    calls_before = llm.calls
    tokens_before = llm.total_tokens
    adapter = ALCEBenchmarkAdapter()
    gen = _build_generation(generator, llm)
    result = adapter.evaluate_generation(eval_items, gen)

    elapsed = time.time() - t0
    _safe_print(f"\n  Avg F1:     {result.avg_f1:.1f}%   Avg EM:    {result.avg_exact:.1f}%")
    _safe_print(f"  Avg STR-EM: {result.avg_str_em:.1f}%   Avg length: {result.avg_length:.1f} words")
    _safe_print(f"  LLM calls: {llm.calls - calls_before}   Tokens: {llm.total_tokens - tokens_before}   Elapsed: {elapsed:.1f}s")
    for it in result.per_item[:3]:
        _safe_print(f"    Q: {it['question'][:70]}")
        _safe_print(f"      pred={it['output'][:80]!r}")
        _safe_print(f"      gold={str(it['answer'])[:80]!r}  F1={it['f1']:.2f}  STR-EM={it['str_em']:.2f}")

    return {
        "benchmark": "alce",
        "subset": subset,
        "n": len(eval_items),
        "avg_f1": result.avg_f1,
        "avg_em": result.avg_exact,
        "avg_str_em": result.avg_str_em,
        "avg_length": result.avg_length,
        "elapsed": elapsed,
        "llm_calls": llm.calls - calls_before,
        "tokens": llm.total_tokens - tokens_before,
        "sampling_estimate": _format_estimate(sampling_result),
    }


# ---------------------------------------------------------------------------
# QAMPARI
# ---------------------------------------------------------------------------

def run_qampari(llm: LLM, budget: int, seed: int = 42, generator: str = "simple") -> dict:
    from benchmark.alce_adapter import QampariBenchmarkAdapter
    from bsamp.sampling.adapters.alce import ALCEAdapter

    _safe_print(f"\n{SEPARATOR}\n  BENCHMARK: ALCE / qampari  (list QA, Precision/Recall/F1)  generator={generator}\n{SEPARATOR}")
    t0 = time.time()

    sampled_items, sampling_result = _sample_via_engine(
        ALCEAdapter, {"root_dir": str(ALCE_ROOT), "subsets": ["qampari"]}, budget, seed,
    )
    pop_size = sum(sampling_result.strata_summary.values())
    _safe_print(f"  Population: {pop_size} items in 'qampari'  ({time.time()-t0:.1f}s)")
    _safe_print(f"  Sampled: {len(sampled_items)} items across {len(sampling_result.strata_summary)} strata")

    eval_items = [
        {
            "question": bitem.payload["question"],
            "answers": [
                qp.get("short_answers", [])
                for qp in bitem.target.get("qa_pairs", [])
            ],
            "docs": bitem.payload.get("docs", [])[:5],
            "query_id": bitem.item_id,
        }
        for bitem in sampled_items
    ]

    calls_before = llm.calls
    tokens_before = llm.total_tokens
    adapter = QampariBenchmarkAdapter()
    gen = _build_generation(generator, llm)
    result = adapter.evaluate_generation(eval_items, gen)

    elapsed = time.time() - t0
    _safe_print(f"\n  Avg Precision: {result.avg_precision:.1f}%   Avg Recall: {result.avg_recall:.1f}%")
    _safe_print(f"  Avg F1:        {result.avg_f1:.1f}%   Avg F1-top5: {result.avg_f1_top5:.1f}%")
    _safe_print(f"  Avg #preds:    {result.avg_num_preds:.1f}")
    _safe_print(f"  LLM calls: {llm.calls - calls_before}   Tokens: {llm.total_tokens - tokens_before}   Elapsed: {elapsed:.1f}s")
    for it in result.per_item[:3]:
        _safe_print(f"    Q: {it['question'][:70]}")
        _safe_print(f"      pred={it['output'][:80]!r}")
        _safe_print(f"      P={it['precision']:.2f}  R={it['recall']:.2f}  F1={it['f1']:.2f}")

    return {
        "benchmark": "qampari",
        "n": len(eval_items),
        "avg_precision": result.avg_precision,
        "avg_recall": result.avg_recall,
        "avg_recall_top5": result.avg_recall_top5,
        "avg_f1": result.avg_f1,
        "avg_f1_top5": result.avg_f1_top5,
        "avg_num_preds": result.avg_num_preds,
        "elapsed": elapsed,
        "llm_calls": llm.calls - calls_before,
        "tokens": llm.total_tokens - tokens_before,
        "sampling_estimate": _format_estimate(sampling_result),
    }


# ---------------------------------------------------------------------------
# UltraDomain (FIXED)
# ---------------------------------------------------------------------------

def run_ultradomain(llm: LLM, budget: int, seed: int = 42,
                    domains: tuple[str, ...] = ("physics", "cs", "legal"),
                    generator: str = "simple") -> dict:
    from benchmark.ultradomain_adapter import UltraDomainBenchmarkAdapter
    from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter

    _safe_print(f"\n{SEPARATOR}\n  BENCHMARK: UltraDomain  (domain QA, LLM-as-judge)  generator={generator}\n{SEPARATOR}")
    t0 = time.time()

    sampled_items, sampling_result = _sample_via_engine(
        UltraDomainAdapter,
        {"root_dir": str(ULTRADOMAIN_ROOT), "target_domains": list(domains)},
        budget, seed,
    )
    pop_size = sum(sampling_result.strata_summary.values())
    _safe_print(f"  Population: {pop_size} items in domains {domains}  ({time.time()-t0:.1f}s)")
    _safe_print(f"  Sampled: {len(sampled_items)} items across {len(sampling_result.strata_summary)} strata")

    eval_items: list[dict] = []
    for bitem in sampled_items:
        ctx_text = (bitem.payload.get("context") or "")[:3000]
        domain = bitem.metadata.get("domain", "unknown")
        chunks = {"c0": {"content": ctx_text, "doc_ids": [bitem.metadata.get("title") or domain]}} if ctx_text else {}
        eval_items.append({
            "question": bitem.payload.get("query") or "",
            "answer": bitem.target.get("answer") or "",
            "domain": domain,
            "query_id": bitem.item_id,
            "chunks": chunks,
        })

    calls_before = llm.calls
    tokens_before = llm.total_tokens
    adapter = UltraDomainBenchmarkAdapter(llm_complete=llm.complete)
    gen = _build_generation(generator, llm)
    result = adapter.evaluate_generation(eval_items, gen)

    # Per-domain stats
    per_domain: dict[str, dict] = defaultdict(lambda: {"n": 0, "comp": 0.0, "div": 0.0, "emp": 0.0})
    for it in result.per_item:
        d = it["domain"]
        per_domain[d]["n"] += 1
        per_domain[d]["comp"] += it["comprehensiveness"]
        per_domain[d]["div"] += it["diversity"]
        per_domain[d]["emp"] += it["empowerment"]
    per_domain_summary = {
        d: {
            "n": v["n"],
            "comp": round(v["comp"] / max(v["n"], 1), 2),
            "div": round(v["div"] / max(v["n"], 1), 2),
            "emp": round(v["emp"] / max(v["n"], 1), 2),
        }
        for d, v in per_domain.items()
    }

    elapsed = time.time() - t0
    _safe_print(f"\n  Avg comprehensiveness: {result.avg_comprehensiveness:.2f}/5")
    _safe_print(f"  Avg diversity:         {result.avg_diversity:.2f}/5")
    _safe_print(f"  Avg empowerment:       {result.avg_empowerment:.2f}/5")
    _safe_print(f"  Avg length:            {result.avg_length:.0f} words")
    _safe_print(f"  Per-domain: {per_domain_summary}")
    _safe_print(f"  LLM calls: {llm.calls - calls_before}   Tokens: {llm.total_tokens - tokens_before}   Elapsed: {elapsed:.1f}s")
    for it in result.per_item[:3]:
        _safe_print(f"    [{it['domain']}] Q: {it['question'][:60]}")
        _safe_print(f"      pred={it['output'][:80]!r}")
        _safe_print(f"      C={it['comprehensiveness']:.0f}  D={it['diversity']:.0f}  E={it['empowerment']:.0f}")

    return {
        "benchmark": "ultradomain",
        "n": len(eval_items),
        "avg_comp": result.avg_comprehensiveness,
        "avg_div": result.avg_diversity,
        "avg_emp": result.avg_empowerment,
        "avg_length": result.avg_length,
        "elapsed": elapsed,
        "llm_calls": llm.calls - calls_before,
        "tokens": llm.total_tokens - tokens_before,
        "per_domain": per_domain_summary,
        "sampling_estimate": _format_estimate(sampling_result),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Real LLM benchmark demo")
    parser.add_argument("--benchmark", choices=["hotpotqa", "alce", "ultradomain"])
    parser.add_argument("--all", action="store_true", help="Run all three benchmarks")
    parser.add_argument("--budget", type=int, default=20, help="Number of items per benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator", choices=["simple", "longrag", "selfrag"], default="simple",
                        help="Generation component: simple (default), longrag, selfrag")
    parser.add_argument("--alce-subset", default="asqa", choices=["asqa", "qampari", "eli5"])
    parser.add_argument("--ultradomain-domains", nargs="+", default=["physics", "cs", "legal"])
    parser.add_argument("--out", default="demos/results/benchmark_results.json")
    args = parser.parse_args()

    if not args.all and not args.benchmark:
        parser.error("Either --all or --benchmark must be provided")

    _safe_print(SEPARATOR)
    _safe_print(f"  OMINIRAG REAL-LLM BENCHMARK DEMO")
    _safe_print(f"  Model: {MODEL}    Base URL: {OPENAI_BASE_URL or '(default OpenAI)'}")
    _safe_print(f"  Budget per benchmark: {args.budget}    Seed: {args.seed}    Generator: {args.generator}")
    _safe_print(SEPARATOR)

    llm = LLM()
    results: list[dict] = []
    targets = ["hotpotqa", "alce", "ultradomain"] if args.all else [args.benchmark]

    for name in targets:
        if name == "hotpotqa":
            results.append(run_hotpotqa(llm, args.budget, args.seed, generator=args.generator))
        elif name == "alce":
            if args.alce_subset == "qampari":
                results.append(run_qampari(llm, args.budget, args.seed, generator=args.generator))
            else:
                results.append(run_alce(llm, args.budget, args.seed, subset=args.alce_subset, generator=args.generator))
        elif name == "ultradomain":
            results.append(run_ultradomain(llm, args.budget, args.seed,
                                           domains=tuple(args.ultradomain_domains),
                                           generator=args.generator))

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL,
            "base_url": OPENAI_BASE_URL,
            "budget": args.budget,
            "seed": args.seed,
            "generator": args.generator,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    _safe_print(f"\n{SEPARATOR}\n  Results written to: {out_path}\n{SEPARATOR}")


if __name__ == "__main__":
    main()
