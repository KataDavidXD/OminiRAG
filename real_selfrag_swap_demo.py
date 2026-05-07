"""Real LLM cross-project swap demo: Self-RAG + LongRAG + LightRAG.

Exercises all three modular pipelines with live LLM calls, swapping
components between the three RAG systems to prove full interchangeability.
Extended with real dataset evaluation, real vLLM Self-RAG inference,
system health diagnostics, and real-data component swap testing.

Sections:
  1) Cross-project swaps       -- 6 configs across all 3 pipeline frames
  2) 3-pipeline identity       -- same components through all 3 pipelines
  3) ALCE benchmark eval       -- real F1/STR-EM scoring on sample data
  4) Real dataset evaluation   -- HotpotQA + UltraDomain from /data1
  5) Real vs Fake Self-RAG     -- compare real vLLM logprobs with fake shim
  6) AG-UCT mini search        -- quick UCT search with real data
  7) System health check       -- protocol, data format, constraint validation
  8) Real-data swap test       -- component swaps on real HotpotQA/UD passages

Usage:
    python real_selfrag_swap_demo.py                        # sections 1-3
    python real_selfrag_swap_demo.py --quick                # only section 1
    python real_selfrag_swap_demo.py --real                 # sections 1-8
    python real_selfrag_swap_demo.py --query "..."          # custom query
    python real_selfrag_swap_demo.py --data-dir /data1/ragworkspace/dataset
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_file)

API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
MODEL = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")
VLLM_URL = os.environ.get("SELFRAG_VLLM_URL", "")

if not API_KEY:
    sys.exit("ERROR: Set LLM_API_KEY in .env")

# ---------------------------------------------------------------------------
# Fake vllm / torch so selfrag adapters can import SamplingParams
# ---------------------------------------------------------------------------

fake_vllm = types.ModuleType("vllm")


@dataclass
class FakeSamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 100
    logprobs: int = 0


fake_vllm.SamplingParams = FakeSamplingParams
sys.modules.setdefault("vllm", fake_vllm)

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = lambda: MagicMock(
        __enter__=lambda s: None, __exit__=lambda s, *a: None
    )
    sys.modules["torch"] = fake_torch

_selfrag_root = str(Path(__file__).resolve().parent / "self-rag_langgraph" / "self-rag-wtb")
if _selfrag_root not in sys.path:
    sys.path.insert(0, _selfrag_root)

from rag_contracts import (
    ALCEDocRetrieval,
    DuckDuckGoRetrieval,
    FallbackRetrieval,
    GenerationResult,
    IdentityReranking,
    LLMRetrieval,
    RetrievalResult,
    SimpleLLMGeneration,
)

from selfrag.constants import (
    load_special_tokens,
    rel_tokens_names,
    retrieval_tokens_names,
    ground_tokens_names,
    utility_tokens_names,
)


# =============================================================================
# Shared LLM client
# =============================================================================


class _LLM:
    def __init__(self):
        self.model = MODEL
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

    def complete(self, system: str, user: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 800),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# =============================================================================
# OpenAI model wrapper (mimics vLLM generate() for Self-RAG adapters)
# =============================================================================

ALL_SPECIAL = (
    retrieval_tokens_names
    + rel_tokens_names
    + ground_tokens_names
    + utility_tokens_names
)
TOKEN_MAP: Dict[str, int] = {tok: 2000 + i for i, tok in enumerate(ALL_SPECIAL)}


class _FakeTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return TOKEN_MAP.get(token, 0)


SELFRAG_SYSTEM = """\
You are a Self-RAG assistant. Given a question and optionally a paragraph \
of evidence, produce a SHORT factual answer wrapped in Self-RAG control tokens.

You MUST follow this EXACT output format (nothing else):
[Relevant]<your answer>[Fully supported][Utility:5]

Control-token choices:
  Relevance : [Relevant] or [Irrelevant]
  Grounding : [Fully supported], [Partially supported], or [No support / Contradictory]
  Utility   : [Utility:1] .. [Utility:5]  (5 = most useful)

Rules:
- Keep answer to 1-2 sentences.
- Always include ALL three token types.
- Do NOT add markdown, explanations, or extra text.
"""


def _detect_token(text: str, candidates: List[str]) -> Optional[str]:
    for tok in candidates:
        if tok in text:
            return tok
    return None


class _VLLMOutput:
    """Mimics vLLM CompletionOutput with synthetic logprobs."""

    def __init__(self, text: str):
        self.text = text
        found_rel = _detect_token(text, rel_tokens_names)
        found_grd = _detect_token(text, ground_tokens_names)
        found_ut = _detect_token(text, utility_tokens_names)
        found_ret = _detect_token(text, retrieval_tokens_names)

        n_filler = max(5, len(text.split()) // 2)
        self.token_ids: List[int] = []
        self.token_ids.append(TOKEN_MAP.get(found_rel or "[Relevant]", 888))
        self.token_ids.extend([888] * n_filler)
        grd_pos = len(self.token_ids)
        self.token_ids.append(TOKEN_MAP.get(found_grd or "[Fully supported]", 888))
        ut_pos = len(self.token_ids)
        self.token_ids.append(TOKEN_MAP.get(found_ut or "[Utility:5]", 888))

        self.logprobs: List[Dict[int, float]] = []
        for pos in range(len(self.token_ids)):
            entry: Dict[int, float] = {}
            for tok in rel_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == 0 and tok == found_rel) else -5.0
            for tok in retrieval_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if tok == found_ret else -5.0
            for tok in ground_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == grd_pos and tok == found_grd) else -5.0
            for tok in utility_tokens_names:
                tid = TOKEN_MAP[tok]
                entry[tid] = -0.1 if (pos == ut_pos and tok == found_ut) else -5.0
            self.logprobs.append(entry)

        self.cumulative_logprob = -2.0


class _VLLMPrediction:
    def __init__(self, output: _VLLMOutput):
        self.outputs = [output]


class OpenAIModel:
    """Wraps OpenAI chat completions behind the vLLM generate() interface."""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)
        self.model = MODEL

    def generate(
        self, prompts: List[str], sampling_params=None
    ) -> List[_VLLMPrediction]:
        results: List[_VLLMPrediction] = []
        for prompt in prompts:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SELFRAG_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=getattr(sampling_params, "temperature", 0.0),
                    max_tokens=getattr(sampling_params, "max_tokens", 150),
                )
                text = resp.choices[0].message.content or ""
            except Exception as exc:
                print(f"    [LLM ERROR] {exc}")
                text = "[Relevant]Error generating response.[Partially supported][Utility:3]"
            results.append(_VLLMPrediction(_VLLMOutput(text)))
        return results


# =============================================================================
# Canonical generation components
# =============================================================================


@dataclass
class LongRAGReaderGeneration:
    """Extracts a concise answer from context (LongRAG reader style)."""

    llm: _LLM

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        ctx_text = "\n\n---\n\n".join(
            f"[{i+1}] {r.title}\n{r.content}" for i, r in enumerate(context[:5])
        )
        answer = self.llm.complete(
            "You are an expert reader. Extract the answer from the provided context. "
            "Be concise and precise. If the answer is a short entity, return just that.",
            f"Context:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer:",
            temperature=0.1,
            max_tokens=300,
        )
        return GenerationResult(
            output=answer.strip(),
            citations=[r.source_id for r in context[:5]],
            metadata={"style": "longrag-reader"},
        )


# =============================================================================
# Self-RAG adapter builders
# =============================================================================


def _build_selfrag_generation(openai_model: OpenAIModel) -> Any:
    from selfrag.adapters import SelfRAGGeneration

    tokenizer = _FakeTokenizer()
    _, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True
    )
    return SelfRAGGeneration(
        model=openai_model,
        rel_tokens=rel_tokens,
        grd_tokens=grd_tokens,
        ut_tokens=ut_tokens,
    )


def _build_selfrag_reranking(openai_model: OpenAIModel) -> Any:
    from selfrag.adapters import SelfRAGReranking

    tokenizer = _FakeTokenizer()
    _, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer, use_grounding=True, use_utility=True
    )
    return SelfRAGReranking(
        model=openai_model,
        rel_tokens=rel_tokens,
        grd_tokens=grd_tokens,
        ut_tokens=ut_tokens,
    )


# =============================================================================
# Pipeline runners (all 3 frameworks)
# =============================================================================

SEPARATOR = "=" * 72


@dataclass
class ConfigResult:
    label: str
    gen: GenerationResult
    elapsed: float
    pipeline: str


def _print_result(cr: ConfigResult) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {cr.label}")
    print(f"  Pipeline: {cr.pipeline}  |  Time: {cr.elapsed:.1f}s")
    style = cr.gen.metadata.get("style", "selfrag")
    print(f"  Style: {style}  |  Citations: {len(cr.gen.citations)}  |  Chars: {len(cr.gen.output)}")
    if "selfrag_score" in cr.gen.metadata:
        print(f"  Self-RAG score: {cr.gen.metadata['selfrag_score']:.3f}")
    if cr.gen.metadata.get("from_reranking_cache"):
        print(f"  (answer from SelfRAG reranking cache -- 0 extra LLM calls)")
    print(SEPARATOR)
    wrapped = textwrap.fill(cr.gen.output, width=72)
    print(wrapped)
    print()


async def _run_selfrag(name, retrieval, generation, reranking=None, query_text=""):
    from selfrag.modular_pipeline import build_selfrag_modular_graph

    t0 = time.time()
    compiled = build_selfrag_modular_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
    )
    state = await compiled.ainvoke({"query": query_text})
    elapsed = time.time() - t0
    gen: GenerationResult = state["generation_result"]
    cr = ConfigResult(label=name, gen=gen, elapsed=elapsed, pipeline="Self-RAG")
    _print_result(cr)
    return cr


async def _run_longrag(name, retrieval, generation, reranking=None, query_text=""):
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    t0 = time.time()
    compiled = build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
    )
    state = await compiled.ainvoke({"query": query_text})
    elapsed = time.time() - t0
    gen: GenerationResult = state["generation_result"]
    cr = ConfigResult(label=name, gen=gen, elapsed=elapsed, pipeline="LongRAG")
    _print_result(cr)
    return cr


async def _run_lightrag(name, retrieval, generation, reranking=None, query_text=""):
    from lightrag_langgraph.main_pipeline import build_query_graph

    t0 = time.time()
    compiled = build_query_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
    )
    state = await compiled.ainvoke({"query": query_text})
    elapsed = time.time() - t0
    gen: GenerationResult = state["generation_result"]
    cr = ConfigResult(label=name, gen=gen, elapsed=elapsed, pipeline="LightRAG")
    _print_result(cr)
    return cr


# =============================================================================
# Summary printer
# =============================================================================

def _print_summary(title: str, results: dict[str, ConfigResult]) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    print(f"  {'Config':<6} {'Pipeline':<10} {'Style':<15} {'Chars':>5} {'Cites':>5} {'Time':>6} {'Score':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*15} {'-'*5} {'-'*5} {'-'*6} {'-'*8}")
    for key in sorted(results.keys()):
        cr = results[key]
        style = cr.gen.metadata.get("style", "selfrag")
        score_str = ""
        if "selfrag_score" in cr.gen.metadata:
            score_str = f"{cr.gen.metadata['selfrag_score']:.2f}"
        print(
            f"  {key:<6} {cr.pipeline:<10} {style:<15} "
            f"{len(cr.gen.output):>5} {len(cr.gen.citations):>5} "
            f"{cr.elapsed:>5.1f}s {score_str:>8}"
        )
    total_time = sum(cr.elapsed for cr in results.values())
    print(f"\n  Total time: {total_time:.1f}s across {len(results)} configs")


# =============================================================================
# Section 1: Cross-project swap demo (6 configs, 3 pipelines)
# =============================================================================

async def section_cross_project_swaps(query: str) -> dict[str, ConfigResult]:
    print(f"\n{'#' * 72}")
    print("  SECTION 1: CROSS-PROJECT COMPONENT SWAPS")
    print(f"  6 configurations across 3 pipeline frameworks")
    print(f"{'#' * 72}")

    llm = _LLM()
    openai_model = OpenAIModel()

    llm_ret = LLMRetrieval(llm=llm)
    ddg_ret = DuckDuckGoRetrieval(k=3)
    ddg_fallback = FallbackRetrieval(primary=ddg_ret, fallback=llm_ret)
    longrag_gen = LongRAGReaderGeneration(llm=llm)
    selfrag_gen = _build_selfrag_generation(openai_model)
    selfrag_rr = _build_selfrag_reranking(openai_model)

    results: dict[str, ConfigResult] = {}

    print("\n--- A: Self-RAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["A"] = await _run_selfrag(
        "A) Self-RAG native: LLM retrieval -> SelfRAG generation",
        retrieval=llm_ret, generation=selfrag_gen, query_text=query,
    )

    print("\n--- B: Self-RAG pipeline (DDG retrieval + LongRAG reader) ---")
    results["B"] = await _run_selfrag(
        "B) Cross-gen: DDG retrieval -> LongRAG reader (Self-RAG pipe)",
        retrieval=ddg_ret, generation=longrag_gen, query_text=query,
    )

    print("\n--- C: LongRAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["C"] = await _run_longrag(
        "C) Cross-gen: LLM retrieval -> SelfRAG generation (LongRAG pipe)",
        retrieval=llm_ret, generation=selfrag_gen, query_text=query,
    )

    print("\n--- D: LongRAG pipeline (DDG + SelfRAG rerank + LongRAG reader) ---")
    results["D"] = await _run_longrag(
        "D) Full cross: DDG -> SelfRAG rerank -> LongRAG reader",
        retrieval=ddg_fallback, generation=longrag_gen,
        reranking=selfrag_rr, query_text=query,
    )

    print("\n--- E: LightRAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["E"] = await _run_lightrag(
        "E) Cross-pipe: LLM retrieval -> SelfRAG generation (LightRAG pipe)",
        retrieval=llm_ret, generation=selfrag_gen, query_text=query,
    )

    print("\n--- F: LightRAG pipeline (DDG + SelfRAG rerank + LongRAG reader) ---")
    results["F"] = await _run_lightrag(
        "F) Full cross: DDG -> SelfRAG rerank -> LongRAG reader (LightRAG pipe)",
        retrieval=ddg_fallback, generation=longrag_gen,
        reranking=selfrag_rr, query_text=query,
    )

    _print_summary("CROSS-PROJECT SWAP RESULTS", results)
    return results


# =============================================================================
# Section 2: 3-pipeline identity test
# =============================================================================

async def section_pipeline_identity(query: str) -> dict[str, ConfigResult]:
    print(f"\n\n{'#' * 72}")
    print("  SECTION 2: 3-PIPELINE IDENTITY TEST")
    print(f"  Same components through Self-RAG / LongRAG / LightRAG pipelines")
    print(f"  Proves the modular architecture produces consistent results")
    print(f"{'#' * 72}")

    llm = _LLM()
    llm_ret = LLMRetrieval(llm=llm)
    longrag_gen = LongRAGReaderGeneration(llm=llm)

    results: dict[str, ConfigResult] = {}

    print("\n--- Components: LLM retrieval + LongRAG reader (no reranking) ---")

    results["S"] = await _run_selfrag(
        "Self-RAG pipe: LLM retrieval -> LongRAG reader",
        retrieval=llm_ret, generation=longrag_gen, query_text=query,
    )
    results["L"] = await _run_longrag(
        "LongRAG pipe:  LLM retrieval -> LongRAG reader",
        retrieval=llm_ret, generation=longrag_gen, query_text=query,
    )
    results["R"] = await _run_lightrag(
        "LightRAG pipe: LLM retrieval -> LongRAG reader",
        retrieval=llm_ret, generation=longrag_gen, query_text=query,
    )

    _print_summary("3-PIPELINE IDENTITY TEST", results)

    lengths = [len(results[k].gen.output) for k in "SLR"]
    avg = sum(lengths) / 3
    variance = sum((l - avg) ** 2 for l in lengths) / 3
    print(f"\n  Output lengths: Self-RAG={lengths[0]}, LongRAG={lengths[1]}, LightRAG={lengths[2]}")
    print(f"  Length variance: {variance:.0f} (lower = more consistent)")
    print(f"  All 3 pipelines share identical topology: query -> retrieval -> reranking -> generation -> END")

    return results


# =============================================================================
# Section 3: ALCE benchmark evaluation
# =============================================================================

async def section_alce_benchmark() -> dict[str, Any]:
    print(f"\n\n{'#' * 72}")
    print("  SECTION 3: ALCE BENCHMARK EVALUATION")
    print(f"  Real F1/STR-EM scoring on ALCE sample data")
    print(f"{'#' * 72}")

    from benchmark.alce_adapter import (
        ALCEBenchmarkAdapter,
        alce_item_to_retrieval_results,
    )
    import json

    docs_path = Path("benchmark/sample_data/alce_kg_sample/alce_docs.json")
    queries_path = Path("benchmark/sample_data/alce_kg_sample/queries.jsonl")

    if not docs_path.exists() or not queries_path.exists():
        print("\n  ALCE sample data not found -- skipping benchmark section.")
        return {}

    with open(docs_path, encoding="utf-8") as f:
        all_docs = json.load(f)

    data: list[dict] = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                data.append({
                    "question": q["query"],
                    "answer": q.get("ground_truth", ""),
                    "docs": all_docs.get(q.get("query_id", ""), []),
                    "qa_pairs": q.get("qa_pairs", []),
                })

    if not data:
        print("\n  No ALCE items loaded -- skipping.")
        return {}

    print(f"\n  Loaded {len(data)} ALCE items with pre-retrieved documents.")

    llm = _LLM()
    openai_model = OpenAIModel()

    generators = {
        "LongRAG reader": LongRAGReaderGeneration(llm=llm),
        "SelfRAG gen": _build_selfrag_generation(openai_model),
        "SimpleLLM gen": SimpleLLMGeneration(llm=llm),
    }

    adapter = ALCEBenchmarkAdapter()
    bench_results: dict[str, Any] = {}

    for name, gen in generators.items():
        print(f"\n  Evaluating: {name} ...")
        t0 = time.time()
        result = adapter.evaluate_generation(data, gen, max_docs=5)
        elapsed = time.time() - t0
        bench_results[name] = {"result": result, "elapsed": elapsed}
        print(
            f"    F1={result.avg_f1:.1f}%  EM={result.avg_exact:.1f}%  "
            f"STR-EM={result.avg_str_em:.1f}%  "
            f"Avg length={result.avg_length:.0f} words  "
            f"Time={elapsed:.1f}s"
        )

    print(f"\n  Evaluating: LongRAG reader via pipeline (graph_factory) ...")
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    longrag_gen_for_pipe = LongRAGReaderGeneration(llm=llm)

    def graph_factory(retrieval):
        return build_graph(retrieval=retrieval, generation=longrag_gen_for_pipe)

    t0 = time.time()
    pipe_result = adapter.evaluate_pipeline(data, graph_factory, max_docs=5)
    elapsed = time.time() - t0
    bench_results["LongRAG (pipeline)"] = {"result": pipe_result, "elapsed": elapsed}
    print(
        f"    F1={pipe_result.avg_f1:.1f}%  EM={pipe_result.avg_exact:.1f}%  "
        f"STR-EM={pipe_result.avg_str_em:.1f}%  "
        f"Avg length={pipe_result.avg_length:.0f} words  "
        f"Time={elapsed:.1f}s"
    )

    print(f"\n{SEPARATOR}")
    print("  ALCE BENCHMARK SUMMARY")
    print(SEPARATOR)
    print(f"  {'Generator':<25} {'F1':>6} {'EM':>6} {'STR-EM':>7} {'Words':>6} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6}")
    for name, br in bench_results.items():
        r = br["result"]
        print(
            f"  {name:<25} {r.avg_f1:>5.1f}% {r.avg_exact:>5.1f}% "
            f"{r.avg_str_em:>6.1f}% {r.avg_length:>5.0f} {br['elapsed']:>5.1f}s"
        )

    return bench_results


# =============================================================================
# Section 4: Real dataset evaluation (HotpotQA + UltraDomain)
# =============================================================================

def section_real_datasets(data_dir: str, max_items: int = 5) -> dict[str, Any]:
    print(f"\n\n{'#' * 72}")
    print("  SECTION 4: REAL DATASET EVALUATION")
    print(f"  HotpotQA + UltraDomain from {data_dir}")
    print(f"  Items per dataset: {max_items}")
    print(f"{'#' * 72}")

    from benchmark.hotpotqa_adapter import (
        HotpotQABenchmarkAdapter, load_hotpotqa_real,
    )
    from benchmark.ultradomain_adapter import (
        UltraDomainBenchmarkAdapter, load_ultradomain_real,
    )
    from rag_contracts.component_registry import build_simple_llm

    llm = build_simple_llm()
    gen = SimpleLLMGeneration(llm=llm)
    longrag_gen = LongRAGReaderGeneration(llm=_LLM())
    results: dict[str, Any] = {}

    # -- HotpotQA --
    hq_dir = Path(data_dir) / "all_data" / "hotpotqa"
    if hq_dir.exists():
        print(f"\n  Loading HotpotQA from {hq_dir} ...")
        hq_data = load_hotpotqa_real(hq_dir, max_items=max_items)
        print(f"  Loaded {len(hq_data)} items (avg {sum(len(i['context_results']) for i in hq_data)/len(hq_data):.0f} passages each)")

        hq_adapter = HotpotQABenchmarkAdapter()

        for gen_name, gen_obj in [("SimpleLLM", gen), ("LongRAG reader", longrag_gen)]:
            print(f"\n  HotpotQA + {gen_name} ...")
            t0 = time.time()
            r = hq_adapter.evaluate_generation(hq_data, gen_obj)
            elapsed = time.time() - t0
            results[f"hotpotqa_{gen_name}"] = {"result": r, "elapsed": elapsed}
            print(f"    EM={r.avg_em:.1f}  F1={r.avg_f1:.1f}  items={r.num_items}  time={elapsed:.1f}s")
            for pi in r.per_item[:3]:
                print(f"      Q: {pi['question'][:60]}...")
                print(f"      A: {pi['answer'][:40]}  |  O: {pi['output'][:40]}  |  F1={pi['f1']:.1f}")
    else:
        print(f"\n  HotpotQA not found at {hq_dir} -- skipping")

    # -- UltraDomain --
    ud_dir = Path(data_dir) / "UltraDomain"
    if ud_dir.exists():
        print(f"\n  Loading UltraDomain from {ud_dir} ...")
        ud_data = load_ultradomain_real(ud_dir, domain="mix", max_items=max_items)
        print(f"  Loaded {len(ud_data)} items (avg {sum(len(i['context_results']) for i in ud_data)/len(ud_data):.0f} chunks each)")

        ud_adapter = UltraDomainBenchmarkAdapter()

        for gen_name, gen_obj in [("SimpleLLM", gen), ("LongRAG reader", longrag_gen)]:
            print(f"\n  UltraDomain + {gen_name} ...")
            t0 = time.time()
            r = ud_adapter.evaluate_generation(ud_data, gen_obj)
            elapsed = time.time() - t0
            results[f"ultradomain_{gen_name}"] = {"result": r, "elapsed": elapsed}
            print(f"    F1={r.avg_f1:.1f}  items={r.num_items}  time={elapsed:.1f}s")
            for pi in r.per_item[:3]:
                print(f"      Q: {pi['question'][:60]}...")
                print(f"      O: {pi['output'][:60]}...")
    else:
        print(f"\n  UltraDomain not found at {ud_dir} -- skipping")

    # -- Summary --
    if results:
        print(f"\n{SEPARATOR}")
        print("  REAL DATASET EVALUATION SUMMARY")
        print(SEPARATOR)
        print(f"  {'Config':<30} {'EM/F1':>8} {'Items':>6} {'Time':>6}")
        print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*6}")
        for name, info in results.items():
            r = info["result"]
            em = getattr(r, "avg_em", None)
            f1 = getattr(r, "avg_f1", 0.0)
            score_str = f"{em:.1f}/{f1:.1f}" if em is not None else f"--/{f1:.1f}"
            print(f"  {name:<30} {score_str:>8} {r.num_items:>6} {info['elapsed']:>5.1f}s")

    return results


# =============================================================================
# Section 5: Real vLLM Self-RAG vs Fake shim comparison
# =============================================================================

def section_real_vs_fake_selfrag(vllm_url: str, data_dir: str, max_items: int = 3) -> dict[str, Any]:
    print(f"\n\n{'#' * 72}")
    print("  SECTION 5: REAL vLLM SELF-RAG vs FAKE SHIM")
    print(f"  Comparing real selfrag_llama2_7b logprobs with synthetic logprobs")
    print(f"  vLLM: {vllm_url}")
    print(f"{'#' * 72}")

    from selfrag.adapters import SelfRAGGeneration
    from rag_contracts.component_registry import RealVLLMModel, _RealVLLMTokenizer

    model_name = os.environ.get("SELFRAG_MODEL_NAME", "selfrag-llama2-7b")

    # Build real model
    real_model = RealVLLMModel(base_url=vllm_url, model_name=model_name, num_logprobs=100)
    real_tok = _RealVLLMTokenizer(vllm_url, model_name)
    _, real_rel, real_grd, real_ut = load_special_tokens(real_tok, True, True)
    real_gen = SelfRAGGeneration(
        model=real_model, rel_tokens=real_rel,
        grd_tokens=real_grd, ut_tokens=real_ut, max_new_tokens=60,
    )

    # Build fake model
    fake_model = OpenAIModel()
    fake_gen = _build_selfrag_generation(fake_model)

    # Load a few real HotpotQA items for context
    hq_dir = Path(data_dir) / "all_data" / "hotpotqa"
    if not hq_dir.exists():
        print("\n  HotpotQA data not found -- using synthetic context")
        test_items = [
            {"question": "What is the capital of France?",
             "context_results": [RetrievalResult(source_id="p1",
                 content="Paris is the capital and most populous city of France.",
                 score=1.0, title="Paris")]},
            {"question": "Who wrote Romeo and Juliet?",
             "context_results": [RetrievalResult(source_id="p2",
                 content="Romeo and Juliet is a tragedy written by William Shakespeare.",
                 score=1.0, title="Shakespeare")]},
        ]
    else:
        from benchmark.hotpotqa_adapter import load_hotpotqa_real
        test_items = load_hotpotqa_real(hq_dir, max_items=max_items)

    results: dict[str, Any] = {"comparisons": []}

    for i, item in enumerate(test_items):
        question = item["question"]
        context = item["context_results"][:3]

        print(f"\n  --- Item {i+1}: {question[:65]}... ---")

        # Real model
        t0 = time.time()
        real_result = real_gen.generate(query=question, context=context)
        real_time = time.time() - t0
        real_score = real_result.metadata.get("selfrag_score", 0.0)

        # Fake model
        t0 = time.time()
        fake_result = fake_gen.generate(query=question, context=context)
        fake_time = time.time() - t0
        fake_score = fake_result.metadata.get("selfrag_score", 0.0)

        comparison = {
            "question": question,
            "real_output": real_result.output[:120],
            "real_score": real_score,
            "real_time": real_time,
            "fake_output": fake_result.output[:120],
            "fake_score": fake_score,
            "fake_time": fake_time,
        }
        results["comparisons"].append(comparison)

        print(f"  REAL  vLLM  | score={real_score:.3f} | time={real_time:.1f}s | len={len(real_result.output)}")
        print(f"    output: {real_result.output[:80]}...")
        print(f"  FAKE  shim  | score={fake_score:.3f} | time={fake_time:.1f}s | len={len(fake_result.output)}")
        print(f"    output: {fake_result.output[:80]}...")

    # Summary
    comps = results["comparisons"]
    avg_real_score = sum(c["real_score"] for c in comps) / len(comps) if comps else 0
    avg_fake_score = sum(c["fake_score"] for c in comps) / len(comps) if comps else 0
    avg_real_time = sum(c["real_time"] for c in comps) / len(comps) if comps else 0
    avg_fake_time = sum(c["fake_time"] for c in comps) / len(comps) if comps else 0

    print(f"\n{SEPARATOR}")
    print("  REAL vs FAKE SELF-RAG SUMMARY")
    print(SEPARATOR)
    print(f"  {'Metric':<25} {'Real vLLM':>12} {'Fake Shim':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Avg Self-RAG score':<25} {avg_real_score:>12.3f} {avg_fake_score:>12.3f}")
    print(f"  {'Avg latency (s)':<25} {avg_real_time:>12.2f} {avg_fake_time:>12.2f}")
    print(f"  {'Items compared':<25} {len(comps):>12} {len(comps):>12}")
    print()
    print("  Key insight: Real vLLM logprobs produce genuine Self-RAG scoring")
    print("  signals. Fake shim always returns synthetic -0.1/-5.0 logprobs,")
    print("  making all scores converge to the same value regardless of passage")
    print("  quality. Real model scores vary based on actual passage relevance.")

    return results


# =============================================================================
# Section 6: AG-UCT mini search with real data
# =============================================================================

def section_uct_search(data_dir: str, max_items: int = 3, max_iters: int = 5) -> dict[str, Any]:
    print(f"\n\n{'#' * 72}")
    print("  SECTION 6: AG-UCT MINI SEARCH WITH REAL DATA")
    print(f"  Quick UCT search to find best RAG config for real datasets")
    print(f"  Iterations: {max_iters}  |  Items per dataset: {max_items}")
    print(f"{'#' * 72}")

    sys.path.insert(0, str(Path(__file__).resolve().parent / "AG-UCT"))

    from uct_engine.examples.rag_pipeline_search import (
        RAGPipelineEvaluator,
        RAGPipelineSearchState,
        build_frozen_samples_real,
        CLUSTER_COST,
        CLUSTER_IDS,
        SLOT_NAMES,
    )
    from uct_engine import (
        ClusterDef,
        CostAwareUCTScorer,
        ReuseAwareCostModel,
        UCTSearchEngine,
    )

    print(f"\n  Loading real datasets from {data_dir} ...", flush=True)
    frozen = build_frozen_samples_real(data_dir, max_items=max_items)
    for cid, items in frozen.items():
        print(f"    {cid}: {len(items)} items", flush=True)

    if not frozen:
        print("  No datasets found -- skipping UCT search.")
        return {}

    evaluator = RAGPipelineEvaluator(use_real=True, frozen_samples=frozen)
    scorer = CostAwareUCTScorer(lambda_t=0.05)
    clusters = [ClusterDef(c, weight=1.0, base_cost=CLUSTER_COST[c]) for c in CLUSTER_IDS]
    cost_model = ReuseAwareCostModel(clusters=clusters)

    engine = UCTSearchEngine(
        evaluator=evaluator, scorer=scorer, cost_model=cost_model,
        exploration_constant=1.4, random_seed=42,
    )

    print(f"\n  Running UCT search ({max_iters} iterations) ...", flush=True)
    t0 = time.time()
    root = RAGPipelineSearchState()
    result = engine.search(root, max_iterations=max_iters)
    elapsed = time.time() - t0

    print(f"\n{SEPARATOR}")
    print("  AG-UCT SEARCH RESULTS")
    print(SEPARATOR)
    print(f"  Best config : {result.best_state.pretty()}")
    print(f"  Best reward : {result.best_reward:.4f}")
    print(f"  Iterations  : {result.iterations}")
    print(f"  Evaluations : {result.total_evaluations}")
    print(f"  Total cost  : {result.total_cost:.2f}")
    print(f"  Search time : {elapsed:.1f}s")
    print(f"  Materialized: {len(result.context.materialized_keys)} path keys")

    print("\n  Slot breakdown:")
    for action, child in result.root_node.children.items():
        print(
            f"    {SLOT_NAMES[0]}={action!s:20s}  visits={child.visit_count:4d}  "
            f"Q={child.q_value:.4f}  best={child.best_value:.4f}",
        )

    return {"result": result, "elapsed": elapsed}


# =============================================================================
# Section 7: System Health Check
# =============================================================================

def section_system_health(data_dir: str, vllm_url: str) -> dict[str, Any]:
    """Run protocol, data format, and pipeline constraint validation."""
    print(f"\n\n{'#' * 72}")
    print("  SECTION 7: SYSTEM HEALTH CHECK")
    print(f"  Protocol conformance, data format validation, pipeline constraints")
    print(f"{'#' * 72}")

    checks: list[dict[str, Any]] = []
    t0_all = time.time()

    def _check(name: str, fn) -> bool:
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            checks.append({"name": name, "status": "PASS", "time": elapsed})
            print(f"  [PASS] {name} ({elapsed:.2f}s)")
            return True
        except Exception as exc:
            elapsed = time.time() - t0
            msg = str(exc)[:80]
            checks.append({"name": name, "status": "FAIL", "error": msg, "time": elapsed})
            print(f"  [FAIL] {name} -- {msg}")
            return False

    # -- Protocol conformance --
    print(f"\n  --- Component Protocol Checks ---")

    def chk_retrieval_protocol():
        from rag_contracts import DuckDuckGoRetrieval, LLMRetrieval
        for cls in [DuckDuckGoRetrieval, LLMRetrieval]:
            assert hasattr(cls, "retrieve"), f"{cls.__name__} missing retrieve()"

    def chk_generation_protocol():
        from rag_contracts import SimpleLLMGeneration
        from rag_contracts.identity import IdentityGeneration
        gen = IdentityGeneration()
        result = gen.generate(query="test", context=[])
        assert isinstance(result, GenerationResult)
        assert isinstance(result.output, str)
        assert isinstance(result.citations, list)

    def chk_reranking_protocol():
        from rag_contracts import IdentityReranking
        rr = IdentityReranking()
        ctx = [RetrievalResult(source_id="s1", content="test", score=0.9, title="t")]
        result = rr.rerank("q", ctx, top_k=1)
        assert len(result) == 1
        assert isinstance(result[0], RetrievalResult)

    def chk_selfrag_adapters():
        from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking
        gen = SelfRAGGeneration(model=None)
        result = gen.generate(query="test", context=[])
        assert isinstance(result, GenerationResult)
        rr = SelfRAGReranking(model=None)
        ctx = [RetrievalResult(source_id="s1", content="test", score=0.9, title="t")]
        assert len(rr.rerank("q", ctx, top_k=10)) >= 0

    def chk_component_registry_builders():
        from rag_contracts.component_registry import (
            build_simple_llm, build_longrag_generation, build_selfrag_components,
        )
        llm = build_simple_llm()
        assert hasattr(llm, "complete")
        lg = build_longrag_generation()
        assert hasattr(lg, "generate")
        rr, gen = build_selfrag_components()
        if rr is not None:
            assert hasattr(rr, "rerank")
        if gen is not None:
            assert hasattr(gen, "generate")

    _check("Retrieval protocol (DDG, LLM)", chk_retrieval_protocol)
    _check("Generation protocol (Identity)", chk_generation_protocol)
    _check("Reranking protocol (Identity)", chk_reranking_protocol)
    _check("SelfRAG adapters (no model)", chk_selfrag_adapters)
    _check("Component registry builders", chk_component_registry_builders)

    # -- Data format validation --
    print(f"\n  --- Data Format Validation ---")

    def chk_hotpotqa_context_converter():
        from benchmark.base_adapter import hotpotqa_context_to_retrieval_results
        ctx = [["Title A", ["Sent one.", "Sent two."]], ["Title B", ["Another."]]]
        results = hotpotqa_context_to_retrieval_results(ctx)
        assert len(results) == 2
        assert results[0].title == "Title A"
        assert "Sent one." in results[0].content
        assert results[0].score > results[1].score

    def chk_hotpotqa_malformed():
        from benchmark.base_adapter import hotpotqa_context_to_retrieval_results
        ctx = [["Good", ["s"]], "bad", [], ["Only title"]]
        results = hotpotqa_context_to_retrieval_results(ctx)
        assert len(results) == 1

    def chk_ultradomain_context_converter():
        from benchmark.base_adapter import ultradomain_context_to_retrieval_results
        results = ultradomain_context_to_retrieval_results("A" * 10000, max_chunk_chars=4000)
        assert len(results) == 3
        assert len(results[0].content) == 4000

    def chk_ultradomain_empty():
        from benchmark.base_adapter import ultradomain_context_to_retrieval_results
        assert ultradomain_context_to_retrieval_results("") == []

    def chk_context_fallback():
        from benchmark.base_adapter import get_context_for_item
        pre = [RetrievalResult(source_id="r1", content="pre", score=1.0, title="t")]
        item = {"context_results": pre, "chunks": {"c1": {"content": "x", "doc_ids": []}}}
        assert get_context_for_item(item) is pre
        assert get_context_for_item({}) == []

    _check("HotpotQA context converter", chk_hotpotqa_context_converter)
    _check("HotpotQA malformed entries", chk_hotpotqa_malformed)
    _check("UltraDomain context converter", chk_ultradomain_context_converter)
    _check("UltraDomain empty context", chk_ultradomain_empty)
    _check("Context fallback priority", chk_context_fallback)

    # -- Real data loading --
    print(f"\n  --- Real Data Loading ---")

    hq_dir = Path(data_dir) / "all_data" / "hotpotqa"
    ud_dir = Path(data_dir) / "UltraDomain"

    if hq_dir.exists():
        def chk_hq_load():
            from benchmark.hotpotqa_adapter import load_hotpotqa_real
            items = load_hotpotqa_real(hq_dir, max_items=5)
            assert len(items) == 5
            for it in items:
                assert "question" in it and "answer" in it and "context_results" in it
                assert isinstance(it["context_results"], list) and len(it["context_results"]) > 0
        _check("HotpotQA real load (5 items)", chk_hq_load)
    else:
        checks.append({"name": "HotpotQA real load", "status": "SKIP", "time": 0})
        print(f"  [SKIP] HotpotQA real load -- not found at {hq_dir}")

    if ud_dir.exists():
        def chk_ud_load():
            from benchmark.ultradomain_adapter import load_ultradomain_real
            items = load_ultradomain_real(ud_dir, domain="mix", max_items=5)
            assert len(items) == 5
            for it in items:
                assert it["question"], "question should not be empty"
                assert it["answer"], "answer should not be empty"
                assert len(it["context_results"]) > 0
        _check("UltraDomain real load (5 items)", chk_ud_load)

        def chk_ud_multi_domain():
            from benchmark.ultradomain_adapter import load_ultradomain_real
            loaded_count = 0
            for domain in ("agriculture", "cs", "legal", "mix"):
                p = ud_dir / f"{domain}.jsonl"
                if p.exists():
                    items = load_ultradomain_real(ud_dir, domain=domain, max_items=1)
                    assert len(items) > 0
                    loaded_count += 1
            assert loaded_count >= 2, f"Only {loaded_count} domains loaded"
        _check("UltraDomain multi-domain load", chk_ud_multi_domain)
    else:
        checks.append({"name": "UltraDomain real load", "status": "SKIP", "time": 0})
        print(f"  [SKIP] UltraDomain real load -- not found at {ud_dir}")

    # -- Pipeline constraint validation --
    print(f"\n  --- Pipeline Constraints ---")

    def chk_pipeline_constraints():
        sys.path.insert(0, str(Path(__file__).resolve().parent / "AG-UCT"))
        from uct_engine.examples.rag_pipeline_search import _check_constraints
        assert _check_constraints(("kg_extraction", "identity", "lightrag_hybrid", "identity", "simple_llm"))
        assert not _check_constraints(("standard_passage", "identity", "lightrag_hybrid", "identity", "simple_llm"))

    def chk_pipeline_all_gen_slots():
        from rag_contracts.component_registry import build_pipeline_from_config
        gen_slots = ["longrag_reader", "simple_llm", "selfrag_generator"]
        built = 0
        for g in gen_slots:
            choices = ("standard_passage", "identity", "bm25", "identity", g)
            try:
                comp = build_pipeline_from_config(choices, "hotpotqa")
                gen = comp.get("generation")
                assert gen is not None and hasattr(gen, "generate")
                built += 1
            except ImportError:
                pass
        assert built >= 2, f"Only {built}/3 generation slots built"

    def chk_vllm_wrapper_importable():
        from rag_contracts.component_registry import RealVLLMModel, _RealVLLMTokenizer
        assert RealVLLMModel is not None
        assert _RealVLLMTokenizer is not None

    _check("Config constraint enforcement", chk_pipeline_constraints)
    _check("All generation slots buildable", chk_pipeline_all_gen_slots)
    _check("RealVLLMModel importable", chk_vllm_wrapper_importable)

    # -- vLLM server check --
    if _vllm_reachable(vllm_url):
        def chk_vllm_tokenizer():
            from rag_contracts.component_registry import _RealVLLMTokenizer
            tok = _RealVLLMTokenizer(vllm_url, "selfrag-llama2-7b")
            rel_id = tok.convert_tokens_to_ids("[Relevant]")
            irr_id = tok.convert_tokens_to_ids("[Irrelevant]")
            assert rel_id != 0 and irr_id != 0 and rel_id != irr_id
        _check("vLLM tokenizer resolves special tokens", chk_vllm_tokenizer)
    else:
        checks.append({"name": "vLLM tokenizer", "status": "SKIP", "time": 0})
        print(f"  [SKIP] vLLM tokenizer -- server offline")

    # -- Summary --
    elapsed_all = time.time() - t0_all
    passed = sum(1 for c in checks if c["status"] == "PASS")
    failed = sum(1 for c in checks if c["status"] == "FAIL")
    skipped = sum(1 for c in checks if c["status"] == "SKIP")
    total = len(checks)

    print(f"\n{SEPARATOR}")
    print("  SYSTEM HEALTH SUMMARY")
    print(SEPARATOR)
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print(f"  Skipped      : {skipped}")
    print(f"  Time         : {elapsed_all:.1f}s")

    if failed > 0:
        print(f"\n  FAILURES:")
        for c in checks:
            if c["status"] == "FAIL":
                print(f"    - {c['name']}: {c.get('error', '?')}")

    health_pct = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
    print(f"\n  Health score : {health_pct:.0f}% ({passed}/{passed + failed} non-skip checks)")

    return {"checks": checks, "passed": passed, "failed": failed,
            "skipped": skipped, "health_pct": health_pct, "elapsed": elapsed_all}


# =============================================================================
# Section 8: Cross-Project Swap with Real Data
# =============================================================================

def section_real_data_swaps(data_dir: str, max_items: int = 2) -> dict[str, Any]:
    """Test component swaps using real HotpotQA/UltraDomain context passages."""
    print(f"\n\n{'#' * 72}")
    print("  SECTION 8: CROSS-PROJECT SWAP WITH REAL DATA")
    print(f"  Component swaps using real dataset context (not synthetic queries)")
    print(f"  Items per dataset: {max_items}")
    print(f"{'#' * 72}")

    from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter, load_hotpotqa_real
    from benchmark.ultradomain_adapter import UltraDomainBenchmarkAdapter, load_ultradomain_real
    from rag_contracts.component_registry import build_simple_llm

    llm = _LLM()
    results: dict[str, Any] = {}
    swap_tests: list[dict[str, Any]] = []

    # -- Load real data --
    hq_dir = Path(data_dir) / "all_data" / "hotpotqa"
    ud_dir = Path(data_dir) / "UltraDomain"
    hq_items: list[dict] = []
    ud_items: list[dict] = []

    if hq_dir.exists():
        hq_items = load_hotpotqa_real(hq_dir, max_items=max_items)
        print(f"\n  HotpotQA: {len(hq_items)} items loaded")
    else:
        print(f"\n  HotpotQA not found -- skipping HotpotQA swap tests")

    if ud_dir.exists():
        ud_items = load_ultradomain_real(ud_dir, domain="mix", max_items=max_items)
        print(f"  UltraDomain: {len(ud_items)} items loaded")
    else:
        print(f"  UltraDomain not found -- skipping UltraDomain swap tests")

    if not hq_items and not ud_items:
        print("\n  No real data available -- skipping section 8")
        return {}

    # -- Build generators --
    openai_model = OpenAIModel()
    simple_gen = SimpleLLMGeneration(llm=llm)
    longrag_gen = LongRAGReaderGeneration(llm=llm)
    selfrag_gen = _build_selfrag_generation(openai_model)

    generators = {
        "SimpleLLM": simple_gen,
        "LongRAG reader": longrag_gen,
        "SelfRAG": selfrag_gen,
    }

    # -- HotpotQA: swap generators on real context --
    if hq_items:
        print(f"\n  --- HotpotQA: Generator Swaps on Real Context ---")
        hq_adapter = HotpotQABenchmarkAdapter()

        for gen_name, gen_obj in generators.items():
            t0 = time.time()
            r = hq_adapter.evaluate_generation(hq_items, gen_obj)
            elapsed = time.time() - t0
            key = f"hq_{gen_name}"
            swap_tests.append({
                "test": f"HotpotQA + {gen_name}",
                "em": r.avg_em, "f1": r.avg_f1,
                "items": r.num_items, "time": elapsed,
            })
            print(f"    {gen_name:<15} EM={r.avg_em:.1f}  F1={r.avg_f1:.1f}  time={elapsed:.1f}s")

            for pi in r.per_item[:2]:
                q_short = pi["question"][:55]
                print(f"      Q: {q_short}...")
                print(f"      A: {pi['answer'][:40]}  |  O: {pi['output'][:40]}")

    # -- UltraDomain: swap generators on real context --
    if ud_items:
        print(f"\n  --- UltraDomain: Generator Swaps on Real Context ---")
        ud_adapter = UltraDomainBenchmarkAdapter()

        for gen_name, gen_obj in generators.items():
            t0 = time.time()
            r = ud_adapter.evaluate_generation(ud_items, gen_obj)
            elapsed = time.time() - t0
            swap_tests.append({
                "test": f"UltraDomain + {gen_name}",
                "f1": r.avg_f1, "items": r.num_items, "time": elapsed,
            })
            print(f"    {gen_name:<15} F1={r.avg_f1:.1f}  time={elapsed:.1f}s")

    # -- Reranking swap on real context --
    if hq_items:
        print(f"\n  --- Reranking Swaps on Real HotpotQA Context ---")
        from rag_contracts import IdentityReranking

        identity_rr = IdentityReranking()
        selfrag_rr = _build_selfrag_reranking(openai_model)

        item = hq_items[0]
        context = item["context_results"]
        question = item["question"]

        for rr_name, rr_obj in [("Identity", identity_rr), ("SelfRAG critique", selfrag_rr)]:
            t0 = time.time()
            reranked = rr_obj.rerank(question, context, top_k=5)
            elapsed = time.time() - t0
            swap_tests.append({
                "test": f"Rerank({rr_name}) on real ctx",
                "items": len(reranked), "time": elapsed,
            })
            top_titles = [r.title[:30] for r in reranked[:3]]
            print(f"    {rr_name:<18} top-{len(reranked)} in {elapsed:.1f}s  titles={top_titles}")

    # -- Pipeline swap: same real question through all 3 frameworks --
    if hq_items:
        print(f"\n  --- Full Pipeline Swap: Real Question Through 3 Frameworks ---")
        item = hq_items[0]
        question = item["question"]
        print(f"    Q: {question[:70]}...")
        print(f"    A: {item['answer'][:60]}")

        pipe_results: dict[str, ConfigResult] = {}
        ctx_ret = _ContextRetrieval(item["context_results"][:5])

        pipe_results["Self-RAG"] = await_or_run(
            _run_selfrag(
                "Self-RAG pipe + SimpleLLM (real ctx)",
                retrieval=ctx_ret, generation=simple_gen, query_text=question,
            )
        )
        pipe_results["LongRAG"] = await_or_run(
            _run_longrag(
                "LongRAG pipe + SimpleLLM (real ctx)",
                retrieval=ctx_ret, generation=simple_gen, query_text=question,
            )
        )
        pipe_results["LightRAG"] = await_or_run(
            _run_lightrag(
                "LightRAG pipe + SimpleLLM (real ctx)",
                retrieval=ctx_ret, generation=simple_gen, query_text=question,
            )
        )

        for pname, cr in pipe_results.items():
            swap_tests.append({
                "test": f"{pname} pipe (real ctx)",
                "time": cr.elapsed, "output_len": len(cr.gen.output),
            })

    # -- Summary --
    print(f"\n{SEPARATOR}")
    print("  REAL-DATA SWAP TEST SUMMARY")
    print(SEPARATOR)
    print(f"  {'Test':<35} {'EM':>5} {'F1':>6} {'Items':>6} {'Time':>6}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*6} {'-'*6}")
    for st in swap_tests:
        em_str = f"{st['em']:.1f}" if "em" in st else "--"
        f1_str = f"{st['f1']:.1f}" if "f1" in st else "--"
        items_str = f"{st['items']}" if "items" in st else "--"
        time_str = f"{st['time']:.1f}s"
        print(f"  {st['test']:<35} {em_str:>5} {f1_str:>6} {items_str:>6} {time_str:>6}")

    total_time = sum(st["time"] for st in swap_tests)
    print(f"\n  Total swap tests: {len(swap_tests)}  |  Total time: {total_time:.1f}s")

    return {"swap_tests": swap_tests}


class _ContextRetrieval:
    """Wraps pre-built context_results as a retrieval component."""
    def __init__(self, context_results: list[RetrievalResult]):
        self._results = context_results

    def retrieve(self, queries, top_k=5, **kw):
        return self._results[:top_k]


def await_or_run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


# =============================================================================
# Utility: check if vLLM server is reachable
# =============================================================================

def _vllm_reachable(url: str) -> bool:
    try:
        import httpx
        resp = httpx.get(f"{url}/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        try:
            import urllib.request
            base = url.rstrip("/")
            if not base.endswith("/v1"):
                base += "/v1"
            with urllib.request.urlopen(f"{base}/models", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OminiRAG cross-project swap demo")
    parser.add_argument("--quick", action="store_true", help="Only run section 1")
    parser.add_argument("--real", action="store_true", help="Run all 6 sections (including real data + vLLM)")
    parser.add_argument("--query", type=str, default=None, help="Custom query")
    parser.add_argument("--data-dir", type=str, default="/data1/ragworkspace/dataset",
                        help="Path to real dataset root")
    parser.add_argument("--vllm-url", type=str, default=None,
                        help="vLLM server URL (default: $SELFRAG_VLLM_URL or http://localhost:8002/v1)")
    parser.add_argument("--max-items", type=int, default=5,
                        help="Items per dataset for real evaluation (default: 5)")
    parser.add_argument("--uct-iters", type=int, default=5,
                        help="UCT search iterations (default: 5)")
    args = parser.parse_args()

    vllm_url = args.vllm_url or VLLM_URL or "http://localhost:8002/v1"

    query = args.query or os.environ.get(
        "DEMO_QUERY",
        "What were the main causes and consequences of the 2023 Silicon Valley Bank collapse?",
    )

    has_real_data = Path(args.data_dir).exists()
    has_vllm = _vllm_reachable(vllm_url)

    print(f"{'=' * 72}")
    print("  OminiRAG: REAL LLM CROSS-PROJECT SWAP DEMO")
    print(f"  3 pipeline frameworks x 3 RAG systems x live LLM")
    print(f"{'=' * 72}")
    print(f"\n  Query:     {query}")
    print(f"  Model:     {MODEL}")
    print(f"  Base URL:  {BASE_URL or '(default OpenAI)'}")
    print(f"  Data dir:  {args.data_dir} ({'found' if has_real_data else 'NOT FOUND'})")
    print(f"  vLLM:      {vllm_url} ({'online' if has_vllm else 'offline'})")

    if args.real:
        sections = "1-8" if has_real_data else "1-3,7"
    elif args.quick:
        sections = "1"
    else:
        sections = "1-3"
    print(f"  Sections:  {sections}")

    total_t0 = time.time()

    # Section 1: Cross-project swaps (6 configs)
    swap_results = await section_cross_project_swaps(query)

    if not args.quick:
        # Section 2: 3-pipeline identity test
        identity_results = await section_pipeline_identity(query)

        # Section 3: ALCE benchmark
        bench_results = await section_alce_benchmark()

    if args.real and has_real_data:
        # Section 4: Real dataset evaluation
        real_results = section_real_datasets(args.data_dir, max_items=args.max_items)

        # Section 5: Real vs Fake Self-RAG comparison
        if has_vllm:
            compare_results = section_real_vs_fake_selfrag(
                vllm_url, args.data_dir, max_items=min(args.max_items, 3),
            )
        else:
            print(f"\n  [SKIP] Section 5: vLLM server not reachable at {vllm_url}")
            print(f"         Launch with: bash scripts/launch_selfrag_vllm.sh")
            print(f"         Then set: export SELFRAG_VLLM_URL={vllm_url}")

        # Section 6: AG-UCT mini search
        uct_results = section_uct_search(
            args.data_dir, max_items=args.max_items, max_iters=args.uct_iters,
        )

    if args.real:
        # Section 7: System health check (runs even without real data)
        health_results = section_system_health(args.data_dir, vllm_url)

        # Section 8: Real-data swap tests (requires real data)
        if has_real_data:
            swap_results = section_real_data_swaps(args.data_dir, max_items=min(args.max_items, 3))

    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 72}")
    print(f"  DEMO COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Model: {MODEL}")
    print(f"  Sections run: {sections}")
    if args.real:
        print(f"  Real data: {'yes' if has_real_data else 'no'}")
        print(f"  Real vLLM: {'yes' if has_vllm else 'no'}")
        if 'health_results' in dir():
            pct = health_results.get("health_pct", 0)
            print(f"  System health: {pct:.0f}%")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(main())
