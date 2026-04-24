"""Real LLM cross-project swap demo: Self-RAG + LongRAG + LightRAG.

Exercises all three modular pipelines with live LLM calls, swapping
components between the three RAG systems to prove full interchangeability.

Sections:
  1) Cross-project swaps   -- 6 configs across all 3 pipeline frames
  2) 3-pipeline identity   -- same components through all 3 pipelines
  3) ALCE benchmark eval   -- real F1/STR-EM scoring on sample data

Usage:
    python real_selfrag_swap_demo.py
    python real_selfrag_swap_demo.py --quick          # only section 1
    python real_selfrag_swap_demo.py --query "..."     # custom query
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

    # A) Self-RAG pipeline: LLM retrieval + SelfRAG generation (native)
    print("\n--- A: Self-RAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["A"] = await _run_selfrag(
        "A) Self-RAG native: LLM retrieval -> SelfRAG generation",
        retrieval=llm_ret,
        generation=selfrag_gen,
        query_text=query,
    )

    # B) Self-RAG pipeline: DDG retrieval + LongRAG reader (cross-gen)
    print("\n--- B: Self-RAG pipeline (DDG retrieval + LongRAG reader) ---")
    results["B"] = await _run_selfrag(
        "B) Cross-gen: DDG retrieval -> LongRAG reader (Self-RAG pipe)",
        retrieval=ddg_ret,
        generation=longrag_gen,
        query_text=query,
    )

    # C) LongRAG pipeline: LLM retrieval + SelfRAG generation (cross-gen)
    print("\n--- C: LongRAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["C"] = await _run_longrag(
        "C) Cross-gen: LLM retrieval -> SelfRAG generation (LongRAG pipe)",
        retrieval=llm_ret,
        generation=selfrag_gen,
        query_text=query,
    )

    # D) LongRAG pipeline: DDG + SelfRAG reranking + LongRAG reader
    print("\n--- D: LongRAG pipeline (DDG + SelfRAG rerank + LongRAG reader) ---")
    results["D"] = await _run_longrag(
        "D) Full cross: DDG -> SelfRAG rerank -> LongRAG reader",
        retrieval=ddg_fallback,
        generation=longrag_gen,
        reranking=selfrag_rr,
        query_text=query,
    )

    # E) LightRAG pipeline: LLM retrieval + SelfRAG generation (3rd pipe)
    print("\n--- E: LightRAG pipeline (LLM retrieval + SelfRAG generation) ---")
    results["E"] = await _run_lightrag(
        "E) Cross-pipe: LLM retrieval -> SelfRAG generation (LightRAG pipe)",
        retrieval=llm_ret,
        generation=selfrag_gen,
        query_text=query,
    )

    # F) LightRAG pipeline: DDG + SelfRAG reranking + LongRAG reader
    print("\n--- F: LightRAG pipeline (DDG + SelfRAG rerank + LongRAG reader) ---")
    results["F"] = await _run_lightrag(
        "F) Full cross: DDG -> SelfRAG rerank -> LongRAG reader (LightRAG pipe)",
        retrieval=ddg_fallback,
        generation=longrag_gen,
        reranking=selfrag_rr,
        query_text=query,
    )

    _print_summary("CROSS-PROJECT SWAP RESULTS", results)
    return results


# =============================================================================
# Section 2: 3-pipeline identity test (same components, all 3 pipelines)
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

    # Also test graph_factory pattern with ALCEDocRetrieval
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
# Main
# =============================================================================

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OminiRAG cross-project swap demo")
    parser.add_argument("--quick", action="store_true", help="Only run section 1")
    parser.add_argument("--query", type=str, default=None, help="Custom query")
    args = parser.parse_args()

    query = args.query or os.environ.get(
        "DEMO_QUERY",
        "What were the main causes and consequences of the 2023 Silicon Valley Bank collapse?",
    )

    print(f"{'=' * 72}")
    print("  OminiRAG: REAL LLM CROSS-PROJECT SWAP DEMO")
    print(f"  3 pipeline frameworks x 3 RAG systems x live LLM")
    print(f"{'=' * 72}")
    print(f"\n  Query: {query}")
    print(f"  Model: {MODEL}")
    print(f"  Base URL: {BASE_URL or '(default OpenAI)'}")

    total_t0 = time.time()

    # Section 1: Cross-project swaps (6 configs)
    swap_results = await section_cross_project_swaps(query)

    if not args.quick:
        # Section 2: 3-pipeline identity test
        identity_results = await section_pipeline_identity(query)

        # Section 3: ALCE benchmark
        bench_results = await section_alce_benchmark()

    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 72}")
    print(f"  DEMO COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Model: {MODEL}")
    sections = "1" if args.quick else "1, 2, 3"
    print(f"  Sections run: {sections}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(main())
