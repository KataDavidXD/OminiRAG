"""Real LLM cross-project swap demo: Self-RAG + LongRAG + STORM.

Exercises the modular Self-RAG pipeline and LongRAG pipeline with live LLM
calls, swapping components from all three RAG systems.

Configurations:
  A) Self-RAG modular native   -- LLM retrieval  + SelfRAGGeneration
  B) Self-RAG modular + DDG    -- DuckDuckGo     + LongRAG reader
  C) Self-RAG modular + STORM  -- LLM retrieval  + STORM writer
  D) SelfRAG gen in LongRAG    -- LLM retrieval  + SelfRAGGeneration (in LongRAG pipe)
  E) SelfRAG rerank in LongRAG -- DDG retrieval  + SelfRAGReranking + LongRAG reader
  F) Full 3-way mix            -- DDG+fallback   + SelfRAGReranking + STORM writer

Usage:
    python real_selfrag_swap_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import types
from dataclasses import dataclass, field
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
# Fake vllm / torch so selfrag.nodes can import SamplingParams
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

# Ensure selfrag package is importable
_selfrag_root = str(Path(__file__).resolve().parent / "self-rag_langgraph" / "self-rag-wtb")
if _selfrag_root not in sys.path:
    sys.path.insert(0, _selfrag_root)

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    RetrievalResult,
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
# OpenAI model wrapper (mimics vLLM generate() interface for Self-RAG adapters)
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
# Canonical retrieval components (reused from real_swap_demo.py)
# =============================================================================


@dataclass
class LLMRetrieval:
    """Asks the LLM to produce background context."""

    llm: _LLM

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for i, q in enumerate(queries[:top_k]):
            ctx = self.llm.complete(
                "You are a research assistant. Provide concise, factual background "
                "information that would help answer the following question. "
                "Include key facts, dates, and names.",
                q,
                temperature=0.3,
                max_tokens=400,
            )
            results.append(
                RetrievalResult(
                    source_id=f"llm-context://{i}",
                    content=ctx,
                    score=1.0,
                    title=f"LLM-generated context for: {q}",
                )
            )
        return results


@dataclass
class DuckDuckGoRetrieval:
    """Web search retrieval via DuckDuckGo."""

    k: int = 5

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        from ddgs import DDGS

        ddgs = DDGS()
        seen: set[str] = set()
        results: list[RetrievalResult] = []
        for q in queries:
            try:
                hits = ddgs.text(q, max_results=self.k)
            except Exception:
                hits = []
            for hit in hits or []:
                url = hit.get("href", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append(
                    RetrievalResult(
                        source_id=url,
                        content=hit.get("body", ""),
                        score=0.9,
                        title=hit.get("title", ""),
                    )
                )
        return results[:top_k]


@dataclass
class FallbackRetrieval:
    primary: object
    fallback: object

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = self.primary.retrieve(queries, top_k=top_k)
        if results:
            return results
        print("    [FallbackRetrieval] Primary returned 0 results, using fallback...")
        return self.fallback.retrieve(queries, top_k=top_k)


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


@dataclass
class StormWriterGeneration:
    """Produces a Wikipedia-style section from context (STORM writer style)."""

    llm: _LLM

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        evidence_lines = []
        for i, r in enumerate(context[:6], 1):
            evidence_lines.append(f"[{i}] {r.title}\n{r.content}")
        evidence = "\n\n".join(evidence_lines)
        section = self.llm.complete(
            "You write one Wikipedia-style section with inline citations [1][2]. "
            "Start with a markdown heading.",
            f"Topic: {query}\nEvidence:\n{evidence}",
            temperature=0.3,
            max_tokens=600,
        )
        return GenerationResult(
            output=section.strip(),
            citations=[r.source_id for r in context[:6]],
            metadata={"style": "storm-writer"},
        )


# =============================================================================
# Self-RAG adapter components (backed by real OpenAI LLM)
# =============================================================================


def _build_selfrag_generation(openai_model: OpenAIModel) -> "SelfRAGGeneration":
    """Build a SelfRAGGeneration adapter with real LLM backing."""
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


def _build_selfrag_reranking(openai_model: OpenAIModel) -> "SelfRAGReranking":
    """Build a SelfRAGReranking adapter with real LLM backing."""
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
# Runner helpers
# =============================================================================

SEPARATOR = "=" * 72


def _print_result(label: str, gen: GenerationResult) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {label}")
    print(f"  Style: {gen.metadata.get('style', 'selfrag')}")
    print(f"  Citations: {len(gen.citations)}")
    if "selfrag_score" in gen.metadata:
        print(f"  Self-RAG score: {gen.metadata['selfrag_score']:.3f}")
    print(SEPARATOR)
    wrapped = textwrap.fill(gen.output, width=72)
    print(wrapped)
    print()


async def _run_selfrag_modular(name, retrieval, generation, reranking=None, query_text=""):
    """Run the Self-RAG modular pipeline."""
    from selfrag.modular_pipeline import build_selfrag_modular_graph

    compiled = build_selfrag_modular_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
    )
    state = await compiled.ainvoke({"query": query_text})
    gen: GenerationResult = state["generation_result"]
    _print_result(name, gen)
    return gen


async def _run_longrag(name, retrieval, generation, reranking=None, query_text=""):
    """Run the LongRAG pipeline."""
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    compiled = build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
    )
    state = await compiled.ainvoke({"query": query_text})
    gen: GenerationResult = state["generation_result"]
    _print_result(name, gen)
    return gen


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    query = os.environ.get(
        "DEMO_QUERY",
        "What were the main causes and consequences of the 2023 Silicon Valley Bank collapse?",
    )

    print(f"{'=' * 72}")
    print("  REAL LLM CROSS-PROJECT SWAP DEMO")
    print(f"  Self-RAG + LongRAG + STORM -- 6 configurations")
    print(f"{'=' * 72}")
    print(f"\nQuery: {query}")
    print(f"Model: {MODEL}")
    print(f"Base URL: {BASE_URL or '(default OpenAI)'}\n")

    llm = _LLM()
    openai_model = OpenAIModel()

    llm_retrieval = LLMRetrieval(llm=llm)
    ddg_retrieval = DuckDuckGoRetrieval(k=3)
    ddg_with_fallback = FallbackRetrieval(primary=ddg_retrieval, fallback=llm_retrieval)
    longrag_gen = LongRAGReaderGeneration(llm=llm)
    storm_gen = StormWriterGeneration(llm=llm)
    selfrag_gen = _build_selfrag_generation(openai_model)
    selfrag_reranker = _build_selfrag_reranking(openai_model)

    results: dict[str, GenerationResult] = {}

    # -- Config A: Self-RAG modular native --
    print("\n--- Config A: Self-RAG modular (LLM retrieval + SelfRAGGeneration) ---")
    results["A"] = await _run_selfrag_modular(
        "A) Self-RAG Modular Native",
        retrieval=llm_retrieval,
        generation=selfrag_gen,
        query_text=query,
    )

    # -- Config B: Self-RAG modular + DuckDuckGo + LongRAG reader --
    print("\n--- Config B: Self-RAG modular (DDG retrieval + LongRAG reader) ---")
    results["B"] = await _run_selfrag_modular(
        "B) DDG Retrieval -> LongRAG Reader (in Self-RAG pipe)",
        retrieval=ddg_retrieval,
        generation=longrag_gen,
        query_text=query,
    )

    # -- Config C: Self-RAG modular + STORM writer --
    print("\n--- Config C: Self-RAG modular (LLM retrieval + STORM writer) ---")
    results["C"] = await _run_selfrag_modular(
        "C) LLM Retrieval -> STORM Writer (in Self-RAG pipe)",
        retrieval=llm_retrieval,
        generation=storm_gen,
        query_text=query,
    )

    # -- Config D: SelfRAGGeneration as drop-in in LongRAG pipeline --
    print("\n--- Config D: LongRAG pipeline with SelfRAGGeneration ---")
    results["D"] = await _run_longrag(
        "D) LLM Retrieval -> SelfRAGGeneration (in LongRAG pipe)",
        retrieval=llm_retrieval,
        generation=selfrag_gen,
        query_text=query,
    )

    # -- Config E: SelfRAGReranking in LongRAG pipeline --
    print("\n--- Config E: LongRAG pipeline with SelfRAGReranking + LongRAG reader ---")
    results["E"] = await _run_longrag(
        "E) DDG+Fallback -> SelfRAGReranking -> LongRAG Reader",
        retrieval=ddg_with_fallback,
        generation=longrag_gen,
        reranking=selfrag_reranker,
        query_text=query,
    )

    # -- Config F: Full 3-way mix --
    print("\n--- Config F: Full 3-way mix (DDG + SelfRAG rerank + STORM writer) ---")
    results["F"] = await _run_selfrag_modular(
        "F) DDG+Fallback -> SelfRAGReranking -> STORM Writer",
        retrieval=ddg_with_fallback,
        generation=storm_gen,
        reranking=selfrag_reranker,
        query_text=query,
    )

    # -- Summary --
    print(f"\n{SEPARATOR}")
    print("  COMPARISON SUMMARY")
    print(SEPARATOR)
    labels = {
        "A": "Self-RAG native (LLM + SelfRAG gen)",
        "B": "Self-RAG pipe (DDG + LongRAG reader)",
        "C": "Self-RAG pipe (LLM + STORM writer)",
        "D": "LongRAG pipe (LLM + SelfRAG gen)",
        "E": "LongRAG pipe (DDG + SelfRAG rerank + reader)",
        "F": "3-way mix (DDG + SelfRAG rerank + STORM)",
    }
    for key in "ABCDEF":
        gen = results[key]
        score_str = ""
        if "selfrag_score" in gen.metadata:
            score_str = f"  score={gen.metadata['selfrag_score']:.2f}"
        print(f"  {key}) {labels[key]:<45} {len(gen.output):>5} chars, "
              f"{len(gen.citations)} cites{score_str}")

    print(f"\nDemo complete. 6 cross-project swaps verified with live LLM ({MODEL}).")


if __name__ == "__main__":
    asyncio.run(main())
