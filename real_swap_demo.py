"""Real cross-project component swap demo.

Wires actual LLM-backed components from STORM and LongRAG, runs the LongRAG
pipeline with three configurations, and compares the outputs.

Configurations:
  A) LongRAG default  -- LLM-based retrieval + LongRAG reader
  B) STORM retrieval  -- DuckDuckGo web search + LongRAG reader
  C) STORM generation -- LLM-based retrieval + STORM section writer
  D) Full cross-proj  -- DDG with LLM fallback + STORM section writer

Usage:
    python real_swap_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag_contracts import GenerationResult, IdentityReranking, RetrievalResult
from rag_contracts import WTBCacheConfig, WTBCachedLLM, attach_wtb_cache_metadata

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_env_file = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_file)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared LLM client (reuses the same env vars as STORM's real_components)
# ═══════════════════════════════════════════════════════════════════════════════


class _LLM:
    def __init__(self):
        self._wtb_llm = None
        cache_config = WTBCacheConfig.from_env()
        if cache_config.cache_active:
            self._wtb_llm = WTBCachedLLM(
                config=cache_config,
                system_name="ominirag",
                node_path="real_swap_demo.llm",
            )
            self.model = self._wtb_llm.model
            self.client = None
            return

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: Set LLM_API_KEY or OPENAI_API_KEY in .env")
        base_url = (
            os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or None
        )
        self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, **kwargs) -> str:
        if self._wtb_llm is not None:
            return self._wtb_llm.complete(system, user, **kwargs)

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

    def wtb_cache_metadata(self):
        if self._wtb_llm is None:
            return None
        return self._wtb_llm.wtb_cache_metadata()


# ═══════════════════════════════════════════════════════════════════════════════
# Component A: LLM-based "Retrieval"  (canonical rag_contracts.Retrieval)
# Simulates LongRAG's approach: the LLM itself provides context.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LLMRetrieval:
    """Asks the LLM to produce background context for the query."""

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
                    metadata=attach_wtb_cache_metadata(
                        {"style": "llm-retrieval"},
                        self.llm,
                    ),
                )
            )
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# Component B: DuckDuckGo Web Search  (canonical rag_contracts.Retrieval)
# Uses STORM's DuckDuckGo retriever, adapted to canonical protocol.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DuckDuckGoRetrieval:
    """Web search retrieval via DuckDuckGo (no API key required).

    Uses the new ``ddgs`` package (``duckduckgo_search`` is deprecated and
    returns 0 results).
    """

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
                body = hit.get("body", "")
                results.append(
                    RetrievalResult(
                        source_id=url,
                        content=body,
                        score=0.9,
                        title=hit.get("title", ""),
                    )
                )
        return results[:top_k]


@dataclass
class FallbackRetrieval:
    """Tries the primary retrieval first; if it returns nothing, falls back.

    This solves the problem where DDG (or any web search) has no results for
    a niche query -- the pipeline still gets usable context from the fallback.
    """

    primary: object
    fallback: object

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = self.primary.retrieve(queries, top_k=top_k)
        if results:
            return results
        print("    [FallbackRetrieval] Primary returned 0 results, using fallback...")
        return self.fallback.retrieve(queries, top_k=top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# Component C: LongRAG-style Reader  (canonical rag_contracts.Generation)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LongRAGReaderGeneration:
    """Extracts a concise answer from context (LongRAG reader style)."""

    llm: _LLM

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
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
            metadata=attach_wtb_cache_metadata(
                {"style": "longrag-reader"},
                self.llm,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Component D: STORM-style Section Writer  (canonical rag_contracts.Generation)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StormWriterGeneration:
    """Produces a Wikipedia-style section from context (STORM writer style)."""

    llm: _LLM

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        evidence_lines: list[str] = []
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
            metadata=attach_wtb_cache_metadata(
                {"style": "storm-writer"},
                self.llm,
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════


def _print_result(label: str, gen: GenerationResult) -> None:
    border = "=" * 72
    print(f"\n{border}")
    print(f"  {label}")
    print(f"  Style: {gen.metadata.get('style', 'unknown')}")
    print(f"  Citations: {len(gen.citations)}")
    print(border)
    wrapped = textwrap.fill(gen.output, width=72)
    print(wrapped)
    print()


async def run_config(name: str, retrieval, generation, query: str) -> GenerationResult:
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    graph = build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=IdentityReranking(),
    )
    state = await graph.ainvoke({
        "query": query,
        "query_id": "real_demo_1",
        "answers": [],
        "test_data_name": "nq",
    })
    gen: GenerationResult = state["generation_result"]
    _print_result(name, gen)
    return gen


async def main() -> None:
    query = os.environ.get(
        "DEMO_QUERY",
        "What were the main causes and consequences of the 2023 Silicon Valley Bank collapse?",
    )

    print(f"Query: {query}\n")
    print("Initializing LLM client...")
    llm = _LLM()
    print(f"  Model: {llm.model}")

    llm_retrieval = LLMRetrieval(llm=llm)
    ddg_retrieval = DuckDuckGoRetrieval(k=5)
    ddg_with_fallback = FallbackRetrieval(primary=ddg_retrieval, fallback=llm_retrieval)
    longrag_gen = LongRAGReaderGeneration(llm=llm)
    storm_gen = StormWriterGeneration(llm=llm)

    print("\n--- Config A: LongRAG default (LLM retrieval + LongRAG reader) ---")
    result_a = await run_config(
        "A) LongRAG Default", llm_retrieval, longrag_gen, query
    )

    print("\n--- Config B: STORM retrieval swap (DuckDuckGo + LongRAG reader) ---")
    result_b = await run_config(
        "B) STORM Retrieval -> LongRAG Reader", ddg_retrieval, longrag_gen, query
    )

    print("\n--- Config C: STORM generation swap (LLM retrieval + STORM writer) ---")
    result_c = await run_config(
        "C) LLM Retrieval -> STORM Writer", llm_retrieval, storm_gen, query
    )

    print("\n--- Config D: DDG with fallback + STORM writer (full cross-project) ---")
    result_d = await run_config(
        "D) DDG+Fallback -> STORM Writer", ddg_with_fallback, storm_gen, query
    )

    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)
    print(f"  A) LongRAG default         : {len(result_a.output):>5} chars, {len(result_a.citations)} citations")
    print(f"  B) DDG retrieval only      : {len(result_b.output):>5} chars, {len(result_b.citations)} citations")
    print(f"  C) STORM generation        : {len(result_c.output):>5} chars, {len(result_c.citations)} citations")
    print(f"  D) DDG+fallback + STORM gen: {len(result_d.output):>5} chars, {len(result_d.citations)} citations")
    print("\nDemo complete. Cross-project component swap with real LLM verified.")


if __name__ == "__main__":
    asyncio.run(main())
