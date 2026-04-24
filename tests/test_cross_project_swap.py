"""Integration tests: cross-project component swap builds and runs a graph.

Verifies that components from LightRAG and Self-RAG can be injected into
LongRAG's pipeline and the combined graph executes correctly.
"""

from __future__ import annotations

import asyncio

import pytest

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    RetrievalResult,
)
from longRAG_example.longrag_langgraph.main_pipeline import build_graph


# ═══════════════════════════════════════════════════════════════════════════════
# Fake components that simulate project-specific implementations
# ═══════════════════════════════════════════════════════════════════════════════


class LongRAGStyleRetrieval:
    """Simulates LongRAG's dataset-backed retrieval."""

    NAME = "longrag-tevatron"

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                source_id=f"longrag://{i}",
                content=f"LongRAG context for {q}",
                score=1.0,
                title=f"LongRAG-Doc-{i}",
            )
            for i, q in enumerate(queries[:top_k])
        ]


class LightRAGStyleRetrieval:
    """Simulates LightRAG's KG-based retrieval returning multiple chunks."""

    NAME = "lightrag-hybrid"

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for i, q in enumerate(queries):
            results.append(RetrievalResult(
                source_id=f"lightrag-chunk://{i}",
                content=f"KG-enriched context for {q}",
                score=0.9 - i * 0.05,
                title=f"LightRAG-KG-Doc-{i}",
                metadata={"retrieval_mode": "hybrid", "has_kg": True},
            ))
        return results[:top_k]


class LongRAGStyleGeneration:
    """Simulates LongRAG's LLM reader."""

    NAME = "longrag-reader"

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            output=f"[LongRAG] answer to '{query}'",
            citations=[r.source_id for r in context],
        )


class LightRAGStyleGeneration:
    """Simulates LightRAG's detailed answer generation."""

    NAME = "lightrag-answer"

    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult:
        return GenerationResult(
            output=f"[LightRAG] detailed answer for '{query}'",
            citations=[r.source_id for r in context],
            metadata={"framework": "lightrag"},
        )


class ScoreBoostReranking:
    """A custom reranker that boosts scores by a fixed amount."""

    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]:
        boosted = sorted(results, key=lambda r: r.score, reverse=True)
        return boosted[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


def _run(coro):
    """Helper to run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestDefaultPipeline:
    """LongRAG pipeline with its own default components."""

    def test_default_builds_and_runs(self):
        graph = build_graph(
            retrieval=LongRAGStyleRetrieval(),
            generation=LongRAGStyleGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "What is the capital of France?",
            "query_id": "t1",
            "answers": ["Paris"],
            "test_data_name": "nq",
        }))
        assert "generation_result" in state
        gen: GenerationResult = state["generation_result"]
        assert "[LongRAG]" in gen.output
        assert len(gen.citations) > 0
        assert gen.citations[0].startswith("longrag://")

    def test_retrieval_results_populated(self):
        graph = build_graph(
            retrieval=LongRAGStyleRetrieval(),
            generation=LongRAGStyleGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "test",
            "query_id": "t2",
            "answers": [],
            "test_data_name": "nq",
        }))
        assert "retrieval_results" in state
        assert len(state["retrieval_results"]) > 0


class TestCrossProjectSwapRetrieval:
    """Swap LongRAG's retrieval with LightRAG's KG retrieval."""

    def test_lightrag_retrieval_in_longrag_pipeline(self):
        graph = build_graph(
            retrieval=LightRAGStyleRetrieval(),
            generation=LongRAGStyleGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "What is LangGraph?",
            "query_id": "swap1",
            "answers": [],
            "test_data_name": "nq",
        }))
        gen: GenerationResult = state["generation_result"]
        assert "[LongRAG]" in gen.output
        assert all(c.startswith("lightrag-chunk://") for c in gen.citations)


class TestCrossProjectSwapGeneration:
    """Swap LongRAG's generation with LightRAG's generation."""

    def test_lightrag_generation_in_longrag_pipeline(self):
        graph = build_graph(
            retrieval=LongRAGStyleRetrieval(),
            generation=LightRAGStyleGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "Explain transformers",
            "query_id": "swap2",
            "answers": [],
            "test_data_name": "nq",
        }))
        gen: GenerationResult = state["generation_result"]
        assert "[LightRAG]" in gen.output
        assert all(c.startswith("longrag://") for c in gen.citations)


class TestCrossProjectSwapBoth:
    """Swap both retrieval AND generation simultaneously."""

    def test_both_lightrag_components_in_longrag(self):
        graph = build_graph(
            retrieval=LightRAGStyleRetrieval(),
            generation=LightRAGStyleGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "AI safety",
            "query_id": "swap3",
            "answers": [],
            "test_data_name": "nq",
        }))
        gen: GenerationResult = state["generation_result"]
        assert "[LightRAG]" in gen.output
        assert all(c.startswith("lightrag-chunk://") for c in gen.citations)


class TestCustomReranking:
    """Inject a custom reranker into the pipeline."""

    def test_reranker_affects_results(self):
        graph = build_graph(
            retrieval=LongRAGStyleRetrieval(),
            generation=LongRAGStyleGeneration(),
            reranking=ScoreBoostReranking(),
        )
        state = _run(graph.ainvoke({
            "query": "test reranking",
            "query_id": "rerank1",
            "answers": [],
            "test_data_name": "nq",
        }))
        assert "generation_result" in state
        assert len(state["retrieval_results"]) > 0

    def test_identity_reranking_explicit(self):
        graph = build_graph(
            retrieval=LongRAGStyleRetrieval(),
            generation=LongRAGStyleGeneration(),
            reranking=IdentityReranking(),
        )
        state = _run(graph.ainvoke({
            "query": "test identity",
            "query_id": "rerank2",
            "answers": [],
            "test_data_name": "nq",
        }))
        assert "generation_result" in state


class TestProtocolDuckTyping:
    """Verify that any object with the right method signature works (duck typing)."""

    def test_anonymous_retrieval_class(self):
        class InlineRetrieval:
            def retrieve(self, queries, top_k=10):
                return [
                    RetrievalResult(source_id="inline://0", content="inline")
                ]

        class InlineGeneration:
            def generate(self, query, context, instruction=""):
                return GenerationResult(output="inline-answer")

        graph = build_graph(
            retrieval=InlineRetrieval(),
            generation=InlineGeneration(),
        )
        state = _run(graph.ainvoke({
            "query": "duck typing test",
            "query_id": "duck1",
            "answers": [],
            "test_data_name": "nq",
        }))
        assert state["generation_result"].output == "inline-answer"
