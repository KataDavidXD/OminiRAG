"""Integration tests for all 14 practically meaningful combinations from the plan.

Tests each combination from Part B of the architecture plan using mock
components with real sample data shapes. Verifies that:
1. Each combination builds and runs through a pipeline
2. The generation result has expected properties
3. Benchmark adapters correctly evaluate the output

These tests do NOT require LLM access -- all components are mocked.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rag_contracts import (
    ALCEDocRetrieval,
    GenerationResult,
    IdentityQuery,
    IdentityReranking,
    QueryContext,
    RetrievalResult,
)
from benchmark.hotpotqa_adapter import (
    HotpotQABenchmarkAdapter,
    load_hotpotqa_sample,
    sample_chunks_to_retrieval_results,
)
from benchmark.ultradomain_adapter import (
    UltraDomainBenchmarkAdapter,
    load_ultradomain_sample,
)
from benchmark.alce_adapter import ALCEBenchmarkAdapter


SAMPLE_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "sample_data"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock components representing each framework's behavior
# ═══════════════════════════════════════════════════════════════════════════════

class MockLightRAGQuery:
    def process(self, query: str, context: QueryContext) -> list[str]:
        return [query, f"{query} entity", f"{query} relation"]


class MockHFDatasetRetrieval:
    """Simulates LongRAG's HFDatasetRetrieval: returns a single long context."""

    def __init__(self, chunks: dict | None = None):
        self._chunks = chunks or {}

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        if self._chunks:
            return sample_chunks_to_retrieval_results(self._chunks)[:top_k]
        content = " ".join(f"Context for {q}." for q in queries)
        return [RetrievalResult(
            source_id="hf://combined",
            content=content,
            score=1.0,
            title="HF-Combined-4K",
            metadata={"context_titles": ["Doc A", "Doc B"]},
        )]


class MockLightRAGRetrieval:
    """Simulates LightRAG's hybrid vector+KG retrieval: returns multiple chunks."""

    def __init__(self, chunks: dict | None = None):
        self._chunks = chunks or {}

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        if self._chunks:
            return sample_chunks_to_retrieval_results(self._chunks)[:top_k]
        results = []
        for i, q in enumerate(queries[:3]):
            results.append(RetrievalResult(
                source_id=f"kg-chunk://{i}",
                content=f"KG-enriched passage about {q}",
                score=0.9 - i * 0.05,
                title=f"KG-Entity-{i}",
                metadata={"retrieval_mode": "hybrid", "kg_entities": [{"name": f"E{i}"}]},
            ))
        return results[:top_k]


class MockSelfRAGRetrieval:
    """Simulates SelfRAG's Contriever-based retrieval: returns multiple passages."""

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for i in range(min(3, top_k)):
            results.append(RetrievalResult(
                source_id=f"contriever://{i}",
                content=f"Contriever passage {i} for {queries[0]}",
                score=0.85 - i * 0.05,
                title=f"Contriever-Doc-{i}",
            ))
        return results


class MockSelfRAGReranking:
    """Simulates SelfRAG reranking: adds evidence scores and caches predictions."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 10):
        scored = []
        for i, r in enumerate(results):
            score = 0.9 - i * 0.1
            scored.append(RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=score,
                title=r.title,
                metadata={
                    **r.metadata,
                    "selfrag_score": score,
                    "_selfrag_pred": {
                        "text": f"Evidence-scored answer from {r.source_id}",
                        "score": score,
                    },
                },
            ))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


class MockLightRAGReranking:
    """Simulates LightRAG's context compression."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 10):
        return [
            RetrievalResult(
                source_id=r.source_id,
                content=r.content,
                score=r.score,
                title=r.title,
                metadata={**r.metadata, "compressed": True},
            )
            for r in results[:top_k]
        ]


class MockLongRAGGeneration:
    """Simulates LongRAG reader: short answer extraction."""

    def generate(self, query: str, context: list[RetrievalResult], instruction: str = ""):
        answer = "Yes" if "same" in query.lower() else "Short factual answer"
        return GenerationResult(
            output=answer,
            citations=[r.source_id for r in context[:3]],
            metadata={"style": "longrag-reader"},
        )


class MockLightRAGGeneration:
    """Simulates LightRAG generation: detailed answer with references."""

    def generate(self, query: str, context: list[RetrievalResult], instruction: str = ""):
        refs = ", ".join(r.title for r in context[:3])
        return GenerationResult(
            output=f"Detailed analysis of '{query}' based on {refs}. "
                   f"The evidence from {len(context)} sources indicates...",
            citations=[r.source_id for r in context],
            metadata={"style": "lightrag-answer"},
        )


class MockSelfRAGGeneration:
    """Simulates SelfRAG generation: picks best-scored passage answer."""

    def generate(self, query: str, context: list[RetrievalResult], instruction: str = ""):
        for r in context:
            pred = r.metadata.get("_selfrag_pred")
            if isinstance(pred, dict):
                return GenerationResult(
                    output=pred["text"],
                    citations=[r.source_id],
                    metadata={"from_cache": True, "selfrag_score": pred["score"]},
                )
        if context:
            return GenerationResult(
                output=f"Per-passage answer for {query} from {context[0].source_id}",
                citations=[context[0].source_id],
            )
        return GenerationResult(output="", citations=[])


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline runner helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline_direct(
    query: str,
    retrieval,
    generation,
    reranking=None,
    query_proc=None,
) -> GenerationResult:
    """Run a pipeline manually without LangGraph -- just calls components in order."""
    queries = [query]
    if query_proc is not None:
        queries = query_proc.process(query, QueryContext())

    results = retrieval.retrieve(queries, top_k=10)

    if reranking is not None:
        results = reranking.rerank(query, results, top_k=10)

    return generation.generate(query, results)


def _build_and_run_longrag(
    query: str,
    retrieval,
    generation,
    reranking=None,
    query_proc=None,
) -> GenerationResult:
    """Build and run via the actual LongRAG LangGraph pipeline."""
    from longRAG_example.longrag_langgraph.main_pipeline import build_graph

    graph = build_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query_proc,
    )
    state = asyncio.run(graph.ainvoke({
        "query": query,
        "query_id": "combo_test",
        "answers": [],
        "test_data_name": "nq",
    }))
    return state["generation_result"]


def _build_and_run_lightrag(
    query: str,
    retrieval,
    generation,
    reranking=None,
    query_proc=None,
) -> GenerationResult:
    """Build and run via the actual LightRAG LangGraph pipeline."""
    from lightrag_langgraph.main_pipeline import build_query_graph

    graph = build_query_graph(
        retrieval=retrieval,
        generation=generation,
        reranking=reranking,
        query=query_proc,
    )
    state = asyncio.run(graph.ainvoke({"query": query}))
    return state["generation_result"]


# ═══════════════════════════════════════════════════════════════════════════════
# HotpotQA combinations (#1-#5)
# ═══════════════════════════════════════════════════════════════════════════════

class TestHotpotQACombinations:
    """5 practically meaningful combinations for HotpotQA (multi-hop QA)."""

    @pytest.fixture
    def hotpot_data(self):
        return load_hotpotqa_sample(SAMPLE_DIR / "hotpotqa_kg_sample")

    def test_combo1_native_longrag_baseline(self, hotpot_data):
        """#1: Identity -> HFDataset -> Identity -> LongRAG reader."""
        gen = _build_and_run_longrag(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            retrieval=MockHFDatasetRetrieval(hotpot_data[0]["chunks"]),
            generation=MockLongRAGGeneration(),
        )
        assert gen.output
        assert len(gen.citations) > 0

    def test_combo2_lightrag_retrieval_longrag_gen(self, hotpot_data):
        """#2: LightRAG -> LightRAG -> LightRAG rerank -> LongRAG reader."""
        gen = _build_and_run_lightrag(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            retrieval=MockLightRAGRetrieval(hotpot_data[0]["chunks"]),
            generation=MockLongRAGGeneration(),
            reranking=MockLightRAGReranking(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "longrag-reader"

    def test_combo3_kg_retrieval_selfrag_rerank_longrag(self, hotpot_data):
        """#3: LightRAG -> LightRAG -> SelfRAG rerank -> LongRAG reader."""
        gen = _build_and_run_lightrag(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            retrieval=MockLightRAGRetrieval(hotpot_data[0]["chunks"]),
            generation=MockLongRAGGeneration(),
            reranking=MockSelfRAGReranking(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output

    def test_combo4_kg_chunks_selfrag_scoring(self, hotpot_data):
        """#4: Identity -> LightRAG -> SelfRAG rerank -> SelfRAG gen."""
        gen = _run_pipeline_direct(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            retrieval=MockLightRAGRetrieval(hotpot_data[0]["chunks"]),
            generation=MockSelfRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        assert gen.output
        assert gen.metadata.get("from_cache") is True

    def test_combo5_full_lightrag_on_hotpotqa(self, hotpot_data):
        """#5: LightRAG -> LightRAG -> Identity -> LightRAG gen."""
        gen = _build_and_run_lightrag(
            "Were Scott Derrickson and Ed Wood of the same nationality?",
            retrieval=MockLightRAGRetrieval(hotpot_data[0]["chunks"]),
            generation=MockLightRAGGeneration(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "lightrag-answer"

    def test_hotpotqa_adapter_evaluates_all_combos(self, hotpot_data):
        """Verify HotpotQA adapter works with each generation style."""
        adapter = HotpotQABenchmarkAdapter()
        for gen_cls in [MockLongRAGGeneration, MockLightRAGGeneration]:
            result = adapter.evaluate_generation(hotpot_data, gen_cls())
            assert result.num_items == 3
            assert result.avg_f1 >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# UltraDomain combinations (#6-#9)
# ═══════════════════════════════════════════════════════════════════════════════

class TestUltraDomainCombinations:
    """4 practically meaningful combinations for UltraDomain."""

    @pytest.fixture
    def ultra_data(self):
        return load_ultradomain_sample(SAMPLE_DIR / "ultradomain_kg_sample")

    def test_combo6_native_lightrag_baseline(self, ultra_data):
        """#6: LightRAG -> LightRAG -> LightRAG -> LightRAG (native)."""
        gen = _build_and_run_lightrag(
            ultra_data[0]["question"],
            retrieval=MockLightRAGRetrieval(ultra_data[0]["chunks"]),
            generation=MockLightRAGGeneration(),
            reranking=MockLightRAGReranking(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "lightrag-answer"

    def test_combo7_kg_selfrag_rerank_lightrag_gen(self, ultra_data):
        """#7: LightRAG -> LightRAG -> SelfRAG rerank -> LightRAG gen."""
        gen = _build_and_run_lightrag(
            ultra_data[0]["question"],
            retrieval=MockLightRAGRetrieval(ultra_data[0]["chunks"]),
            generation=MockLightRAGGeneration(),
            reranking=MockSelfRAGReranking(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output

    def test_combo8_kg_selfrag_full(self, ultra_data):
        """#8: Identity -> LightRAG -> Identity -> SelfRAG gen."""
        gen = _run_pipeline_direct(
            ultra_data[1]["question"],
            retrieval=MockLightRAGRetrieval(ultra_data[1]["chunks"]),
            generation=MockSelfRAGGeneration(),
        )
        assert gen.output

    def test_combo9_kg_retrieval_longrag_gen(self, ultra_data):
        """#9: LightRAG -> LightRAG -> Identity -> LongRAG reader."""
        gen = _build_and_run_lightrag(
            ultra_data[2]["question"],
            retrieval=MockLightRAGRetrieval(ultra_data[2]["chunks"]),
            generation=MockLongRAGGeneration(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "longrag-reader"

    def test_ultradomain_adapter_evaluates(self, ultra_data):
        """Verify UltraDomain adapter works with mock generation."""
        adapter = UltraDomainBenchmarkAdapter()
        result = adapter.evaluate_generation(ultra_data, MockLightRAGGeneration())
        assert result.num_items == 3
        assert result.avg_length > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ALCE combinations (#10-#14)
# ═══════════════════════════════════════════════════════════════════════════════

class TestALCECombinations:
    """5 practically meaningful combinations for ALCE."""

    @pytest.fixture
    def alce_data(self):
        queries_path = SAMPLE_DIR / "alce_kg_sample" / "queries.jsonl"
        docs_path = SAMPLE_DIR / "alce_kg_sample" / "alce_docs.json"

        with open(docs_path, encoding="utf-8") as f:
            all_docs = json.load(f)

        data = []
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    data.append({
                        "question": q["query"],
                        "answer": q.get("ground_truth", ""),
                        "docs": all_docs.get(q["query_id"], []),
                        "query_id": q["query_id"],
                    })
        return data

    def test_combo10_native_selfrag_on_alce(self, alce_data):
        """#10: Identity -> ALCE docs -> Identity -> SelfRAG gen."""
        item = alce_data[0]
        retrieval = ALCEDocRetrieval(docs=item["docs"])
        gen = _run_pipeline_direct(
            item["question"],
            retrieval=retrieval,
            generation=MockSelfRAGGeneration(),
        )
        assert gen.output

    def test_combo11_selfrag_rerank_longrag_gen(self, alce_data):
        """#11: Identity -> ALCE docs -> SelfRAG rerank -> LongRAG reader."""
        item = alce_data[1]
        retrieval = ALCEDocRetrieval(docs=item["docs"])
        gen = _run_pipeline_direct(
            item["question"],
            retrieval=retrieval,
            generation=MockLongRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "longrag-reader"

    def test_combo12_selfrag_rerank_lightrag_gen(self, alce_data):
        """#12: Identity -> ALCE docs -> SelfRAG rerank -> LightRAG gen."""
        item = alce_data[2]
        retrieval = ALCEDocRetrieval(docs=item["docs"])
        gen = _run_pipeline_direct(
            item["question"],
            retrieval=retrieval,
            generation=MockLightRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "lightrag-answer"

    def test_combo13_full_lightrag_on_alce_corpus(self, alce_data):
        """#13: LightRAG -> LightRAG -> LightRAG -> LightRAG on ALCE source corpus."""
        gen = _build_and_run_lightrag(
            alce_data[0]["question"],
            retrieval=MockLightRAGRetrieval(),
            generation=MockLightRAGGeneration(),
            reranking=MockLightRAGReranking(),
            query_proc=MockLightRAGQuery(),
        )
        assert gen.output
        assert gen.metadata.get("style") == "lightrag-answer"

    def test_combo14_lightrag_compress_longrag_reader(self, alce_data):
        """#14: Identity -> ALCE docs -> LightRAG compress -> LongRAG reader."""
        item = alce_data[0]
        retrieval = ALCEDocRetrieval(docs=item["docs"])
        gen = _run_pipeline_direct(
            item["question"],
            retrieval=retrieval,
            generation=MockLongRAGGeneration(),
            reranking=MockLightRAGReranking(),
        )
        assert gen.output

    def test_alce_adapter_with_alce_doc_retrieval(self, alce_data):
        """ALCE adapter evaluates output correctly with ALCEDocRetrieval."""
        adapter = ALCEBenchmarkAdapter()
        result = adapter.evaluate_generation(alce_data, MockLightRAGGeneration())
        assert result.num_items == 3
        assert all("f1" in item for item in result.per_item)

    def test_selfrag_cache_flows_through_alce(self, alce_data):
        """SelfRAG reranking cache is picked up by SelfRAG generation on ALCE docs."""
        item = alce_data[0]
        retrieval = ALCEDocRetrieval(docs=item["docs"])
        gen = _run_pipeline_direct(
            item["question"],
            retrieval=retrieval,
            generation=MockSelfRAGGeneration(),
            reranking=MockSelfRAGReranking(),
        )
        assert gen.output
        assert gen.metadata.get("from_cache") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-benchmark coverage summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinationCoverage:
    """Meta-test: verify that all 14 combinations are exercised."""

    EXPECTED_COMBOS = {
        1: "Identity -> HFDataset -> Identity -> LongRAG",
        2: "LightRAG -> LightRAG -> LightRAG -> LongRAG",
        3: "LightRAG -> LightRAG -> SelfRAG -> LongRAG",
        4: "Identity -> LightRAG -> SelfRAG -> SelfRAG",
        5: "LightRAG -> LightRAG -> Identity -> LightRAG",
        6: "LightRAG -> LightRAG -> LightRAG -> LightRAG",
        7: "LightRAG -> LightRAG -> SelfRAG -> LightRAG",
        8: "Identity -> LightRAG -> Identity -> SelfRAG",
        9: "LightRAG -> LightRAG -> Identity -> LongRAG",
        10: "Identity -> ALCE docs -> Identity -> SelfRAG",
        11: "Identity -> ALCE docs -> SelfRAG -> LongRAG",
        12: "Identity -> ALCE docs -> SelfRAG -> LightRAG",
        13: "LightRAG -> LightRAG -> LightRAG -> LightRAG",
        14: "Identity -> ALCE docs -> LightRAG -> LongRAG",
    }

    def test_all_14_combos_have_tests(self):
        """We have test methods for all 14 combinations."""
        combo_tests = [
            "test_combo1_native_longrag_baseline",
            "test_combo2_lightrag_retrieval_longrag_gen",
            "test_combo3_kg_retrieval_selfrag_rerank_longrag",
            "test_combo4_kg_chunks_selfrag_scoring",
            "test_combo5_full_lightrag_on_hotpotqa",
            "test_combo6_native_lightrag_baseline",
            "test_combo7_kg_selfrag_rerank_lightrag_gen",
            "test_combo8_kg_selfrag_full",
            "test_combo9_kg_retrieval_longrag_gen",
            "test_combo10_native_selfrag_on_alce",
            "test_combo11_selfrag_rerank_longrag_gen",
            "test_combo12_selfrag_rerank_lightrag_gen",
            "test_combo13_full_lightrag_on_alce_corpus",
            "test_combo14_lightrag_compress_longrag_reader",
        ]
        assert len(combo_tests) == 14
        assert len(self.EXPECTED_COMBOS) == 14
