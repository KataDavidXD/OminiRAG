"""Tests for benchmark adapters (HotpotQA, UltraDomain, ALCE) with sample data.

Uses the KG-format sample data under benchmark/sample_data/ to verify that
adapters correctly load data, run generation, and compute metrics.
All tests use mock generation components -- no LLM calls required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_contracts import GenerationResult, RetrievalResult


SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "sample_data"


# ═══════════════════════════════════════════════════════════════════════════════
# Mock generation that returns predictable output
# ═══════════════════════════════════════════════════════════════════════════════

class MockGeneration:
    """Returns a fixed answer or echoes part of the context."""

    def __init__(self, fixed_answer: str | None = None):
        self.fixed_answer = fixed_answer

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        if self.fixed_answer:
            return GenerationResult(
                output=self.fixed_answer,
                citations=[r.source_id for r in context],
            )
        if context:
            return GenerationResult(
                output=context[0].content[:200],
                citations=[r.source_id for r in context],
            )
        return GenerationResult(output="no context available", citations=[])


# ═══════════════════════════════════════════════════════════════════════════════
# HotpotQA adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHotpotQAAdapter:
    def test_load_sample_data(self):
        from benchmark.hotpotqa_adapter import load_hotpotqa_sample

        data = load_hotpotqa_sample(SAMPLE_DATA_DIR / "hotpotqa_kg_sample")
        assert len(data) == 3
        assert all("question" in d for d in data)
        assert all("answer" in d for d in data)
        assert all("chunks" in d for d in data)

    def test_evaluate_generation(self):
        from benchmark.hotpotqa_adapter import (
            HotpotQABenchmarkAdapter,
            load_hotpotqa_sample,
        )

        data = load_hotpotqa_sample(SAMPLE_DATA_DIR / "hotpotqa_kg_sample")
        adapter = HotpotQABenchmarkAdapter()
        gen = MockGeneration(fixed_answer="Yes")
        result = adapter.evaluate_generation(data, gen)

        assert result.num_items == 3
        assert len(result.per_item) == 3
        assert 0.0 <= result.avg_em <= 100.0
        assert 0.0 <= result.avg_f1 <= 100.0
        assert result.per_item[0]["em"] == 1

    def test_exact_match_scoring(self):
        from benchmark.hotpotqa_adapter import compute_exact, compute_f1

        assert compute_exact("Yes", "yes") == 1
        assert compute_exact("Paris", "London") == 0
        assert compute_f1("Arthur's Magazine", "Arthur's Magazine") == 1.0
        assert compute_f1("Paris", "the city of Paris") > 0.0

    def test_chunks_to_retrieval_results(self):
        from benchmark.hotpotqa_adapter import sample_chunks_to_retrieval_results

        chunks = {
            "c1": {"content": "text1", "doc_ids": ["d1"]},
            "c2": {"content": "text2", "doc_ids": []},
        }
        results = sample_chunks_to_retrieval_results(chunks)
        assert len(results) == 2
        assert results[0].source_id == "c1"


# ═══════════════════════════════════════════════════════════════════════════════
# UltraDomain adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUltraDomainAdapter:
    def test_load_sample_data(self):
        from benchmark.ultradomain_adapter import load_ultradomain_sample

        data = load_ultradomain_sample(SAMPLE_DATA_DIR / "ultradomain_kg_sample")
        assert len(data) == 3
        assert all("question" in d for d in data)
        assert all("domain" in d for d in data)
        domains = {d["domain"] for d in data}
        assert "agriculture" in domains
        assert "cs" in domains
        assert "legal" in domains

    def test_evaluate_generation_no_llm(self):
        from benchmark.ultradomain_adapter import (
            UltraDomainBenchmarkAdapter,
            load_ultradomain_sample,
        )

        data = load_ultradomain_sample(SAMPLE_DATA_DIR / "ultradomain_kg_sample")
        adapter = UltraDomainBenchmarkAdapter()
        gen = MockGeneration()
        result = adapter.evaluate_generation(data, gen)

        assert result.num_items == 3
        assert len(result.per_item) == 3
        assert result.avg_comprehensiveness == 0.0
        assert result.avg_length > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ALCE adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestALCEAdapterWithSampleData:
    def _load_alce_sample_data(self) -> list[dict]:
        queries_path = SAMPLE_DATA_DIR / "alce_kg_sample" / "queries.jsonl"
        docs_path = SAMPLE_DATA_DIR / "alce_kg_sample" / "alce_docs.json"

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

    def test_load_alce_sample_data(self):
        data = self._load_alce_sample_data()
        assert len(data) == 3
        assert all(len(d["docs"]) > 0 for d in data)

    def test_evaluate_with_alce_adapter(self):
        from benchmark.alce_adapter import ALCEBenchmarkAdapter

        data = self._load_alce_sample_data()
        adapter = ALCEBenchmarkAdapter()
        gen = MockGeneration()
        result = adapter.evaluate_generation(data, gen)

        assert result.num_items == 3
        assert all("f1" in item for item in result.per_item)

    def test_alce_doc_retrieval_wraps_docs(self):
        from rag_contracts import ALCEDocRetrieval

        data = self._load_alce_sample_data()
        docs = data[0]["docs"]
        retrieval = ALCEDocRetrieval(docs=docs)
        results = retrieval.retrieve(["any query"])

        assert len(results) == len(docs)
        assert results[0].title == docs[0]["title"]
        assert results[0].content == docs[0]["text"]


# ═══════════════════════════════════════════════════════════════════════════════
# KG sample data structural validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestKGSampleDataStructure:
    """Verify that all KG sample data files have the expected structure."""

    @pytest.mark.parametrize("benchmark", ["hotpotqa_kg_sample", "ultradomain_kg_sample"])
    def test_required_files_exist(self, benchmark):
        d = SAMPLE_DATA_DIR / benchmark
        assert (d / "queries.jsonl").exists()
        assert (d / "chunks.json").exists()
        assert (d / "graph.json").exists()
        assert (d / "kv.json").exists()
        assert (d / "vdb_chunks.json").exists()
        assert (d / "vdb_entities.json").exists()
        assert (d / "vdb_relations.json").exists()

    @pytest.mark.parametrize("benchmark", ["hotpotqa_kg_sample", "ultradomain_kg_sample"])
    def test_graph_has_nodes_and_edges(self, benchmark):
        d = SAMPLE_DATA_DIR / benchmark
        with open(d / "graph.json", encoding="utf-8") as f:
            graph = json.load(f)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0
        for node in graph["nodes"]:
            assert "name" in node
            assert "type" in node

    def test_alce_has_docs(self):
        d = SAMPLE_DATA_DIR / "alce_kg_sample"
        assert (d / "alce_docs.json").exists()
        with open(d / "alce_docs.json", encoding="utf-8") as f:
            docs = json.load(f)
        for key, doc_list in docs.items():
            if key.startswith("_"):
                continue
            assert len(doc_list) > 0
            assert all("title" in doc and "text" in doc for doc in doc_list)
