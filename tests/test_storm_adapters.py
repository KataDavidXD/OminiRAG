"""Unit tests for STORM <-> canonical adapter round-trips."""

from __future__ import annotations

import pytest

from rag_contracts import (
    GenerationResult,
    IdentityReranking,
    QueryContext,
    RetrievalResult,
)
from storm.storm_langgraph.adapters import (
    CanonicalToStormRetriever,
    StormGenerationAdapter,
    StormQueryAdapter,
    StormRetrievalAdapter,
    StormRerankingAdapter,
    information_to_retrieval_result,
    retrieval_result_to_information,
)
from storm.storm_langgraph.types import (
    DialogueTurn,
    Information,
    StormInformationTable,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Conversion helper round-trips
# ═══════════════════════════════════════════════════════════════════════════════


class TestConversionRoundTrip:
    def test_information_to_retrieval_result_preserves_fields(self):
        info = Information(
            url="https://example.com",
            description="desc",
            snippets=["snip1", "snip2"],
            title="Example",
            meta={"key": "val"},
        )
        rr = information_to_retrieval_result(info)
        assert rr.source_id == "https://example.com"
        assert rr.title == "Example"
        assert "snip1" in rr.content
        assert "snip2" in rr.content
        assert rr.metadata["description"] == "desc"
        assert rr.metadata["snippets"] == ["snip1", "snip2"]
        assert rr.metadata["key"] == "val"

    def test_retrieval_result_to_information_preserves_fields(self):
        rr = RetrievalResult(
            source_id="https://example.com",
            content="full text",
            score=0.9,
            title="Example",
            metadata={"description": "desc", "snippets": ["s1"]},
        )
        info = retrieval_result_to_information(rr)
        assert info.url == "https://example.com"
        assert info.title == "Example"
        assert info.snippets == ["s1"]
        assert info.description == "desc"

    def test_round_trip_information(self):
        original = Information(
            url="http://x.com",
            description="d",
            snippets=["a", "b"],
            title="T",
            meta={"extra": 1},
        )
        rr = information_to_retrieval_result(original)
        restored = retrieval_result_to_information(rr)
        assert restored.url == original.url
        assert restored.title == original.title
        assert restored.snippets == original.snippets
        assert restored.description == original.description

    def test_retrieval_result_without_snippets_uses_content(self):
        rr = RetrievalResult(
            source_id="s1", content="fallback content", metadata={}
        )
        info = retrieval_result_to_information(rr)
        assert info.snippets == ["fallback content"]

    def test_retrieval_result_empty_content_gives_empty_snippets(self):
        rr = RetrievalResult(source_id="s1", content="", metadata={})
        info = retrieval_result_to_information(rr)
        assert info.snippets == []


# ═══════════════════════════════════════════════════════════════════════════════
# Forward adapter: StormRetrievalAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeStormRetriever:
    def retrieve(self, queries, exclude_urls=None):
        return [
            Information(
                url=f"http://fake/{i}",
                description=f"desc-{q}",
                snippets=[f"snippet for {q}"],
                title=f"Title-{q}",
            )
            for i, q in enumerate(queries)
        ]


class TestStormRetrievalAdapter:
    def test_converts_to_canonical(self):
        adapter = StormRetrievalAdapter(storm_retriever=_FakeStormRetriever())
        results = adapter.retrieve(["q1", "q2"], top_k=5)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].source_id == "http://fake/0"
        assert results[1].title == "Title-q2"

    def test_top_k_limits(self):
        adapter = StormRetrievalAdapter(storm_retriever=_FakeStormRetriever())
        results = adapter.retrieve(["a", "b", "c", "d"], top_k=2)
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Forward adapter: StormRerankingAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class TestStormRerankingAdapter:
    def test_reranks_from_info_table(self):
        turn = DialogueTurn(
            user_utterance="q",
            agent_utterance="a",
            search_results=[
                Information(
                    url="http://doc1",
                    description="d1",
                    snippets=["the capital of France is Paris"],
                    title="France",
                ),
                Information(
                    url="http://doc2",
                    description="d2",
                    snippets=["Berlin is the capital of Germany"],
                    title="Germany",
                ),
            ],
        )
        table = StormInformationTable(conversations=[("test", [turn])])
        table.rebuild()
        table.prepare_for_retrieval()

        adapter = StormRerankingAdapter(info_table=table)
        results = adapter.rerank("capital of France", [], top_k=5)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Forward adapter: StormGenerationAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeSectionWriter:
    def write_section(self, topic, section_name, section_outline, collected_info):
        return f"Section about {section_name} for {topic} with {len(collected_info)} sources."


class TestStormGenerationAdapter:
    def test_generates_from_canonical_context(self):
        adapter = StormGenerationAdapter(section_writer=_FakeSectionWriter())
        ctx = [
            RetrievalResult(
                source_id="http://x", content="data", metadata={"snippets": ["data"]}
            )
        ]
        result = adapter.generate("AI", ctx, instruction="History of AI")
        assert isinstance(result, GenerationResult)
        assert "History of AI" in result.output
        assert "1 sources" in result.output
        assert result.citations == ["http://x"]


# ═══════════════════════════════════════════════════════════════════════════════
# Forward adapter: StormQueryAdapter
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeQuestionAsker:
    def ask(self, topic, persona, dialogue_history):
        return f"What is {topic} about?"


class _FakeQueryGenerator:
    def generate_queries(self, topic, question, max_queries):
        return [f"{topic} overview", f"{topic} details"][:max_queries]


class TestStormQueryAdapter:
    def test_produces_expanded_queries(self):
        adapter = StormQueryAdapter(
            question_asker=_FakeQuestionAsker(),
            query_generator=_FakeQueryGenerator(),
            persona="Writer",
        )
        ctx = QueryContext(topic="LLMs")
        queries = adapter.process("LLMs", ctx)
        assert len(queries) == 2
        assert "LLMs overview" in queries

    def test_fallback_on_termination_phrase(self):
        class _TermAsker:
            def ask(self, topic, persona, history):
                return "Thank you so much for your help!"

        adapter = StormQueryAdapter(
            question_asker=_TermAsker(),
            query_generator=_FakeQueryGenerator(),
        )
        queries = adapter.process("test", QueryContext())
        assert queries == ["test"]


# ═══════════════════════════════════════════════════════════════════════════════
# Reverse adapter: CanonicalToStormRetriever
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeCanonicalRetrieval:
    def retrieve(self, queries, top_k=10):
        return [
            RetrievalResult(
                source_id=f"canonical://{q}",
                content=f"content for {q}",
                title=f"Title-{q}",
                metadata={"snippets": [f"snip-{q}"]},
            )
            for q in queries[:top_k]
        ]


class TestCanonicalToStormRetriever:
    def test_wraps_canonical_to_storm_interface(self):
        adapter = CanonicalToStormRetriever(
            canonical_retrieval=_FakeCanonicalRetrieval()
        )
        infos = adapter.retrieve(["q1", "q2"])
        assert len(infos) == 2
        assert all(isinstance(i, Information) for i in infos)
        assert infos[0].url == "canonical://q1"
        assert infos[0].snippets == ["snip-q1"]

    def test_exclude_urls(self):
        adapter = CanonicalToStormRetriever(
            canonical_retrieval=_FakeCanonicalRetrieval()
        )
        infos = adapter.retrieve(["q1", "q2"], exclude_urls=["canonical://q1"])
        assert len(infos) == 1
        assert infos[0].url == "canonical://q2"
