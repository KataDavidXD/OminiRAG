"""Layer 1: Real dataset format compatibility tests.

Validates that real HotpotQA and UltraDomain datasets load correctly and
convert into the formats expected by OminiRAG's benchmark adapters.
Also checks alignment between ``bsamp`` loader output and adapter expectations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_contracts import GenerationResult, RetrievalResult

HOTPOTQA_DIR = Path("/data1/ragworkspace/dataset/all_data/hotpotqa")
ULTRADOMAIN_DIR = Path("/data1/ragworkspace/dataset/UltraDomain")

skip_no_hotpotqa = pytest.mark.skipif(
    not HOTPOTQA_DIR.exists(), reason="HotpotQA dataset not available"
)
skip_no_ultradomain = pytest.mark.skipif(
    not ULTRADOMAIN_DIR.exists(), reason="UltraDomain dataset not available"
)


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

@skip_no_hotpotqa
class TestHotpotQARealFormat:

    def test_hotpotqa_real_json_loads(self):
        from benchmark.hotpotqa_adapter import load_hotpotqa_real
        items = load_hotpotqa_real(HOTPOTQA_DIR, max_items=10)
        assert len(items) == 10
        for item in items:
            assert "question" in item
            assert "answer" in item
            assert "context_results" in item
            assert isinstance(item["context_results"], list)

    def test_hotpotqa_context_to_retrieval_results(self):
        from benchmark.base_adapter import hotpotqa_context_to_retrieval_results
        context = [
            ["Title A", ["Sentence one.", "Sentence two."]],
            ["Title B", ["Another sentence."]],
        ]
        results = hotpotqa_context_to_retrieval_results(context)
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].title == "Title A"
        assert "Sentence one." in results[0].content
        assert "Sentence two." in results[0].content
        assert results[1].title == "Title B"
        assert results[0].score > results[1].score

    def test_hotpotqa_context_handles_malformed_entries(self):
        from benchmark.base_adapter import hotpotqa_context_to_retrieval_results
        context = [
            ["Good Title", ["sent"]],
            "not a list",
            [],
            ["Only title"],
        ]
        results = hotpotqa_context_to_retrieval_results(context)
        assert len(results) == 1
        assert results[0].title == "Good Title"

    def test_hotpotqa_real_to_adapter_format(self):
        """Real data items work with evaluate_generation using a mock generator."""
        from benchmark.hotpotqa_adapter import (
            HotpotQABenchmarkAdapter,
            load_hotpotqa_real,
        )

        items = load_hotpotqa_real(HOTPOTQA_DIR, max_items=3)

        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(
            output="mock answer", citations=["c1"]
        )

        adapter = HotpotQABenchmarkAdapter()
        result = adapter.evaluate_generation(items, mock_gen)
        assert result.num_items == 3
        assert mock_gen.generate.call_count == 3
        for call_args in mock_gen.generate.call_args_list:
            ctx = call_args.kwargs.get("context") or call_args[1].get("context", [])
            if not ctx:
                ctx = call_args[0][1] if len(call_args[0]) > 1 else []
            assert isinstance(ctx, list)

    def test_hotpotqa_full_dataset_count(self):
        from benchmark.hotpotqa_adapter import load_hotpotqa_real
        items = load_hotpotqa_real(HOTPOTQA_DIR)
        assert len(items) >= 100, f"Expected >=100 items, got {len(items)}"


# ---------------------------------------------------------------------------
# UltraDomain
# ---------------------------------------------------------------------------

@skip_no_ultradomain
class TestUltraDomainRealFormat:

    def test_ultradomain_real_jsonl_loads(self):
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ULTRADOMAIN_DIR, domain="mix", max_items=10)
        assert len(items) == 10
        for item in items:
            assert "question" in item
            assert "answer" in item
            assert "context_results" in item
            assert isinstance(item["context_results"], list)
            assert len(item["context_results"]) > 0

    def test_ultradomain_context_to_retrieval_results(self):
        from benchmark.base_adapter import ultradomain_context_to_retrieval_results
        long_text = "A" * 10000
        results = ultradomain_context_to_retrieval_results(long_text, max_chunk_chars=4000)
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert len(results[0].content) == 4000
        assert len(results[2].content) == 2000

    def test_ultradomain_context_empty(self):
        from benchmark.base_adapter import ultradomain_context_to_retrieval_results
        results = ultradomain_context_to_retrieval_results("")
        assert results == []

    def test_ultradomain_real_to_adapter_format(self):
        """Real data items work with evaluate_generation using a mock generator."""
        from benchmark.ultradomain_adapter import (
            UltraDomainBenchmarkAdapter,
            load_ultradomain_real,
        )

        items = load_ultradomain_real(ULTRADOMAIN_DIR, domain="mix", max_items=2)

        mock_gen = MagicMock()
        mock_gen.generate.return_value = GenerationResult(
            output="mock answer", citations=["c1"]
        )

        adapter = UltraDomainBenchmarkAdapter()
        result = adapter.evaluate_generation(items, mock_gen)
        assert result.num_items == 2
        assert mock_gen.generate.call_count == 2

    def test_ultradomain_field_mapping(self):
        """Verify the real JSONL 'input' field maps to 'question'."""
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ULTRADOMAIN_DIR, domain="mix", max_items=1)
        assert items[0]["question"], "question should not be empty"
        assert items[0]["answer"], "answer should not be empty"

    def test_ultradomain_multiple_domains(self):
        from benchmark.ultradomain_adapter import load_ultradomain_real
        for domain in ("agriculture", "cs", "legal", "mix"):
            path = ULTRADOMAIN_DIR / f"{domain}.jsonl"
            if path.exists():
                items = load_ultradomain_real(ULTRADOMAIN_DIR, domain=domain, max_items=2)
                assert len(items) > 0, f"No items loaded for domain {domain}"


# ---------------------------------------------------------------------------
# bsamp alignment
# ---------------------------------------------------------------------------

class TestBsampAlignment:

    def test_bsamp_hotpotqa_format_aligns_with_adapter(self):
        """bsamp HotpotQA ``_row_to_item`` output uses ``question``/``answer`` --
        same field names the benchmark adapter expects.  Also exposes
        ``context_titles`` / ``context_sentences`` which can be converted to
        ``context_results`` via ``hotpotqa_context_to_retrieval_results``.

        The gap: bsamp returns ``context_titles`` + ``context_sentences`` (flat
        lists) whereas the raw HotpotQA JSON uses ``context`` as a nested list
        of ``[title, [sents]]``.  The adapter now handles both.
        """
        try:
            from bsamp.loader.hotpot_qa import HotpotQAAPI
        except ImportError:
            pytest.skip("bsamp.loader.hotpot_qa not available")

        bsamp_item_keys = {
            "id", "question", "answer", "type", "level",
            "context_titles", "context_sentences", "context_text",
            "supporting_facts_titles", "supporting_facts_sent_ids",
        }
        adapter_required = {"question", "answer"}
        assert adapter_required <= bsamp_item_keys, (
            f"bsamp item missing adapter-required keys: "
            f"{adapter_required - bsamp_item_keys}"
        )

    def test_bsamp_ultradomain_format_aligns_with_adapter(self):
        """bsamp UltraDomain loader returns items with ``query``/``answer``/
        ``context`` whereas the adapter expects ``question``/``answer``.

        The gap: bsamp uses ``query`` while the adapter uses ``question``.
        ``load_ultradomain_real()`` maps the raw field ``input`` -> ``question``,
        but bsamp uses yet another name (``query``).  Any bridge between the two
        must rename the field.
        """
        try:
            from bsamp.loader.UltraDomain import UltraDomainAPI
        except ImportError:
            pytest.skip("bsamp.loader.UltraDomain not available")

        bsamp_item_keys = {"query", "answer", "context", "domain"}
        adapter_required = {"question", "answer"}
        field_gap = adapter_required - bsamp_item_keys
        assert "question" in field_gap, (
            "Expected 'question' to be in the gap (bsamp uses 'query' instead)"
        )
        assert "answer" not in field_gap, "Both should share 'answer'"


# ---------------------------------------------------------------------------
# get_context_for_item fallback logic
# ---------------------------------------------------------------------------

class TestGetContextForItem:

    def test_context_results_priority(self):
        from benchmark.base_adapter import get_context_for_item
        pre_built = [RetrievalResult(source_id="r1", content="pre")]
        item = {"context_results": pre_built, "chunks": {"c1": {"content": "chunk", "doc_ids": []}}}
        result = get_context_for_item(item)
        assert result is pre_built

    def test_chunks_fallback(self):
        from benchmark.base_adapter import get_context_for_item
        item = {"chunks": {"c1": {"content": "chunk text", "doc_ids": ["d1"]}}}
        result = get_context_for_item(item)
        assert len(result) == 1
        assert result[0].content == "chunk text"

    def test_empty_fallback(self):
        from benchmark.base_adapter import get_context_for_item
        result = get_context_for_item({})
        assert result == []
