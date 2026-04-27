"""Tests for the shared OminiRAG WTB cache helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_contracts import RetrievalResult, SimpleLLMGeneration
from rag_contracts.wtb_cache import (
    WTBCacheConfig,
    WTBCachedLLM,
    inspect_swappable_system_cache_support,
)


class _FakeChatModel:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.invoke_count = 0

    def invoke(self, messages: list):
        self.invoke_count += 1
        rendered = "|".join(str(message) for message in messages)
        return SimpleNamespace(content=f"{self.prefix}:{rendered}")


def _make_llm(cache_path, prefix: str):
    wtb_llm = pytest.importorskip("wtb.infrastructure.llm")
    wtb_llm.reset_service_cache()

    config = WTBCacheConfig(
        enabled=True,
        cache_path=str(cache_path),
        api_key="cache-check",
        base_url="https://cache-check.invalid/v1",
        text_model="cache-check-text",
        embedding_model="cache-check-embedding",
    )
    llm = WTBCachedLLM(config=config, system_name="test", node_path="generation")
    fake = _FakeChatModel(prefix)
    llm.service.get_chat_model = lambda **_kwargs: fake
    return llm, fake, wtb_llm


def test_cached_llm_reuses_response_cache_across_instances(tmp_path):
    cache_path = tmp_path / "llm_response_cache.db"

    first, first_fake, wtb_llm = _make_llm(cache_path, "first")
    first_text = first.complete("system", "same prompt", temperature=0.0, max_tokens=32)
    first_meta = first.wtb_cache_metadata()

    second, second_fake, wtb_llm = _make_llm(cache_path, "second")
    second_text = second.complete("system", "same prompt", temperature=0.0, max_tokens=32)
    second_meta = second.wtb_cache_metadata()

    assert first_meta["last_cache_hit"] is False
    assert first_fake.invoke_count == 1
    assert second_meta["last_cache_hit"] is True
    assert second_fake.invoke_count == 0
    assert first_meta["last_cache_key"] == second_meta["last_cache_key"]
    assert first_text == second_text

    wtb_llm.reset_service_cache()


def test_common_generation_attaches_wtb_cache_metadata(tmp_path):
    cache_path = tmp_path / "llm_response_cache.db"
    context = [RetrievalResult(source_id="doc://1", content="Paris is in France.")]

    llm, fake, wtb_llm = _make_llm(cache_path, "reader")
    result = SimpleLLMGeneration(llm=llm).generate("Where is Paris?", context)

    assert fake.invoke_count == 1
    assert result.metadata["style"] == "llm-reader"
    assert result.metadata["wtb_cache"]["enabled"] is True
    assert result.metadata["wtb_cache"]["system"] == "test"
    assert result.metadata["wtb_cache"]["last_cache_hit"] is False

    wtb_llm.reset_service_cache()


def test_system_status_reports_all_swappable_targets():
    statuses = inspect_swappable_system_cache_support()
    assert {status.name for status in statuses} == {"longrag", "storm", "selfrag"}
    assert all(status.cache_surface for status in statuses)

