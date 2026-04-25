"""Integration tests for the sampling engine.

Uses synthetic BenchmarkItems that mirror real UltraDomain / FreshWiki
field shapes so the full pipeline can be validated without the actual
HuggingFace datasets.  A deterministic ``eval_fn`` returns rewards
derived from item metadata so estimator behaviour is verifiable.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from bsamp.sampling.engine import SamplingEngine, SamplingResult
from bsamp.sampling.stratification import StratificationConfig
from bsamp.sampling.types import (
    BenchmarkItem,
    Estimate,
    ItemRealization,
    SamplingState,
    StratumStats,
)


# ---------------------------------------------------------------------------
# Synthetic data builders (mirror real schemas)
# ---------------------------------------------------------------------------

_UD_DOMAINS = ["physics", "cs", "mathematics"]
_FW_CLASSES = ["Stub", "Start", "C", "B", "GA", "FA"]


def _make_ultradomain_items(n_per_domain: int = 40, seed: int = 7) -> list[BenchmarkItem]:
    """Create items shaped like real UltraDomain rows."""
    rng = random.Random(seed)
    items: list[BenchmarkItem] = []
    for domain in _UD_DOMAINS:
        for i in range(n_per_domain):
            length = rng.randint(5000, 120000)
            items.append(BenchmarkItem(
                item_id=f"{domain}::{i}",
                benchmark="ultradomain",
                stratum="",
                payload={
                    "query": f"Question about {domain} #{i}",
                    "context": f"Context text for {domain} item {i} " * 20,
                },
                target={
                    "answers": [f"Answer for {domain} item {i}"],
                    "answer": f"Answer for {domain} item {i}",
                    "label": domain,
                },
                metadata={
                    "domain": domain,
                    "length": length,
                    "context_id": f"ctx_{domain}_{i}",
                    "title": f"Title {domain} {i}",
                    "authors": f"Author {i}",
                },
            ))
    return items


def _make_freshwiki_items(n: int = 30, seed: int = 7) -> list[BenchmarkItem]:
    """Create items shaped like real FreshWiki rows."""
    rng = random.Random(seed)
    quality_map = {"Stub": "low", "Start": "low", "C": "mid", "B": "mid", "GA": "high", "FA": "high"}
    items: list[BenchmarkItem] = []
    for i in range(n):
        pc = rng.choice(_FW_CLASSES)
        text_length = rng.randint(2000, 80000)
        n_sections = rng.randint(3, 25)
        items.append(BenchmarkItem(
            item_id=f"freshwiki::Topic_{i}",
            benchmark="freshwiki",
            stratum="",
            payload={
                "topic": f"Topic {i}",
                "text": f"Full article text for topic {i} " * 50,
                "content": [{"section_title": f"s{j}", "section_content": []} for j in range(n_sections)],
            },
            target={
                "summary": f"Summary of topic {i}",
                "url": f"https://en.wikipedia.org/wiki/Topic_{i}",
            },
            metadata={
                "predicted_class": pc,
                "quality_bucket": quality_map[pc],
                "predicted_scores": {c: rng.random() for c in _FW_CLASSES},
                "text_length": text_length,
                "n_sections": n_sections,
                "title": f"Topic_{i}",
            },
        ))
    return items


def _deterministic_eval(config: dict[str, Any], item_id: str) -> float:
    """Stable reward: hash the item_id to a float in [0, 1]."""
    h = hash(item_id) % 10000
    return h / 10000.0


# ---------------------------------------------------------------------------
# Stub adapter that wraps pre-built items (no filesystem needed)
# ---------------------------------------------------------------------------

class _SyntheticAdapter:
    """Minimal adapter for testing -- holds items in memory."""

    def __init__(self, items: list[BenchmarkItem], adapter_name: str = "ultradomain") -> None:
        self._items = items
        self._name = adapter_name

    @property
    def name(self) -> str:
        return self._name

    def load_items(self) -> list[BenchmarkItem]:
        return list(self._items)

    def population_size(self) -> int:
        return len(self._items)

    def available_strata_keys(self) -> list[str]:
        if self._name == "freshwiki":
            return ["predicted_class", "quality_bucket", "text_length"]
        return ["domain", "length"]


# ===================================================================
# Test 1: UltraDomain stratified proportional sampling
# ===================================================================

class TestProportionalSampling:
    def test_basic_proportional(self):
        items = _make_ultradomain_items(n_per_domain=40)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=30,
            seed=42,
        )
        result = engine.run()

        assert isinstance(result, SamplingResult)
        assert len(result.items) == 30
        assert len(result.realization.realized_items) == 30
        assert sum(result.realization.allocation) == 30
        assert result.estimate is None  # no eval_fn provided

        domains_hit = {item.metadata["domain"] for item in result.items}
        assert len(domains_hit) >= 2, "Should sample from multiple domains"

    def test_proportional_with_eval(self):
        items = _make_ultradomain_items(n_per_domain=40)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=30,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        result = engine.run()

        assert result.estimate is not None
        assert isinstance(result.estimate, Estimate)
        assert 0.0 <= result.estimate.mean <= 1.0
        assert result.estimate.n_evaluated == 30
        assert result.estimate.ci_lower <= result.estimate.mean <= result.estimate.ci_upper


# ===================================================================
# Test 2: UltraDomain Neyman with pilot phase
# ===================================================================

class TestNeymanSampling:
    def test_neyman_two_phase(self):
        items = _make_ultradomain_items(n_per_domain=40)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="neyman",
            budget=50,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        result = engine.run()

        assert isinstance(result, SamplingResult)
        assert result.estimate is not None
        assert result.state.sampler_type == "neyman"
        assert len(result.state.realizations) == 2, "pilot + main = 2 realizations"
        assert result.state.budget_used <= 50

    def test_neyman_small_budget_degrades_to_pilot_only(self):
        items = _make_ultradomain_items(n_per_domain=5)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="neyman",
            budget=15,
            seed=42,
        )
        result = engine.run()

        assert len(result.items) <= 15
        assert len(result.state.realizations) >= 1


# ===================================================================
# Test 3: UltraDomain MH sampling
# ===================================================================

class TestMHSampling:
    def test_mh_basic(self):
        items = _make_ultradomain_items(n_per_domain=40)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="mh",
            budget=40,
            seed=42,
            eval_fn=_deterministic_eval,
            mh_iterations=10,
            mh_temperature=1.0,
            mh_anneal_rate=0.95,
        )
        result = engine.run()

        assert isinstance(result, SamplingResult)
        assert result.state.sampler_type == "mh"
        assert len(result.state.realizations) > 1, "pilot + MH steps"
        assert result.state.budget_used <= 40

        sampler_state = result.state.sampler_state
        assert sampler_state["step_count"] > 0
        assert len(sampler_state["energy_trace"]) > 0

    def test_mh_allocation_labels_persist(self):
        items = _make_ultradomain_items(n_per_domain=40)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="mh",
            budget=30,
            seed=99,
            mh_iterations=5,
        )
        result = engine.run()

        sampler_state = result.state.sampler_state
        assert "sorted_labels" in sampler_state
        assert len(sampler_state["sorted_labels"]) > 0
        assert sampler_state["sorted_labels"] == sorted(sampler_state["sorted_labels"])


# ===================================================================
# Test 4: FreshWiki stratified by quality_bucket
# ===================================================================

class TestFreshWikiSampling:
    def test_freshwiki_quality_strata(self):
        items = _make_freshwiki_items(n=30)
        adapter = _SyntheticAdapter(items, "freshwiki")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=15,
            seed=42,
        )
        result = engine.run()

        assert len(result.items) == 15
        quality_buckets = {item.metadata["quality_bucket"] for item in result.items}
        assert len(quality_buckets) >= 1

    def test_freshwiki_with_eval(self):
        items = _make_freshwiki_items(n=30)
        adapter = _SyntheticAdapter(items, "freshwiki")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=15,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        result = engine.run()

        assert result.estimate is not None
        assert result.estimate.n_evaluated == 15


# ===================================================================
# Test 5: State serialization roundtrip
# ===================================================================

class TestStateRoundtrip:
    def test_sampling_state_json_roundtrip(self):
        items = _make_ultradomain_items(n_per_domain=20)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=20,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        result = engine.run()

        json_str = result.state.to_json()
        restored = SamplingState.from_json(json_str)

        assert restored.config_id == result.state.config_id
        assert restored.benchmark == result.state.benchmark
        assert restored.budget_total == result.state.budget_total
        assert restored.budget_used == result.state.budget_used
        assert len(restored.history) == len(result.state.history)
        assert len(restored.realizations) == len(result.state.realizations)
        for orig, rest in zip(result.state.history, restored.history):
            assert orig.item_id == rest.item_id
            assert abs(orig.reward - rest.reward) < 1e-12

    def test_sampling_result_json(self):
        items = _make_ultradomain_items(n_per_domain=10)
        adapter = _SyntheticAdapter(items, "ultradomain")

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=10,
            seed=42,
        )
        result = engine.run()

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "items" in parsed
        assert "realization" in parsed
        assert "strata_summary" in parsed
        assert "state" in parsed


# ===================================================================
# Test 6: Multi-benchmark sequential sampling
# ===================================================================

class TestMultiBenchmark:
    def test_sequential_ud_then_fw(self):
        ud_items = _make_ultradomain_items(n_per_domain=30)
        fw_items = _make_freshwiki_items(n=20)

        ud_adapter = _SyntheticAdapter(ud_items, "ultradomain")
        fw_adapter = _SyntheticAdapter(fw_items, "freshwiki")

        ud_engine = SamplingEngine(
            adapter=ud_adapter,
            method="neyman",
            budget=25,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        ud_result = ud_engine.run()

        fw_engine = SamplingEngine(
            adapter=fw_adapter,
            method="proportional",
            budget=10,
            seed=42,
            eval_fn=_deterministic_eval,
        )
        fw_result = fw_engine.run()

        assert ud_result.state.benchmark == "ultradomain"
        assert fw_result.state.benchmark == "freshwiki"
        assert ud_result.estimate is not None
        assert fw_result.estimate is not None

        ud_ids = {item.item_id for item in ud_result.items}
        fw_ids = {item.item_id for item in fw_result.items}
        assert ud_ids.isdisjoint(fw_ids), "No overlap between benchmarks"

    def test_method_aliases(self):
        items = _make_ultradomain_items(n_per_domain=10)
        adapter = _SyntheticAdapter(items, "ultradomain")

        for alias in ["proportional", "prop", "neyman", "optimal", "mh", "metropolis"]:
            engine = SamplingEngine(
                adapter=adapter,
                method=alias,
                budget=10,
                seed=42,
            )
            result = engine.run()
            assert len(result.items) <= 10
