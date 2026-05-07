"""Layer 4: AG-UCT + Benchmark integration tests.

Validates the full chain: AG-UCT -> component_registry -> pipeline ->
benchmark adapter -> scoring, using both real and simulated data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dotenv import load_dotenv

_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    load_dotenv(_env)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AG-UCT"))

HOTPOTQA_DIR = Path("/data1/ragworkspace/dataset/all_data/hotpotqa")
ULTRADOMAIN_DIR = Path("/data1/ragworkspace/dataset/UltraDomain")
DATA_DIR = Path("/data1/ragworkspace/dataset")

skip_no_data = pytest.mark.skipif(
    not HOTPOTQA_DIR.exists(), reason="Real datasets not available"
)
skip_no_api_key = pytest.mark.skipif(
    not os.environ.get("LLM_API_KEY"), reason="LLM_API_KEY not set"
)


# ---------------------------------------------------------------------------
# Real dataset loading into UCT
# ---------------------------------------------------------------------------

@skip_no_data
class TestFrozenSamplesReal:

    def test_build_frozen_samples_real_hotpotqa(self):
        from uct_engine.examples.rag_pipeline_search import build_frozen_samples_real
        samples = build_frozen_samples_real(str(DATA_DIR), max_items=5)
        assert "hotpotqa" in samples
        items = samples["hotpotqa"]
        assert len(items) == 5
        for item in items:
            assert "question" in item
            assert "context_results" in item

    def test_build_frozen_samples_real_ultradomain(self):
        from uct_engine.examples.rag_pipeline_search import build_frozen_samples_real
        samples = build_frozen_samples_real(str(DATA_DIR), max_items=5, ud_domain="mix")
        assert "ultradomain" in samples
        items = samples["ultradomain"]
        assert len(items) == 5

    def test_frozen_samples_real_items_are_plain_dicts(self):
        from uct_engine.examples.rag_pipeline_search import (
            build_frozen_samples_real,
            _is_plain_dict,
        )
        samples = build_frozen_samples_real(str(DATA_DIR), max_items=3)
        for cid, items in samples.items():
            for item in items:
                assert _is_plain_dict(item), f"Item from {cid} should be a plain dict"


# ---------------------------------------------------------------------------
# evaluate_config_real
# ---------------------------------------------------------------------------

@skip_no_data
@skip_no_api_key
class TestEvaluateConfigReal:

    def test_evaluate_config_real_hotpotqa(self):
        from uct_engine.examples.rag_pipeline_search import (
            build_frozen_samples_real,
            evaluate_config_real,
        )
        samples = build_frozen_samples_real(str(DATA_DIR), max_items=2)
        choices = ("standard_passage", "identity", "bm25", "identity", "simple_llm")
        score = evaluate_config_real(
            choices, benchmark="hotpotqa", frozen_sample=samples.get("hotpotqa"),
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_evaluate_config_real_ultradomain(self):
        from uct_engine.examples.rag_pipeline_search import (
            build_frozen_samples_real,
            evaluate_config_real,
        )
        samples = build_frozen_samples_real(str(DATA_DIR), max_items=2)
        choices = ("standard_passage", "identity", "bm25", "identity", "simple_llm")
        score = evaluate_config_real(
            choices, benchmark="ultradomain", frozen_sample=samples.get("ultradomain"),
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Pipeline building
# ---------------------------------------------------------------------------

class TestBuildPipelineFromConfig:

    def test_build_pipeline_all_slot_values(self):
        """Every generation slot value produces a non-None component."""
        from uct_engine.examples.rag_pipeline_search import SLOT_OPTIONS

        generation_slots = SLOT_OPTIONS[4]
        for gen_name in generation_slots:
            choices = ("standard_passage", "identity", "bm25", "identity", gen_name)
            try:
                from rag_contracts.component_registry import build_pipeline_from_config
                components = build_pipeline_from_config(choices, "hotpotqa")
                gen = components.get("generation")
                assert gen is not None, f"Generation component is None for {gen_name}"
                assert hasattr(gen, "generate"), f"{gen_name} missing generate()"
            except ImportError:
                pass

    def test_build_pipeline_lightrag_constraint(self):
        """lightrag_hybrid requires kg_extraction chunking."""
        from uct_engine.examples.rag_pipeline_search import _check_constraints
        assert _check_constraints(("kg_extraction", "identity", "lightrag_hybrid", "identity", "simple_llm"))
        assert not _check_constraints(("standard_passage", "identity", "lightrag_hybrid", "identity", "simple_llm"))


# ---------------------------------------------------------------------------
# UCT search engine
# ---------------------------------------------------------------------------

class TestUCTSearch:

    def test_uct_search_simulated_3_iterations(self):
        """Simulated UCT search terminates and returns valid result."""
        from uct_engine.examples.rag_pipeline_search import (
            RAGPipelineEvaluator,
            RAGPipelineSearchState,
            CLUSTER_COST,
            CLUSTER_IDS,
        )
        from uct_engine import (
            ClusterDef,
            CostAwareUCTScorer,
            ReuseAwareCostModel,
            UCTSearchEngine,
        )

        evaluator = RAGPipelineEvaluator(use_real=False)
        scorer = CostAwareUCTScorer(lambda_t=0.05)
        clusters = [ClusterDef(c, weight=1.0, base_cost=CLUSTER_COST[c]) for c in CLUSTER_IDS]
        cost_model = ReuseAwareCostModel(clusters=clusters)
        engine = UCTSearchEngine(
            evaluator=evaluator, scorer=scorer, cost_model=cost_model,
            exploration_constant=1.4, random_seed=42,
        )
        root = RAGPipelineSearchState()
        result = engine.search(root, max_iterations=3)
        assert result.best_state.is_terminal()
        assert result.best_reward > 0
        assert result.iterations == 3

    @skip_no_data
    @skip_no_api_key
    def test_uct_search_real_3_iterations(self):
        """UCT search with real data (3 iterations)."""
        from uct_engine.examples.rag_pipeline_search import (
            RAGPipelineEvaluator,
            RAGPipelineSearchState,
            build_frozen_samples_real,
            CLUSTER_COST,
            CLUSTER_IDS,
        )
        from uct_engine import (
            ClusterDef,
            CostAwareUCTScorer,
            ReuseAwareCostModel,
            UCTSearchEngine,
        )

        frozen = build_frozen_samples_real(str(DATA_DIR), max_items=2)
        evaluator = RAGPipelineEvaluator(use_real=True, frozen_samples=frozen)
        scorer = CostAwareUCTScorer(lambda_t=0.05)
        clusters = [ClusterDef(c, weight=1.0, base_cost=CLUSTER_COST[c]) for c in CLUSTER_IDS]
        cost_model = ReuseAwareCostModel(clusters=clusters)
        engine = UCTSearchEngine(
            evaluator=evaluator, scorer=scorer, cost_model=cost_model,
            exploration_constant=1.4, random_seed=42,
        )
        root = RAGPipelineSearchState()
        result = engine.search(root, max_iterations=3)
        assert result.best_state.is_terminal()
        assert result.iterations == 3
