"""
RAGCacheAwareEvaluator -- bridges AG-UCT's search to WTB physical execution.

Implements the ``uct_engine.Evaluator`` ABC.  For each terminal
configuration it:

1. Computes the marginal cost using the ``ReuseLedger`` (real Path_t).
2. Executes the pipeline via ``WTBTestBench`` with cache-aware reuse.
3. Scores results through OminiRAG's benchmark adapters.
4. Reports ``materialized_keys`` back so the engine can update Path_t.

When ``use_real=False`` (default), rewards come from the existing
simulated reward table -- only the *cost accounting* is physical.
When ``use_real=True``, actual pipeline execution through the bench
produces rewards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Hashable, List, Optional

from .config_types import RAGConfig, BenchmarkQuestion, NODE_ORDER

logger = logging.getLogger(__name__)

# Per-node cost weights (normalised so a full 4-node execution costs 4.0)
PER_NODE_COST = {
    1: 0.0,   # frame selection -- no compute
    2: 1.0,   # query processing
    3: 1.0,   # retrieval
    4: 1.0,   # reranking
    5: 1.0,   # generation
}


def state_to_rag_config(state: Any) -> RAGConfig:
    """Convert an AG-UCT ``SearchState`` with a 5-tuple key to ``RAGConfig``."""
    key = state.state_key()
    return RAGConfig.from_tuple(key)


class RAGCacheAwareEvaluator:
    """Evaluator that uses the ReuseLedger for physical cost accounting.

    Parameters
    ----------
    ledger
        ``ReuseLedger`` instance (shared across the search session).
    bench
        ``WTBTestBench`` instance for real execution (only used when
        ``use_real=True``).
    cluster_ids
        List of benchmark cluster IDs to evaluate on.
    frozen_samples
        ``{cluster_id: list[BenchmarkQuestion]}`` pre-drawn via
        ``SamplingEngine``.  Required for real evaluation.
    cluster_costs
        ``{cluster_id: float}`` base cost per cluster.
    use_real
        When True, execute pipelines and measure real reward.
    reward_table
        Optional simulated reward lookup for ``use_real=False``.
    default_reward
        Fallback simulated reward.
    """

    def __init__(
        self,
        ledger: Any,
        bench: Any = None,
        cluster_ids: Optional[List[str]] = None,
        frozen_samples: Optional[Dict[str, List[BenchmarkQuestion]]] = None,
        cluster_costs: Optional[Dict[str, float]] = None,
        use_real: bool = False,
        reward_table: Optional[Dict[tuple, float]] = None,
        default_reward: float = 0.55,
    ) -> None:
        from .reuse_ledger import ReuseLedger

        self.ledger: ReuseLedger = ledger
        self.bench = bench
        self.cluster_ids = cluster_ids or ["hotpotqa", "ultradomain", "alce"]
        self.frozen_samples = frozen_samples or {}
        self.cluster_costs = cluster_costs or {
            "hotpotqa": 1.0, "ultradomain": 1.2, "alce": 1.5,
        }
        self.use_real = use_real
        self.reward_table = reward_table or {}
        self.default_reward = default_reward

    # ------------------------------------------------------------------
    # AG-UCT Evaluator interface
    # ------------------------------------------------------------------

    def evaluate(self, state: Any, context: Any) -> Any:
        """Evaluate a terminal configuration across all benchmark clusters.

        Returns an ``EvaluationResult`` (imported from ``uct_engine``).
        """
        from uct_engine import BenchmarkClusterResult, EvaluationResult

        assert state.is_terminal(), "evaluator expects a terminal state"
        config = state_to_rag_config(state)

        cluster_results: list[BenchmarkClusterResult] = []
        total_reward = 0.0
        total_cost = 0.0
        weight = 1.0 / len(self.cluster_ids)

        for cid in self.cluster_ids:
            questions = self.frozen_samples.get(cid, [])

            # --- Cost accounting against physical ledger ---
            new_keys: list[Hashable] = []
            cluster_cost = 0.0

            if questions:
                for q in questions:
                    for depth in range(1, 6):
                        prefix = config.prefix(depth)
                        reuse_key: Hashable = (prefix, q.question_id)
                        if reuse_key not in context.materialized_keys:
                            new_keys.append(reuse_key)
                            cluster_cost += (
                                PER_NODE_COST.get(depth, 1.0)
                                * self.cluster_costs.get(cid, 1.0)
                                / len(questions)
                            )
            else:
                for depth in range(1, 6):
                    prefix = config.prefix(depth)
                    prefix_key: Hashable = (prefix, cid)
                    if prefix_key not in context.materialized_keys:
                        new_keys.append(prefix_key)
                        cluster_cost += (
                            self.cluster_costs.get(cid, 1.0) / 5.0
                        )

            # --- Reward ---
            if self.use_real and questions and self.bench is not None:
                reward = self._execute_and_measure(config, questions, cid)
            else:
                reward = self._simulated_reward(config, cid)

            cluster_results.append(BenchmarkClusterResult(
                cluster_id=cid,
                reward=reward,
                cost=cluster_cost,
                materialized_keys=new_keys,
                metadata={
                    "config": config.config_key(),
                    "dataset": cid,
                    "real_eval": self.use_real,
                },
            ))
            total_reward += weight * reward
            total_cost += cluster_cost

        return EvaluationResult(
            reward=total_reward,
            total_cost=total_cost,
            cluster_results=cluster_results,
            metadata={
                "config_pretty": config.config_key(),
                "real_eval": self.use_real,
            },
        )

    # ------------------------------------------------------------------
    # Simulated reward (fallback)
    # ------------------------------------------------------------------

    def _simulated_reward(self, config: RAGConfig, cluster_id: str) -> float:
        choices_4 = (config.query, config.retrieval, config.reranking, config.generation)
        if choices_4 in self.reward_table:
            return self.reward_table[choices_4]
        full = config.slots()
        if full in self.reward_table:
            return self.reward_table[full]
        return self.default_reward

    # ------------------------------------------------------------------
    # Real execution via WTB bench
    # ------------------------------------------------------------------

    def _execute_and_measure(
        self,
        config: RAGConfig,
        questions: List[BenchmarkQuestion],
        cluster_id: str,
    ) -> float:
        """Execute the config on questions using the WTB bench, return reward."""
        try:
            from .batch_runner import run_batch_with_reuse
            results = run_batch_with_reuse(
                configs=[config],
                questions=questions,
                bench=self.bench,
                ledger=self.ledger,
            )
            return self._score_results(results, config, questions, cluster_id)
        except Exception as exc:
            logger.warning("Real execution failed for %s on %s: %s",
                           config.config_key(), cluster_id, exc)
            return self.default_reward

    def _score_results(
        self,
        results: List[Dict[str, Any]],
        config: RAGConfig,
        questions: List[BenchmarkQuestion],
        cluster_id: str,
    ) -> float:
        """Score execution results through the appropriate benchmark adapter."""
        try:
            if cluster_id == "hotpotqa":
                from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
                adapter = HotpotQABenchmarkAdapter()
                eval_items = []
                for q, r in zip(questions, results):
                    eval_items.append({
                        "question": q.question,
                        "answer": q.target.get("answer", ""),
                        "query_id": q.question_id,
                        "generation_result": r.get("generation_result"),
                    })
                if hasattr(adapter, "score_generation_results"):
                    result = adapter.score_generation_results(eval_items)
                    return result.avg_f1 / 100.0
            elif cluster_id == "alce":
                from benchmark.alce_adapter import ALCEBenchmarkAdapter
                adapter = ALCEBenchmarkAdapter()
                eval_items = []
                for q, r in zip(questions, results):
                    eval_items.append({
                        "question": q.question,
                        "answer": q.target.get("answer", ""),
                        "docs": q.payload.get("docs", []),
                        "generation_result": r.get("generation_result"),
                    })
                if hasattr(adapter, "score_generation_results"):
                    result = adapter.score_generation_results(eval_items)
                    return result.avg_f1 / 100.0
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", cluster_id, exc)

        return self.default_reward
