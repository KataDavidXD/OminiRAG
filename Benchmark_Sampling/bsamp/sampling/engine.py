"""Unified facade for stratified / Neyman / MH sampling.

Entry point for the entire sampling subsystem. Orchestrates data loading,
stratification, sampling, and estimation in a single ``engine.run()`` call.

Supported methods:
  - proportional: sample sizes proportional to stratum population (default)
  - neyman:       two-phase Neyman optimal allocation (pilot -> variance estimate -> optimal)
  - mh:           Metropolis-Hastings adaptive allocation (iterative variance minimisation)

Standardised eval interface:
  ``eval_fn: Callable[[dict, str], float]``
  Any function with signature ``(rag_config, item_id) -> reward`` can be plugged in.
  When omitted, only sampled items are returned without computing estimates.

Usage::

    from bsamp.sampling.engine import SamplingEngine
    from bsamp.sampling.adapters import UltraDomainAdapter

    engine = SamplingEngine(
        adapter=UltraDomainAdapter(root_dir="..."),
        method="neyman",
        budget=200,
        seed=42,
    )
    result = engine.run()

    result.items           # list[BenchmarkItem]  -- sampled items
    result.realization     # ItemRealization       -- allocation vector + realised item_ids
    result.strata_summary  # dict[str, int]        -- stratum name -> population count
    result.state           # SamplingState          -- full serialisable state
    result.estimate        # Estimate | None        -- point estimate + confidence interval
    result.to_json()       # str                    -- JSON output
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from bsamp.sampling.adapters.base import BenchmarkAdapter
from bsamp.sampling.budget import BudgetController
from bsamp.sampling.estimator import SequentialEstimator, StoppingConfig
from bsamp.sampling.samplers.base import BaseSampler
from bsamp.sampling.samplers.mh import MetropolisHastingsSampler
from bsamp.sampling.samplers.stratified import StratifiedSampler
from bsamp.sampling.stratification import (
    StratificationConfig,
    build_freshwiki_config,
    build_hotpotqa_config,
    build_alce_config,
    build_ultradomain_config,
    stratify,
)
from bsamp.sampling.types import (
    BenchmarkItem,
    EvalRecord,
    Estimate,
    ItemRealization,
    SamplingState,
    StratumStats,
)


@dataclass
class SamplingResult:
    """Immutable output of a single ``SamplingEngine.run()`` call."""

    items: list[BenchmarkItem]
    realization: ItemRealization
    strata_summary: dict[str, int]
    state: SamplingState
    estimate: Estimate | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "realization": self.realization.to_dict(),
            "strata_summary": self.strata_summary,
            "state": self.state.to_dict(),
            "estimate": self.estimate.to_dict() if self.estimate else None,
        }

    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii)

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_json(), encoding="utf-8")


_METHOD_ALIASES = {
    "proportional": "proportional",
    "prop": "proportional",
    "neyman": "neyman",
    "optimal": "neyman",
    "mh": "mh",
    "metropolis": "mh",
    "metropolis-hastings": "mh",
}


class SamplingEngine:
    """One-call facade for benchmark sampling.

    Parameters
    ----------
    adapter:
        A loaded ``BenchmarkAdapter`` (FreshWiki or UltraDomain).
    method:
        ``"proportional"`` | ``"neyman"`` | ``"mh"``
    budget:
        Maximum number of items to draw.
    seed:
        RNG seed for reproducibility.
    stratification_config:
        Explicit config. Auto-detected from ``adapter.name`` if *None*.
    eval_fn:
        Optional ``(config_dict, item_id) -> float`` for live estimation.
        When supplied, the engine feeds rewards into a
        ``SequentialEstimator`` and returns an ``Estimate``.
    rag_config:
        Arbitrary dict identifying the RAG config under test.
        Only used when *eval_fn* is provided.
    mh_iterations:
        Number of MH steps (only for ``method="mh"``).
    mh_temperature:
        Initial temperature (only for ``method="mh"``).
    mh_anneal_rate:
        Temperature annealing rate (only for ``method="mh"``).
    stopping:
        Early-stopping configuration. Defaults to sensible values.
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        method: str = "proportional",
        budget: int = 100,
        seed: int = 42,
        stratification_config: StratificationConfig | None = None,
        eval_fn: Callable[[dict[str, Any], str], float] | None = None,
        rag_config: dict[str, Any] | None = None,
        mh_iterations: int = 50,
        mh_temperature: float = 1.0,
        mh_anneal_rate: float = 0.99,
        stopping: StoppingConfig | None = None,
    ) -> None:
        self._adapter = adapter
        self._method = _METHOD_ALIASES.get(method.lower(), method.lower())
        self._budget = budget
        self._seed = seed
        self._eval_fn = eval_fn
        self._rag_config = rag_config or {}
        self._mh_iterations = mh_iterations
        self._mh_temperature = mh_temperature
        self._mh_anneal_rate = mh_anneal_rate
        self._stopping = stopping or StoppingConfig()

        if stratification_config is not None:
            self._strat_config = stratification_config
        elif adapter.name == "freshwiki":
            self._strat_config = build_freshwiki_config()
        elif adapter.name == "hotpotqa":
            self._strat_config = build_hotpotqa_config()
        elif adapter.name == "alce":
            self._strat_config = build_alce_config()
        else:
            self._strat_config = build_ultradomain_config()

    def run(self) -> SamplingResult:
        """Execute the full sampling pipeline and return results."""
        population = self._adapter.load_items()
        strata, strata_stats = stratify(population, self._strat_config)

        item_index = {item.item_id: item for items in strata.values() for item in items}

        sampler = self._build_sampler()

        clamped_budget = min(self._budget, len(population))

        if self._method == "mh":
            return self._run_mh(sampler, strata, strata_stats, item_index, clamped_budget)

        if self._method == "neyman":
            return self._run_neyman(sampler, strata, strata_stats, item_index, clamped_budget)

        return self._run_single_pass(sampler, strata, strata_stats, item_index, clamped_budget)

    # ------------------------------------------------------------------
    # Internal: sampler factory
    # ------------------------------------------------------------------

    def _build_sampler(self) -> BaseSampler:
        if self._method == "mh":
            return MetropolisHastingsSampler(
                rng_seed=self._seed,
                initial_temperature=self._mh_temperature,
                anneal_rate=self._mh_anneal_rate,
            )
        allocation = "neyman" if self._method == "neyman" else "proportional"
        return StratifiedSampler(allocation=allocation, seed=self._seed)

    # ------------------------------------------------------------------
    # Single-pass (proportional)
    # ------------------------------------------------------------------

    def _run_single_pass(
        self,
        sampler: BaseSampler,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        item_index: dict[str, BenchmarkItem],
        budget: int,
    ) -> SamplingResult:
        realization = sampler.select(strata, strata_stats, budget)
        items = [item_index[iid] for iid in realization.realized_items if iid in item_index]
        estimate, history = self._evaluate_if_possible(items, strata_stats, step=0)

        state = self._build_state(
            sampler, strata_stats, history, [realization],
            budget_used=len(items),
        )

        return SamplingResult(
            items=items,
            realization=realization,
            strata_summary={k: len(v) for k, v in strata.items()},
            state=state,
            estimate=estimate,
        )

    # ------------------------------------------------------------------
    # Neyman (pilot + optimal)
    # ------------------------------------------------------------------

    def _run_neyman(
        self,
        sampler: BaseSampler,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        item_index: dict[str, BenchmarkItem],
        budget: int,
    ) -> SamplingResult:
        pilot_budget = max(2 * len(strata), budget // 5)
        pilot_budget = min(pilot_budget, budget)

        pilot_sampler = StratifiedSampler(allocation="proportional", seed=self._seed)
        pilot_realization = pilot_sampler.select(strata, strata_stats, pilot_budget)
        pilot_items = [item_index[iid] for iid in pilot_realization.realized_items if iid in item_index]

        pilot_estimate, pilot_history = self._evaluate_if_possible(
            pilot_items, strata_stats, step=0,
        )

        remaining = budget - len(pilot_items)
        all_history = list(pilot_history)
        realizations = [pilot_realization]

        if remaining > 0:
            main_realization = sampler.select(strata, strata_stats, remaining)
            main_items = [item_index[iid] for iid in main_realization.realized_items if iid in item_index]
            main_estimate, main_history = self._evaluate_if_possible(
                main_items, strata_stats, step=1,
            )
            all_items = pilot_items + main_items
            all_history.extend(main_history)
            realizations.append(main_realization)
            final_estimate = main_estimate or pilot_estimate
            final_realization = main_realization
        else:
            all_items = pilot_items
            final_estimate = pilot_estimate
            final_realization = pilot_realization

        state = self._build_state(
            sampler, strata_stats, all_history, realizations,
            budget_used=len(all_items),
        )

        return SamplingResult(
            items=all_items,
            realization=final_realization,
            strata_summary={k: len(v) for k, v in strata.items()},
            state=state,
            estimate=final_estimate,
        )

    # ------------------------------------------------------------------
    # MH (iterative)
    # ------------------------------------------------------------------

    def _run_mh(
        self,
        sampler: BaseSampler,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        item_index: dict[str, BenchmarkItem],
        budget: int,
    ) -> SamplingResult:
        # Phase 1: pilot with proportional allocation for variance estimates
        pilot_budget = max(2 * len(strata), budget // 5)
        pilot_budget = min(pilot_budget, budget)
        pilot_sampler = StratifiedSampler(allocation="proportional", seed=self._seed)
        pilot_real = pilot_sampler.select(strata, strata_stats, pilot_budget)
        pilot_items = [item_index[iid] for iid in pilot_real.realized_items if iid in item_index]
        _, pilot_history = self._evaluate_if_possible(pilot_items, strata_stats, step=0)

        remaining = budget - len(pilot_items)

        if remaining <= 0:
            estimator = SequentialEstimator(strata_stats, confidence=self._stopping.confidence)
            for rec in pilot_history:
                estimator.update(rec)
            state = self._build_state(
                sampler, strata_stats, pilot_history, [pilot_real],
                budget_used=len(pilot_items),
            )
            return SamplingResult(
                items=pilot_items,
                realization=pilot_real,
                strata_summary={k: len(v) for k, v in strata.items()},
                state=state,
                estimate=estimator.estimate() if self._eval_fn else None,
            )

        # Phase 2: MH exploration -- iterate over full remaining budget
        # Each select() does one propose/accept step; drawn items are discarded
        # during exploration since we only care about the converged allocation.
        for _ in range(max(0, self._mh_iterations - 1)):
            sampler.select(strata, strata_stats, remaining)

        # Phase 3: final draw with converged allocation
        final_real = sampler.select(strata, strata_stats, remaining)
        final_items = [item_index[iid] for iid in final_real.realized_items if iid in item_index]
        _, final_history = self._evaluate_if_possible(final_items, strata_stats, step=1)

        all_items = pilot_items + final_items
        all_history = list(pilot_history) + list(final_history)

        estimator = SequentialEstimator(strata_stats, confidence=self._stopping.confidence)
        for rec in all_history:
            estimator.update(rec)

        state = self._build_state(
            sampler, strata_stats, all_history, [pilot_real, final_real],
            budget_used=len(all_items),
        )

        return SamplingResult(
            items=all_items,
            realization=final_real,
            strata_summary={k: len(v) for k, v in strata.items()},
            state=state,
            estimate=estimator.estimate() if self._eval_fn else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_if_possible(
        self,
        items: list[BenchmarkItem],
        strata_stats: dict[str, StratumStats],
        step: int,
    ) -> tuple[Estimate | None, list[EvalRecord]]:
        """Run eval_fn on items if available; update strata_stats in-place."""
        if self._eval_fn is None or not items:
            return None, []

        records: list[EvalRecord] = []
        for item in items:
            reward = self._eval_fn(self._rag_config, item.item_id)
            rec = EvalRecord(
                item_id=item.item_id,
                stratum=item.stratum,
                reward=reward,
                step=step,
                cached=False,
                wall_time_ms=0.0,
            )
            records.append(rec)
            stats = strata_stats.get(item.stratum)
            if stats is not None:
                stats.update(reward)

        estimator = SequentialEstimator(strata_stats, confidence=self._stopping.confidence)
        return estimator.estimate(), records

    def _build_state(
        self,
        sampler: BaseSampler,
        strata_stats: dict[str, StratumStats],
        history: list[EvalRecord],
        realizations: list[ItemRealization],
        budget_used: int,
    ) -> SamplingState:
        rng = random.Random(self._seed)
        return SamplingState(
            config_id=json.dumps(self._rag_config, sort_keys=True) if self._rag_config else "",
            benchmark=self._adapter.name,
            sampler_type=self._method,
            budget_total=self._budget,
            budget_used=budget_used,
            strata_stats=dict(strata_stats),
            sampler_state=sampler.get_state(),
            rng_state=rng.getstate(),
            history=history,
            realizations=realizations,
        )
