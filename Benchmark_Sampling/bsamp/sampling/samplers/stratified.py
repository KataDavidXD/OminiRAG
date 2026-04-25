"""Deterministic stratified sampler with proportional or Neyman allocation.

Allocation strategies:
  - proportional: n_h proportional to N_h / N (sample sizes mirror population shares)
  - neyman:       n_h proportional to N_h * S_h (Neyman optimal -- high-variance strata
                  receive more samples)

Neyman parameters:
  - N_h: population size of stratum h
  - S_h: standard deviation of stratum h (from pilot-phase variance estimates)
  - B:   total budget
  If a stratum has sample_size < 2 (no variance estimate), falls back to proportional.

After allocation, items are drawn within each stratum via ``rng.sample()`` (Layer 2: Realisation).
"""

from __future__ import annotations

import math
import random
from typing import Any

from bsamp.sampling.samplers.base import BaseSampler
from bsamp.sampling.types import BenchmarkItem, ItemRealization, StratumStats


class StratifiedSampler(BaseSampler):
    """Deterministic stratified sampler with proportional or Neyman allocation."""

    def __init__(
        self,
        allocation: str = "proportional",
        seed: int | None = None,
    ) -> None:
        self._allocation_method = allocation
        self._rng = random.Random(seed)
        self._step = 0

    # ------------------------------------------------------------------
    # BaseSampler interface
    # ------------------------------------------------------------------

    def select(
        self,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        budget: int,
    ) -> ItemRealization:
        ordered_labels = sorted(strata.keys())

        if self._allocation_method == "neyman":
            alloc = self._neyman_allocation(ordered_labels, strata_stats, budget)
        else:
            alloc = self._proportional_allocation(ordered_labels, strata_stats, budget)

        rng_state_before = self._rng.getstate()

        realized: list[str] = []
        for label, n_h in zip(ordered_labels, alloc):
            pool = strata[label]
            k = min(n_h, len(pool))
            if k <= 0:
                continue
            drawn = self._rng.sample(pool, k)
            realized.extend(item.item_id for item in drawn)

        self._step += 1

        return ItemRealization(
            allocation=alloc,
            realized_items=realized,
            rng_state_before=rng_state_before,
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "allocation_method": self._allocation_method,
            "rng_state": self._rng.getstate(),
            "step": self._step,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self._allocation_method = state["allocation_method"]
        self._rng.setstate(state["rng_state"])
        self._step = state["step"]

    # ------------------------------------------------------------------
    # Allocation algorithms
    # ------------------------------------------------------------------

    def _proportional_allocation(
        self,
        labels: list[str],
        stats: dict[str, StratumStats],
        budget: int,
    ) -> list[int]:
        """n_h proportional to N_h / N, with at least 1 per stratum."""
        N = sum(stats[l].population_size for l in labels)
        if N == 0:
            return [0] * len(labels)

        raw = [budget * stats[l].population_size / N for l in labels]
        alloc = [max(int(r), 1) for r in raw]
        alloc = self._adjust_to_budget(alloc, budget, labels, stats)
        return alloc

    def _neyman_allocation(
        self,
        labels: list[str],
        stats: dict[str, StratumStats],
        budget: int,
    ) -> list[int]:
        """n_h proportional to N_h * s_h (Neyman optimal).

        Falls back to proportional when variance estimates are missing.
        """
        weights: list[float] = []
        for l in labels:
            s = stats[l]
            sd = math.sqrt(s.sample_variance) if s.sample_size >= 2 else 0.0
            weights.append(s.population_size * sd)

        total_w = sum(weights)
        if total_w < 1e-12:
            return self._proportional_allocation(labels, stats, budget)

        raw = [budget * w / total_w for w in weights]
        alloc = [max(int(r), 1) for r in raw]
        alloc = self._adjust_to_budget(alloc, budget, labels, stats)
        return alloc

    @staticmethod
    def _adjust_to_budget(
        alloc: list[int],
        budget: int,
        labels: list[str],
        stats: dict[str, StratumStats],
    ) -> list[int]:
        """Greedily add/remove units so sum(alloc) == budget."""
        current = sum(alloc)
        while current < budget:
            best_idx = -1
            best_score = -1.0
            for i, l in enumerate(labels):
                if alloc[i] < stats[l].population_size:
                    score = stats[l].population_size
                    if score > best_score:
                        best_score = score
                        best_idx = i
            if best_idx < 0:
                break
            alloc[best_idx] += 1
            current += 1

        while current > budget:
            best_idx = -1
            best_score = float("inf")
            for i, l in enumerate(labels):
                if alloc[i] > 1:
                    score = stats[l].population_size
                    if score < best_score:
                        best_score = score
                        best_idx = i
            if best_idx < 0:
                break
            alloc[best_idx] -= 1
            current -= 1

        for i, l in enumerate(labels):
            alloc[i] = min(alloc[i], stats[l].population_size)

        return alloc
