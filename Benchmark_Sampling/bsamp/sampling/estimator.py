"""Sequential stratified estimator with early stopping.

Implements the stratified mean estimator:
  F_hat = sum_h (N_h / N) * mean_h

With variance:
  Var(F_hat) = sum_h (N_h / N)^2 * (s_h^2 / n_h) * (1 - n_h / N_h)

The finite-population correction (1 - n_h / N_h) matters for small populations
like FreshWiki where a large fraction of items may be sampled.

Early stopping rules:
  1. Budget exhausted
  2. CI width below absolute threshold
  3. Relative precision (SE / |mean|) below threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from bsamp.sampling.types import Estimate, EvalRecord, StratumStats


_Z_TABLE = {
    0.90: 1.6449,
    0.95: 1.9600,
    0.99: 2.5758,
}


@dataclass
class StoppingConfig:
    ci_threshold: float = 0.05
    relative_precision: float = 0.10
    confidence: float = 0.95


class SequentialEstimator:
    """Stratified mean estimator with running CI and early-stopping logic."""

    def __init__(
        self,
        strata_stats: dict[str, StratumStats],
        confidence: float = 0.95,
    ) -> None:
        self._strata = strata_stats
        self._confidence = confidence
        self._n_total = sum(s.population_size for s in strata_stats.values())
        self._z = _Z_TABLE.get(confidence, 1.96)

    @property
    def n_total(self) -> int:
        return self._n_total

    @property
    def n_evaluated(self) -> int:
        return sum(s.sample_size for s in self._strata.values())

    def update(self, record: EvalRecord) -> None:
        stats = self._strata.get(record.stratum)
        if stats is None:
            return
        stats.update(record.reward)

    def update_batch(self, records: list[EvalRecord]) -> None:
        for r in records:
            self.update(r)

    def estimate(self) -> Estimate:
        N = self._n_total
        if N == 0:
            return self._empty_estimate()

        mean_hat = 0.0
        var_hat = 0.0

        for stats in self._strata.values():
            w = stats.population_size / N
            mean_hat += w * stats.mean

            if stats.sample_size >= 2:
                fpc = max(1.0 - stats.sample_size / stats.population_size, 0.0)
                var_hat += (w ** 2) * (stats.sample_variance / stats.sample_size) * fpc

        se = math.sqrt(var_hat) if var_hat > 0 else 0.0
        hw = self._z * se

        return Estimate(
            mean=mean_hat,
            std_error=se,
            ci_lower=mean_hat - hw,
            ci_upper=mean_hat + hw,
            confidence=self._confidence,
            n_evaluated=self.n_evaluated,
            n_total=N,
            strata=dict(self._strata),
        )

    def should_stop(self, config: StoppingConfig, budget_remaining: int) -> tuple[bool, str | None]:
        if budget_remaining <= 0:
            return True, "budget"

        est = self.estimate()

        ci_width = est.ci_upper - est.ci_lower
        if ci_width < config.ci_threshold and est.n_evaluated > 0:
            return True, "confidence"

        if est.n_evaluated > 0 and abs(est.mean) > 1e-12:
            if est.std_error / abs(est.mean) < config.relative_precision:
                return True, "precision"

        return False, None

    def _empty_estimate(self) -> Estimate:
        return Estimate(
            mean=0.0,
            std_error=float("inf"),
            ci_lower=float("-inf"),
            ci_upper=float("inf"),
            confidence=self._confidence,
            n_evaluated=0,
            n_total=self._n_total,
            strata=dict(self._strata),
        )
