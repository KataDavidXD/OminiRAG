"""Configuration comparison -- CI elimination and paired t-test.

Two comparison modes:
  1. ``should_eliminate()``: non-overlapping CI quick elimination.
     If A's CI upper bound < B's CI lower bound, eliminate A.
  2. ``paired_compare()``:  paired t-test over shared items.
     Requires both configs to be evaluated on the same realised item set
     (enforced via WTB fork semantics).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from bsamp.sampling.types import Estimate, EvalRecord


@dataclass
class PairedResult:
    mean_diff: float
    std_error: float
    ci_lower: float
    ci_upper: float
    t_stat: float
    p_value: float
    n_shared: int
    winner: str | None
    confidence: float


def should_eliminate(est_a: Estimate, est_b: Estimate) -> str | None:
    """Non-overlapping CI elimination. Returns 'A', 'B', or None."""
    if est_a.ci_upper < est_b.ci_lower:
        return "A"
    if est_b.ci_upper < est_a.ci_lower:
        return "B"
    return None


def paired_compare(
    records_a: list[EvalRecord],
    records_b: list[EvalRecord],
    confidence: float = 0.95,
) -> PairedResult:
    """Paired t-test over shared items evaluated by both configs.

    ``records_a`` and ``records_b`` come from two config evaluations on the
    same realised item set.  Items are matched by ``item_id``.
    """
    map_a = {r.item_id: r.reward for r in records_a}
    map_b = {r.item_id: r.reward for r in records_b}

    shared = sorted(set(map_a.keys()) & set(map_b.keys()))
    n = len(shared)

    if n == 0:
        return PairedResult(
            mean_diff=0.0,
            std_error=float("inf"),
            ci_lower=float("-inf"),
            ci_upper=float("inf"),
            t_stat=0.0,
            p_value=1.0,
            n_shared=0,
            winner=None,
            confidence=confidence,
        )

    diffs = [map_a[item_id] - map_b[item_id] for item_id in shared]
    mean_d = sum(diffs) / n

    if n < 2:
        return PairedResult(
            mean_diff=mean_d,
            std_error=float("inf"),
            ci_lower=float("-inf"),
            ci_upper=float("inf"),
            t_stat=0.0,
            p_value=1.0,
            n_shared=n,
            winner=None,
            confidence=confidence,
        )

    ss = sum((d - mean_d) ** 2 for d in diffs)
    sd = math.sqrt(ss / (n - 1))
    se = sd / math.sqrt(n)

    t_stat = mean_d / se if se > 0 else 0.0

    z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}.get(confidence, 1.96)
    hw = z * se

    p_value = _approx_two_sided_p(t_stat, n - 1)

    winner: str | None = None
    if mean_d - hw > 0:
        winner = "A"
    elif mean_d + hw < 0:
        winner = "B"

    return PairedResult(
        mean_diff=mean_d,
        std_error=se,
        ci_lower=mean_d - hw,
        ci_upper=mean_d + hw,
        t_stat=t_stat,
        p_value=p_value,
        n_shared=n,
        winner=winner,
        confidence=confidence,
    )


def _approx_two_sided_p(t: float, df: int) -> float:
    """Rough two-sided p-value from t-statistic using normal approx."""
    x = abs(t)
    p_one_tail = 0.5 * math.erfc(x / math.sqrt(2))
    return 2.0 * p_one_tail
