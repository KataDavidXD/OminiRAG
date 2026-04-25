"""Sampling diagnostics -- quality metrics for sampled data.

Provides:
  - ESS (effective sample size):  based on reward-chain autocorrelation, ESS = n / (1 + 2*sum(rho_k))
  - Variance reduction (design effect): Var(stratified) / Var(SRS); < 1 means stratification helps
  - Gelman-Rubin R-hat:  multi-chain convergence diagnostic; values near 1.0 indicate convergence
  - CI width trace:       confidence-interval width at each estimation step
  - Acceptance rate:      accept / total for MH/HMC; ideal range 0.2-0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from bsamp.sampling.types import Estimate, EvalRecord, StratumStats


@dataclass
class SamplingDiagnostics:
    """Aggregated diagnostics for a sampling session."""

    acceptance_rate: float = 0.0
    effective_sample_size: float = 0.0
    r_hat: float | None = None
    energy_trace: list[float] = field(default_factory=list)
    allocation_trace: list[list[int]] = field(default_factory=list)
    variance_reduction: float = 0.0
    ci_width_trace: list[float] = field(default_factory=list)


def compute_ess(rewards: Sequence[float], max_lag: int | None = None) -> float:
    """Effective sample size from autocorrelation of reward chain.

    ``ESS = n / (1 + 2 * sum_{k=1}^{K} rho_k)``
    where rho_k is the lag-k autocorrelation. Summation stops when rho_k
    first becomes negative (initial positive sequence heuristic).
    """
    n = len(rewards)
    if n < 4:
        return float(n)

    mean = sum(rewards) / n
    c0 = sum((r - mean) ** 2 for r in rewards) / n
    if c0 < 1e-15:
        return float(n)

    if max_lag is None:
        max_lag = min(n // 2, 200)

    tau = 1.0
    for lag in range(1, max_lag + 1):
        cov = sum(
            (rewards[i] - mean) * (rewards[i + lag] - mean)
            for i in range(n - lag)
        ) / n
        rho = cov / c0
        if rho < 0:
            break
        tau += 2 * rho

    return n / tau


def compute_variance_reduction(
    stratified_var: float,
    simple_random_var: float,
) -> float:
    """Design effect: ``Var(stratified) / Var(simple_random)``.

    Values < 1 indicate variance reduction from stratification.
    """
    if simple_random_var < 1e-15:
        return 1.0
    return stratified_var / simple_random_var


def simple_random_variance(records: list[EvalRecord]) -> float:
    """Variance of the simple-random-sample mean estimator."""
    n = len(records)
    if n < 2:
        return 0.0
    rewards = [r.reward for r in records]
    mean = sum(rewards) / n
    ss = sum((r - mean) ** 2 for r in rewards)
    return ss / (n * (n - 1))


def build_diagnostics(
    records: list[EvalRecord],
    estimates: list[Estimate],
    acceptance_rate: float = 0.0,
    energy_trace: list[float] | None = None,
    allocation_trace: list[list[int]] | None = None,
) -> SamplingDiagnostics:
    """Construct a diagnostics summary from sampling history."""
    rewards = [r.reward for r in records]
    ess = compute_ess(rewards) if rewards else 0.0

    ci_widths = [e.ci_upper - e.ci_lower for e in estimates]

    strat_var = estimates[-1].std_error ** 2 if estimates else 0.0
    srs_var = simple_random_variance(records)
    vr = compute_variance_reduction(strat_var, srs_var)

    return SamplingDiagnostics(
        acceptance_rate=acceptance_rate,
        effective_sample_size=ess,
        r_hat=None,
        energy_trace=energy_trace or [],
        allocation_trace=allocation_trace or [],
        variance_reduction=vr,
        ci_width_trace=ci_widths,
    )


def gelman_rubin(chains: list[list[float]]) -> float:
    """Gelman-Rubin R-hat for convergence of multiple chains.

    Each chain is a list of scalar samples. Returns R-hat >= 1;
    values near 1.0 indicate convergence.
    """
    m = len(chains)
    if m < 2:
        return 1.0

    ns = [len(c) for c in chains]
    n = min(ns)
    if n < 2:
        return float("inf")

    chain_means = [sum(c[:n]) / n for c in chains]
    grand_mean = sum(chain_means) / m

    B = n / (m - 1) * sum((cm - grand_mean) ** 2 for cm in chain_means)

    W = 0.0
    for j, c in enumerate(chains):
        cm = chain_means[j]
        W += sum((c[i] - cm) ** 2 for i in range(n)) / (n - 1)
    W /= m

    if W < 1e-15:
        return 1.0

    var_hat = (1 - 1 / n) * W + (1 / n) * B
    return math.sqrt(var_hat / W)
