"""Tests for benchmark.sampling.estimator."""
from __future__ import annotations

import math

from bsamp.sampling.types import EvalRecord, StratumStats
from bsamp.sampling.estimator import SequentialEstimator, StoppingConfig


def _make_stats(label: str, pop: int) -> StratumStats:
    return StratumStats(stratum=label, population_size=pop)


def test_single_stratum_convergence():
    stats = {"A": _make_stats("A", 100)}
    est = SequentialEstimator(stats, confidence=0.95)

    for i in range(50):
        est.update(EvalRecord(
            item_id=f"item_{i}", stratum="A", reward=1.0,
            step=0, cached=False, wall_time_ms=10.0,
        ))

    result = est.estimate()
    assert abs(result.mean - 1.0) < 1e-9
    assert result.std_error < 0.01
    assert result.ci_lower < 1.0 < result.ci_upper or abs(result.mean - 1.0) < 1e-9


def test_two_strata_weighted_mean():
    stats = {
        "A": _make_stats("A", 80),
        "B": _make_stats("B", 20),
    }
    est = SequentialEstimator(stats)

    for i in range(20):
        est.update(EvalRecord(f"a_{i}", "A", 1.0, 0, False, 10.0))
    for i in range(10):
        est.update(EvalRecord(f"b_{i}", "B", 0.0, 0, False, 10.0))

    result = est.estimate()
    expected = 0.8 * 1.0 + 0.2 * 0.0
    assert abs(result.mean - expected) < 1e-9


def test_early_stop_budget():
    stats = {"A": _make_stats("A", 100)}
    est = SequentialEstimator(stats)
    cfg = StoppingConfig()
    stop, reason = est.should_stop(cfg, budget_remaining=0)
    assert stop is True
    assert reason == "budget"


def test_early_stop_confidence():
    stats = {"A": _make_stats("A", 1000)}
    est = SequentialEstimator(stats)

    for i in range(200):
        est.update(EvalRecord(f"item_{i}", "A", 0.5, 0, False, 5.0))

    cfg = StoppingConfig(ci_threshold=1.0, relative_precision=1.0)
    stop, reason = est.should_stop(cfg, budget_remaining=100)
    assert stop is True
    assert reason in ("confidence", "precision")


def test_no_stop_early():
    stats = {"A": _make_stats("A", 10000)}
    est = SequentialEstimator(stats)

    for i in range(3):
        est.update(EvalRecord(f"item_{i}", "A", float(i), 0, False, 5.0))

    cfg = StoppingConfig(ci_threshold=0.001, relative_precision=0.001)
    stop, _ = est.should_stop(cfg, budget_remaining=100)
    assert stop is False


def test_batch_update():
    stats = {"X": _make_stats("X", 50)}
    est = SequentialEstimator(stats)

    records = [
        EvalRecord(f"x_{i}", "X", 2.0, 0, False, 5.0)
        for i in range(10)
    ]
    est.update_batch(records)
    result = est.estimate()
    assert abs(result.mean - 2.0) < 1e-9
    assert result.n_evaluated == 10
