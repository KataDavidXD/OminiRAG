from __future__ import annotations

from bsamp.sampling.types import EvalRecord, Estimate, StratumStats
from bsamp.sampling.comparison import should_eliminate, paired_compare


def _est(mean: float, ci_lo: float, ci_hi: float) -> Estimate:
    return Estimate(
        mean=mean, std_error=(ci_hi - ci_lo) / 4,
        ci_lower=ci_lo, ci_upper=ci_hi,
        confidence=0.95, n_evaluated=50, n_total=100,
        strata={},
    )


class TestShouldEliminate:
    def test_a_clearly_worse(self):
        a = _est(0.3, 0.25, 0.35)
        b = _est(0.8, 0.75, 0.85)
        assert should_eliminate(a, b) == "A"

    def test_b_clearly_worse(self):
        a = _est(0.8, 0.75, 0.85)
        b = _est(0.3, 0.25, 0.35)
        assert should_eliminate(a, b) == "B"

    def test_overlapping_ci(self):
        a = _est(0.5, 0.40, 0.60)
        b = _est(0.55, 0.45, 0.65)
        assert should_eliminate(a, b) is None


class TestPairedCompare:
    def test_a_wins(self):
        recs_a = [EvalRecord(f"i{i}", "s0", 1.0, 0, False, 5.0) for i in range(30)]
        recs_b = [EvalRecord(f"i{i}", "s0", 0.0, 0, False, 5.0) for i in range(30)]
        result = paired_compare(recs_a, recs_b)
        assert result.winner == "A"
        assert result.mean_diff > 0
        assert result.n_shared == 30

    def test_b_wins(self):
        recs_a = [EvalRecord(f"i{i}", "s0", 0.0, 0, False, 5.0) for i in range(30)]
        recs_b = [EvalRecord(f"i{i}", "s0", 1.0, 0, False, 5.0) for i in range(30)]
        result = paired_compare(recs_a, recs_b)
        assert result.winner == "B"
        assert result.mean_diff < 0

    def test_no_shared_items(self):
        recs_a = [EvalRecord(f"a{i}", "s0", 1.0, 0, False, 5.0) for i in range(10)]
        recs_b = [EvalRecord(f"b{i}", "s0", 0.0, 0, False, 5.0) for i in range(10)]
        result = paired_compare(recs_a, recs_b)
        assert result.n_shared == 0
        assert result.winner is None

    def test_tied(self):
        recs_a = [EvalRecord(f"i{i}", "s0", 0.5, 0, False, 5.0) for i in range(20)]
        recs_b = [EvalRecord(f"i{i}", "s0", 0.5, 0, False, 5.0) for i in range(20)]
        result = paired_compare(recs_a, recs_b)
        assert abs(result.mean_diff) < 1e-9
        assert result.winner is None
