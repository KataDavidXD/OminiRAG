from __future__ import annotations

from bsamp.sampling.types import BenchmarkItem, StratumStats
from bsamp.sampling.samplers.stratified import StratifiedSampler


def _pool(label: str, n: int) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            item_id=f"{label}::{i}",
            benchmark="test",
            stratum=label,
            payload={"q": i},
            target={"a": i},
            metadata={"domain": label, "length": i * 100},
        )
        for i in range(n)
    ]


def _stats(label: str, pop: int, sample_var: float = 1.0) -> StratumStats:
    s = StratumStats(stratum=label, population_size=pop)
    if sample_var > 0:
        for v in [0.0, sample_var * 2]:
            s.update(v)
    return s


class TestProportionalAllocation:
    def test_budget_sum(self):
        strata = {"A": _pool("A", 80), "B": _pool("B", 20)}
        stats = {"A": _stats("A", 80), "B": _stats("B", 20)}
        sampler = StratifiedSampler(allocation="proportional", seed=42)

        real = sampler.select(strata, stats, budget=20)
        assert sum(real.allocation) == 20
        assert len(real.realized_items) == 20

    def test_proportions_roughly_correct(self):
        strata = {"A": _pool("A", 800), "B": _pool("B", 200)}
        stats = {"A": _stats("A", 800), "B": _stats("B", 200)}
        sampler = StratifiedSampler(allocation="proportional", seed=1)

        real = sampler.select(strata, stats, budget=100)
        alloc = dict(zip(sorted(strata.keys()), real.allocation))
        assert alloc["A"] > alloc["B"]
        assert alloc["A"] >= 70

    def test_items_from_correct_strata(self):
        strata = {"X": _pool("X", 50), "Y": _pool("Y", 50)}
        stats = {"X": _stats("X", 50), "Y": _stats("Y", 50)}
        sampler = StratifiedSampler(seed=7)

        real = sampler.select(strata, stats, budget=20)
        x_items = {i for i in real.realized_items if i.startswith("X::")}
        y_items = {i for i in real.realized_items if i.startswith("Y::")}
        assert len(x_items) + len(y_items) == len(real.realized_items)

    def test_no_duplicates(self):
        strata = {"A": _pool("A", 100)}
        stats = {"A": _stats("A", 100)}
        sampler = StratifiedSampler(seed=0)

        real = sampler.select(strata, stats, budget=50)
        assert len(real.realized_items) == len(set(real.realized_items))


class TestNeymanAllocation:
    def test_high_variance_gets_more(self):
        strata = {"lo": _pool("lo", 100), "hi": _pool("hi", 100)}
        stats_lo = StratumStats(stratum="lo", population_size=100)
        for v in [1.0, 1.01, 1.02, 0.99, 0.98]:
            stats_lo.update(v)

        stats_hi = StratumStats(stratum="hi", population_size=100)
        for v in [0.0, 5.0, 0.0, 5.0, 0.0]:
            stats_hi.update(v)

        stats = {"lo": stats_lo, "hi": stats_hi}
        sampler = StratifiedSampler(allocation="neyman", seed=42)
        real = sampler.select(strata, stats, budget=50)

        alloc = dict(zip(sorted(strata.keys()), real.allocation))
        assert alloc["hi"] > alloc["lo"], "Higher-variance stratum should get more samples"

    def test_falls_back_without_variance(self):
        strata = {"A": _pool("A", 60), "B": _pool("B", 40)}
        stats = {"A": StratumStats("A", 60), "B": StratumStats("B", 40)}
        sampler = StratifiedSampler(allocation="neyman", seed=0)

        real = sampler.select(strata, stats, budget=20)
        assert sum(real.allocation) == 20


class TestSamplerState:
    def test_checkpoint_restore(self):
        strata = {"A": _pool("A", 100)}
        stats = {"A": _stats("A", 100)}
        sampler = StratifiedSampler(seed=42)

        real1 = sampler.select(strata, stats, budget=10)
        saved = sampler.get_state()

        real2 = sampler.select(strata, stats, budget=10)

        sampler.set_state(saved)
        real3 = sampler.select(strata, stats, budget=10)

        assert real2.realized_items == real3.realized_items
