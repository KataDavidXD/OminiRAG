from __future__ import annotations

import json

from bsamp.sampling.types import (
    BenchmarkItem,
    CacheKey,
    EvalRecord,
    Estimate,
    ItemRealization,
    SamplingState,
    StratumStats,
)


def _make_item(item_id: str = "test::1", stratum: str = "s0") -> BenchmarkItem:
    return BenchmarkItem(
        item_id=item_id,
        benchmark="test",
        stratum=stratum,
        payload={"query": "hello"},
        target={"answer": "world"},
        metadata={"domain": "test", "length": 100},
    )


class TestBenchmarkItem:
    def test_roundtrip(self):
        item = _make_item()
        d = item.to_dict()
        restored = BenchmarkItem.from_dict(d)
        assert restored == item

    def test_frozen(self):
        item = _make_item()
        try:
            item.item_id = "changed"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestEvalRecord:
    def test_roundtrip(self):
        rec = EvalRecord(
            item_id="x::1",
            stratum="s0",
            reward=0.75,
            step=3,
            cached=False,
            wall_time_ms=120.5,
            allocation_snapshot=[10, 20],
        )
        d = rec.to_dict()
        restored = EvalRecord.from_dict(d)
        assert restored.item_id == rec.item_id
        assert restored.reward == rec.reward
        assert restored.allocation_snapshot == [10, 20]


class TestStratumStats:
    def test_running_stats(self):
        s = StratumStats(stratum="s0", population_size=100)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            s.update(v)
        assert s.sample_size == 5
        assert abs(s.mean - 3.0) < 1e-9
        assert s.sample_variance > 0

    def test_empty(self):
        s = StratumStats(stratum="s0", population_size=50)
        assert s.mean == 0.0
        assert s.variance == 0.0
        assert s.std_error == float("inf")


class TestEstimate:
    def test_roundtrip(self):
        stats = {"s0": StratumStats(stratum="s0", population_size=100, sample_size=10,
                                     running_sum=5.0, running_sum_sq=3.0)}
        est = Estimate(
            mean=0.5, std_error=0.02, ci_lower=0.46, ci_upper=0.54,
            confidence=0.95, n_evaluated=10, n_total=100, strata=stats,
        )
        d = est.to_dict()
        restored = Estimate.from_dict(d)
        assert restored.mean == est.mean
        assert restored.strata["s0"].population_size == 100


class TestItemRealization:
    def test_roundtrip(self):
        import random
        rng = random.Random(42)
        state = rng.getstate()
        ir = ItemRealization(
            allocation=[5, 10],
            realized_items=["a::1", "b::2"],
            rng_state_before=state,
        )
        d = ir.to_dict()
        restored = ItemRealization.from_dict(d)
        assert restored.allocation == [5, 10]
        assert restored.realized_items == ["a::1", "b::2"]


class TestSamplingState:
    def test_json_roundtrip(self):
        import random
        rng = random.Random(99)

        ss = SamplingState(
            config_id="cfg_1",
            benchmark="test",
            sampler_type="stratified",
            budget_total=100,
            budget_used=20,
            strata_stats={"s0": StratumStats("s0", 50, 10, 5.0, 3.0)},
            sampler_state={"allocation_method": "proportional"},
            rng_state=rng.getstate(),
            history=[EvalRecord("x::1", "s0", 0.8, 1, False, 50.0)],
            realizations=[ItemRealization([10], ["x::1"], rng.getstate())],
            stopped=False,
            stop_reason=None,
        )
        json_str = ss.to_json()
        restored = SamplingState.from_json(json_str)
        assert restored.config_id == "cfg_1"
        assert restored.budget_used == 20
        assert len(restored.history) == 1
        assert restored.history[0].reward == 0.8


class TestCacheKey:
    def test_named_tuple(self):
        k = CacheKey(config_hash="abc", item_id="x::1")
        assert k.config_hash == "abc"
        assert k == CacheKey("abc", "x::1")
