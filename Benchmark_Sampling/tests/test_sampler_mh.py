import pytest

from bsamp.sampling.samplers.mh import MetropolisHastingsSampler
from bsamp.sampling.types import BenchmarkItem, StratumStats


def test_mh_sampler_initialization():
    sampler = MetropolisHastingsSampler(rng_seed=42)
    assert sampler.step_count == 0
    assert sampler.n_accepted == 0
    assert sampler.temperature == 1.0


def test_mh_sampler_select():
    sampler = MetropolisHastingsSampler(rng_seed=42, initial_temperature=10.0)
    
    # Create fake strata
    strata = {
        "A": [
            BenchmarkItem(f"A::{i}", "test", "A", {}, None, {}) for i in range(10)
        ],
        "B": [
            BenchmarkItem(f"B::{i}", "test", "B", {}, None, {}) for i in range(20)
        ],
        "C": [
            BenchmarkItem(f"C::{i}", "test", "C", {}, None, {}) for i in range(30)
        ]
    }
    
    # Create fake stats
    strata_stats = {
        "A": StratumStats("A", 10),
        "B": StratumStats("B", 20),
        "C": StratumStats("C", 30)
    }
    # Pretend we have some high variance in A to force allocation towards A
    strata_stats["A"].sample_size = 5
    strata_stats["A"].running_sum_sq = 500  # High var
    
    strata_stats["B"].sample_size = 5
    strata_stats["B"].running_sum_sq = 5    # Low var
    
    strata_stats["C"].sample_size = 5
    strata_stats["C"].running_sum_sq = 5    # Low var
    
    budget = 15
    realization = sampler.select(strata, strata_stats, budget)
    
    assert sum(realization.allocation) == budget
    assert len(realization.realized_items) == budget
    assert sampler.step_count == 1
    assert len(sampler.energy_trace) == 1
    
    # Run it a few more times
    for _ in range(50):
        sampler.select(strata, strata_stats, budget)
        
    assert sampler.step_count == 51
    assert sampler.temperature < 10.0


def test_mh_sampler_checkpointing():
    sampler = MetropolisHastingsSampler(rng_seed=42)
    strata = {
        "A": [BenchmarkItem(f"A::{i}", "test", "A", {}, None, {}) for i in range(10)],
        "B": [BenchmarkItem(f"B::{i}", "test", "B", {}, None, {}) for i in range(10)]
    }
    strata_stats = {
        "A": StratumStats("A", 10),
        "B": StratumStats("B", 10),
    }
    
    # Run a few steps
    sampler.select(strata, strata_stats, 6)
    sampler.select(strata, strata_stats, 6)
    
    state = sampler.get_state()
    
    # Create new sampler and restore state
    sampler2 = MetropolisHastingsSampler(rng_seed=999) # different seed initially
    sampler2.set_state(state)
    
    assert sampler2.step_count == sampler.step_count
    assert sampler2.n_accepted == sampler.n_accepted
    assert sampler2.temperature == sampler.temperature
    assert sampler2.current_allocation == sampler.current_allocation
    
    # Next realization should be identical
    r1 = sampler.select(strata, strata_stats, 6)
    r2 = sampler2.select(strata, strata_stats, 6)
    
    assert r1.allocation == r2.allocation
    assert r1.realized_items == r2.realized_items
