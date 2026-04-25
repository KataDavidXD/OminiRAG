import pytest
from typing import Any, Dict

from bsamp.sampling.integration.wtb import WTBSamplingBridge, EvalCache
from bsamp.sampling.types import SamplingState, EvalRecord, ItemRealization, StratumStats

def test_cache_hit():
    cache = EvalCache()
    bridge = WTBSamplingBridge(eval_fn=lambda c, i: 42.0, cache=cache, mode="testing")
    
    config = {"model": "gpt-4"}
    item_id = "test::1"
    
    # First call: miss
    reward, cached = bridge.evaluate_with_cache(config, item_id)
    assert reward == 42.0
    assert not cached
    
    # Second call: hit
    reward, cached = bridge.evaluate_with_cache(config, item_id)
    assert reward == 42.0
    assert cached
    
    bridge.close()

def test_wtb_checkpoint_restore():
    bridge = WTBSamplingBridge(mode="testing")
    
    state = SamplingState(
        config_id="config_1",
        benchmark="test_bench",
        sampler_type="stratified",
        budget_total=10,
        budget_used=5,
        strata_stats={"A": StratumStats("A", 100, 5, 200, 10000)},
        sampler_state={"some": "state"},
        rng_state=None,
        history=[
            EvalRecord("test::1", "A", 40.0, 1, False, 100.0)
        ],
        realizations=[
            ItemRealization([5], ["test::1"], None)
        ],
        stopped=False,
        stop_reason=None
    )
    
    # Initialise the workflow
    exec_id = bridge.initialise(state)
    assert exec_id is not None
    
    # Push the state into WTB checkpoint by running step
    # WTBSamplingBridge actually auto-checkpoints when initialized, because it runs the workflow
    # which has one step.
    
    checkpoints = bridge.get_checkpoints()
    assert len(checkpoints) > 0
    
    # Modify state locally to prove restore works
    bridge._sampling_state.budget_used = 999
    
    # Rollback to the initial checkpoint
    bridge.rollback(str(checkpoints[0].id))
    
    # Should be back to 5
    assert bridge._sampling_state.budget_used == 5
    assert bridge._sampling_state.history[0].reward == 40.0
    
    bridge.close()

def test_fork_for_comparison():
    bridge = WTBSamplingBridge(mode="testing")
    
    state = SamplingState(
        config_id="config_1",
        benchmark="test_bench",
        sampler_type="stratified",
        budget_total=10,
        budget_used=5,
        strata_stats={},
        sampler_state={},
        rng_state=None,
        history=[],
        realizations=[
            ItemRealization([5], ["test::1", "test::2", "test::3", "test::4", "test::5"], None)
        ],
        stopped=False,
        stop_reason=None
    )
    
    exec_id = bridge.initialise(state)
    checkpoints = bridge.get_checkpoints()
    assert len(checkpoints) > 0
    
    fork_id = bridge.fork_for_comparison({"config_id": "config_2"})
    assert fork_id is not None
    assert fork_id != exec_id
    
    bridge.close()
