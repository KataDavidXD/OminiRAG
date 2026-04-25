import pytest

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from bsamp.sampling.integration.ray_parallel import RayItemEvaluator
from bsamp.sampling.types import BenchmarkItem

@pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")
def test_ray_parallel_evaluate_batch():
    evaluator = RayItemEvaluator(num_cpus_per_task=1)
    
    items = [
        BenchmarkItem(f"item::{i}", "test", "A", {}, None, {}) for i in range(5)
    ]
    
    config = {"a": 1}
    
    # dummy eval function that must be globally accessible or picklable
    # Using a simple function:
    def my_eval(c, i_id):
        return 42.0

    records = evaluator.evaluate_batch(items, config, my_eval, step=1)
    
    assert len(records) == 5
    for i, r in enumerate(records):
        assert r.item_id == f"item::{i}"
        assert r.reward == 42.0
        assert r.step == 1
        assert not r.cached
        
    evaluator.close()
