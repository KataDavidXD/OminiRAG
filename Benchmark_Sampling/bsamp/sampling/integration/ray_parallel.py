from __future__ import annotations

import time
import logging
from typing import Any, Callable, Dict, List

from bsamp.sampling.types import BenchmarkItem, EvalRecord

try:
    import ray
except ImportError:
    ray = None

logger = logging.getLogger(__name__)


class RayItemEvaluator:
    """Evaluates a batch of benchmark items in parallel using Ray.
    
    This is meant to be used inside the sampling loop. The WTB bridge
    can provide the cache checks. So the simplest design is that the master
    process checks its local cache first, and only sends cache misses to Ray.
    """

    def __init__(self, num_cpus_per_task: int = 1):
        if ray is None:
            raise RuntimeError("Ray is not installed. Please install it with `pip install ray`.")
        
        self.num_cpus_per_task = num_cpus_per_task
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def evaluate_batch(
        self,
        items: List[BenchmarkItem],
        config: Dict[str, Any],
        eval_fn: Callable[[Dict[str, Any], str], float],
        step: int,
    ) -> List[EvalRecord]:
        """Evaluate a list of items in parallel.
        
        Args:
            items: The items to evaluate (should be cache misses).
            config: The RAG configuration to test.
            eval_fn: The function that computes the reward. Signature: f(config, item_id) -> float.
            step: The current sampling iteration step.
            
        Returns:
            A list of EvalRecord objects corresponding to the input items.
        """
        if not items:
            return []
            
        # We need to wrap the eval_fn in a ray.remote. 
        # Note: eval_fn must be serializable by Ray.
        @ray.remote(num_cpus=self.num_cpus_per_task)
        def _ray_eval_worker(
            cfg: Dict[str, Any],
            item: BenchmarkItem,
            s: int,
            fn: Callable[[Dict[str, Any], str], float]
        ) -> EvalRecord:
            start_time = time.time()
            reward = fn(cfg, item.item_id)
            wall_time_ms = (time.time() - start_time) * 1000.0
            
            return EvalRecord(
                item_id=item.item_id,
                stratum=item.stratum,
                reward=reward,
                step=s,
                cached=False,  # If it reached here, it was a cache miss
                wall_time_ms=wall_time_ms,
                allocation_snapshot=None,
            )

        futures = [
            _ray_eval_worker.remote(config, item, step, eval_fn)
            for item in items
        ]
        
        # Block until all evaluations are complete
        records = ray.get(futures)
        return records

    def close(self):
        """Cleanup Ray resources if needed."""
        # Typically ray.shutdown() is called at the very end of the application,
        # so we don't necessarily call it here unless we own the cluster.
        pass
