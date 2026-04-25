"""WTB integration bridge for the sampling component.

Wraps ``SamplingState`` inside a LangGraph workflow so that WTB's
checkpoint / rollback / fork semantics apply to the sampling session.

Key design:
    - Each *evaluation batch* is a single LangGraph node execution.
    - After every node, WTB auto-checkpoints the LangGraph state which
      contains the serialised ``SamplingState``.
    - ``fork_for_comparison`` forks **after item realisation** so both
      branches evaluate the exact same ``realized_items``.
    - An in-memory ``EvalCache`` avoids re-running the same
      ``(config_hash, item_id)`` pair across forks / rollbacks.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from bsamp.sampling.types import (
    CacheKey,
    EvalRecord,
    SamplingState,
)

logger = logging.getLogger(__name__)

_STATE_KEY = "_sampling_state"


# ---------------------------------------------------------------------------
# Eval cache (shared across forks within one process)
# ---------------------------------------------------------------------------

class EvalCache:
    """Thread-safe in-memory cache for ``(config_hash, item_id) -> reward``."""

    def __init__(self) -> None:
        self._store: Dict[CacheKey, float] = {}

    def get(self, key: CacheKey) -> Optional[float]:
        return self._store.get(key)

    def put(self, key: CacheKey, reward: float) -> None:
        self._store[key] = reward

    def has(self, key: CacheKey) -> bool:
        return key in self._store

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def hash_config(config: Dict[str, Any]) -> str:
    """Deterministic hash of a RAG configuration dict."""
    raw = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class SamplingGraphState(TypedDict):
    """State passed through the sampling LangGraph."""

    _sampling_state: dict
    _step_index: int
    _done: bool


# ---------------------------------------------------------------------------
# WTBSamplingBridge
# ---------------------------------------------------------------------------

class WTBSamplingBridge:
    """High-level bridge between the sampling component and WTB.

    Usage::

        bridge = WTBSamplingBridge(eval_fn=my_eval, cache=cache)
        bridge.initialise(sampling_state)

        # Run one evaluation batch (checkpoint happens automatically)
        bridge.step(records)

        # Fork for paired comparison of two configs
        fork_result = bridge.fork_for_comparison(new_state_overrides)

        # Rollback to an earlier checkpoint
        bridge.rollback(checkpoint_id)
    """

    def __init__(
        self,
        eval_fn: Optional[Callable] = None,
        cache: Optional[EvalCache] = None,
        mode: str = "testing",
    ) -> None:
        from wtb.sdk import WTBTestBench, WorkflowProject

        self._bench = WTBTestBench.create(mode=mode)
        self._cache = cache or EvalCache()
        self._eval_fn = eval_fn

        self._project_name = "sampling_session"
        project = WorkflowProject(
            name=self._project_name,
            graph_factory=self._build_graph,
        )
        self._bench.register_project(project)

        self._execution_id: Optional[str] = None
        self._sampling_state: Optional[SamplingState] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self, state: SamplingState) -> str:
        """Start a new sampling session, returning the WTB execution id."""
        self._sampling_state = state
        initial: Dict[str, Any] = {
            _STATE_KEY: state.to_dict(),
            "_step_index": 0,
            "_done": False,
        }
        execution = self._bench.run(
            project=self._project_name,
            initial_state=initial,
        )
        self._execution_id = execution.id
        return execution.id

    def update_state(self, state: SamplingState) -> None:
        """Push a new ``SamplingState`` into the current execution."""
        self._sampling_state = state

    @property
    def execution_id(self) -> Optional[str]:
        return self._execution_id

    @property
    def cache(self) -> EvalCache:
        return self._cache

    # ------------------------------------------------------------------
    # Checkpoint / rollback / fork
    # ------------------------------------------------------------------

    def get_checkpoints(self) -> list:
        """List available checkpoints for the current execution."""
        if self._execution_id is None:
            return []
        return self._bench.get_checkpoints(self._execution_id)

    def rollback(self, checkpoint_id: str) -> None:
        """Rollback the sampling session to a prior checkpoint.

        The in-memory cache is intentionally *not* cleared -- cached
        rewards remain valid because they are keyed on
        ``(config_hash, item_id)``, both of which are deterministic.
        """
        if self._execution_id is None:
            raise RuntimeError("No active execution to rollback")
        result = self._bench.rollback(self._execution_id, checkpoint_id)
        if not result.success:
            raise RuntimeError(f"Rollback failed: {result.error}")

        cp_state = self._bench.get_state(self._execution_id)
        raw = getattr(cp_state, "workflow_variables", {}) or {}
        if _STATE_KEY in raw:
            self._sampling_state = SamplingState.from_dict(raw[_STATE_KEY])
        logger.info("Rolled back execution %s to checkpoint %s",
                     self._execution_id, checkpoint_id)

    def fork_for_comparison(
        self,
        new_state_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Fork *after item realisation* for paired config comparison.

        Both branches share the same ``realized_items``.  The caller
        can supply ``new_state_overrides`` to change e.g. ``config_id``.

        Returns the new (forked) execution id.
        """
        if self._execution_id is None:
            raise RuntimeError("No active execution to fork")

        checkpoints = self.get_checkpoints()
        if not checkpoints:
            raise RuntimeError("No checkpoints available for forking")

        cp_id = str(checkpoints[0].id)

        merge: Dict[str, Any] | None = None
        if new_state_overrides:
            current_dict = (
                self._sampling_state.to_dict()
                if self._sampling_state
                else {}
            )
            current_dict.update(new_state_overrides)
            merge = {
                _STATE_KEY: current_dict,
                "_step_index": 0,
                "_done": False,
            }

        fork_result = self._bench.fork(
            self._execution_id,
            checkpoint_id=cp_id,
            new_initial_state=merge,
        )
        logger.info("Forked execution %s -> %s at checkpoint %s",
                     self._execution_id, fork_result.fork_execution_id, cp_id)
        return fork_result.fork_execution_id

    # ------------------------------------------------------------------
    # Cached evaluation helper
    # ------------------------------------------------------------------

    def evaluate_with_cache(
        self,
        config: Dict[str, Any],
        item_id: str,
        eval_fn: Optional[Callable] = None,
    ) -> tuple[float, bool]:
        """Evaluate a single item, returning ``(reward, was_cached)``.

        Checks the cache first; on miss calls ``eval_fn(config, item_id)``
        and stores the result.
        """
        fn = eval_fn or self._eval_fn
        if fn is None:
            raise RuntimeError("No eval_fn provided")

        cfg_hash = hash_config(config)
        key = CacheKey(config_hash=cfg_hash, item_id=item_id)

        cached = self._cache.get(key)
        if cached is not None:
            return cached, True

        reward = fn(config, item_id)
        self._cache.put(key, reward)
        return reward, False

    # ------------------------------------------------------------------
    # LangGraph graph factory (minimal single-node graph)
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Create a minimal LangGraph with a single pass-through node."""
        from langgraph.graph import StateGraph, END

        def sampling_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "_step_index": state.get("_step_index", 0) + 1,
                "_done": True,
            }

        g = StateGraph(SamplingGraphState)
        g.add_node("sampling_step", sampling_node)
        g.add_edge("__start__", "sampling_step")
        g.add_edge("sampling_step", END)
        return g

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release WTB resources."""
        if self._bench is not None:
            self._bench.close()
