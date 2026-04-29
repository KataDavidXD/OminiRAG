"""
Cache-aware batch runner -- orchestrates WTB execution with reuse.

The main entry point ``run_batch_with_reuse`` partitions work items into
full misses, partial hits, and full hits, then dispatches each group
through the appropriate WTB path (batch run, fork, or checkpoint read).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .config_types import (
    RAGConfig,
    BenchmarkQuestion,
    WorkItem,
    NODE_ORDER,
    state_content_hash,
)
from .reuse_ledger import ReuseLedger, MaterializedEntry
from .graph_factories import config_to_graph_factory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch_with_reuse(
    configs: List[RAGConfig],
    questions: List[BenchmarkQuestion],
    bench: Any,
    ledger: ReuseLedger,
) -> List[Dict[str, Any]]:
    """Execute ``configs x questions`` with cache-aware reuse.

    Returns a list of result dicts, one per ``(config, question)`` pair
    in row-major order (outer loop = configs, inner loop = questions).

    Parameters
    ----------
    configs
        Pipeline configurations to evaluate.
    questions
        Benchmark questions to run through each config.
    bench
        ``WTBTestBench`` instance (must be ``mode="development"`` with
        SQLite persistence for checkpoint access).
    ledger
        Shared ``ReuseLedger`` for cross-config reuse tracking.
    """
    from wtb.sdk import WTBTestBench, WorkflowProject

    # Phase 0: partition
    work_items: List[WorkItem] = []
    for config in configs:
        for question in questions:
            depth, entry = ledger.longest_matching_prefix(config, question.question_id)
            work_items.append(WorkItem(
                config=config,
                question=question,
                reuse_depth=depth,
                reuse_entry=entry,
            ))

    full_misses = [w for w in work_items if w.is_full_miss]
    partial_hits = [w for w in work_items if w.is_partial_hit]
    full_hits = [w for w in work_items if w.is_full_hit]

    logger.info(
        "Batch partition: %d miss, %d partial, %d hit (of %d total)",
        len(full_misses), len(partial_hits), len(full_hits), len(work_items),
    )

    results_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Phase 1: full misses -- execute via WTB batch
    _execute_full_misses(full_misses, bench, ledger, results_map)

    # Phase 2: partial hits -- fork from cached checkpoint
    _execute_partial_hits(partial_hits, bench, ledger, results_map)

    # Phase 3: full hits -- read cached result
    _read_full_hits(full_hits, bench, ledger, results_map)

    # Assemble ordered results
    results: List[Dict[str, Any]] = []
    for config in configs:
        for question in questions:
            key = (config.config_key(), question.question_id)
            results.append(results_map.get(key, {"status": "missing"}))

    return results


def record_checkpoints(
    ledger: ReuseLedger,
    config: RAGConfig,
    question_id: str,
    execution_id: str,
    checkpoints: List[Dict[str, Any]],
    checkpoint_db_path: str = "",
) -> None:
    """Record all prefix-depth entries in the ledger for one execution."""
    ledger.record_all_prefixes(
        config=config,
        question_id=question_id,
        execution_id=execution_id,
        checkpoints=checkpoints,
        checkpoint_db_path=checkpoint_db_path,
    )


# ---------------------------------------------------------------------------
# Internal: Phase 1 -- full misses
# ---------------------------------------------------------------------------

def _execute_full_misses(
    items: List[WorkItem],
    bench: Any,
    ledger: ReuseLedger,
    results_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    """Execute cache-miss work items grouped by config key."""
    if not items:
        return

    from wtb.sdk import WTBTestBench, WorkflowProject

    grouped: Dict[str, List[WorkItem]] = defaultdict(list)
    for w in items:
        grouped[w.config.config_key()].append(w)

    for config_key, group in grouped.items():
        config = group[0].config
        project_name = f"reuse_{config_key.replace('/', '_')}"

        factory = config_to_graph_factory(config)
        project = WorkflowProject(name=project_name, graph_factory=factory)

        try:
            bench.register_project(project)
        except Exception:
            pass

        test_cases = []
        for w in group:
            initial_state = _build_initial_state(w.config, w.question)
            test_cases.append(initial_state)

        try:
            batch = bench.run_batch_test(
                project=project_name,
                variant_matrix=[{"query_processing": "default"}],
                test_cases=test_cases,
            )

            for result, w in zip(batch.results, group):
                rkey = (w.config.config_key(), w.question.question_id)

                result_dict: Dict[str, Any] = {
                    "status": "completed" if result.success else "failed",
                    "execution_id": result.execution_id,
                    "combination_name": result.combination_name,
                    "config_key": w.config.config_key(),
                    "question_id": w.question.question_id,
                    "reuse_type": "full_miss",
                }

                if result.execution_id:
                    try:
                        execution = bench.get_execution(result.execution_id)
                        result_dict["current_state"] = execution.current_state
                        result_dict["metadata"] = execution.metadata

                        cps = bench.get_checkpoints(result.execution_id)
                        cp_dicts = [
                            {"checkpoint_id": str(cp.id), "step": cp.step}
                            for cp in cps
                        ]
                        db_path = (execution.metadata or {}).get(
                            "checkpoint_db_path", ""
                        )
                        record_checkpoints(
                            ledger, w.config, w.question.question_id,
                            result.execution_id, cp_dicts, db_path,
                        )
                    except Exception as exc:
                        logger.debug("Checkpoint recording failed: %s", exc)

                results_map[rkey] = result_dict

        except Exception as exc:
            logger.warning("Batch execution failed for %s: %s", config_key, exc)
            for w in group:
                rkey = (w.config.config_key(), w.question.question_id)
                results_map[rkey] = {
                    "status": "error",
                    "error": str(exc),
                    "reuse_type": "full_miss",
                }


# ---------------------------------------------------------------------------
# Internal: Phase 2 -- partial hits (fork)
# ---------------------------------------------------------------------------

def _execute_partial_hits(
    items: List[WorkItem],
    bench: Any,
    ledger: ReuseLedger,
    results_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    """Fork from cached checkpoints and run remaining nodes."""
    if not items:
        return

    for w in items:
        rkey = (w.config.config_key(), w.question.question_id)
        entry: Optional[MaterializedEntry] = w.reuse_entry

        if entry is None:
            results_map[rkey] = {"status": "error", "error": "no reuse entry",
                                 "reuse_type": "partial_hit"}
            continue

        try:
            from wtb.sdk import BatchTestResult

            mock_result = BatchTestResult(
                combination_name=f"fork_{w.config.config_key()}",
                variant_config={},
                success=True,
                execution_id=entry.execution_id,
                last_checkpoint_id=entry.checkpoint_id,
            )

            new_state = _build_initial_state(w.config, w.question)
            fork = bench.fork_batch_result(
                mock_result,
                checkpoint_id=entry.checkpoint_id,
                new_state=new_state,
            )

            result_dict: Dict[str, Any] = {
                "status": "forked",
                "execution_id": fork.fork_execution_id if fork.fork_execution_id else "",
                "source_execution_id": entry.execution_id,
                "fork_depth": w.reuse_depth,
                "config_key": w.config.config_key(),
                "question_id": w.question.question_id,
                "reuse_type": "partial_hit",
            }

            if fork.fork_execution_id:
                try:
                    execution = bench.get_execution(fork.fork_execution_id)
                    result_dict["current_state"] = execution.current_state
                except Exception:
                    pass

            results_map[rkey] = result_dict

        except Exception as exc:
            logger.warning(
                "Fork failed for %s q=%s (depth=%d): %s",
                w.config.config_key(), w.question.question_id,
                w.reuse_depth, exc,
            )
            results_map[rkey] = {
                "status": "error",
                "error": str(exc),
                "reuse_type": "partial_hit",
            }


# ---------------------------------------------------------------------------
# Internal: Phase 3 -- full hits (cached read)
# ---------------------------------------------------------------------------

def _read_full_hits(
    items: List[WorkItem],
    bench: Any,
    ledger: ReuseLedger,
    results_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> None:
    """Read results directly from cached checkpoint state."""
    for w in items:
        rkey = (w.config.config_key(), w.question.question_id)
        entry: Optional[MaterializedEntry] = w.reuse_entry

        if entry is None:
            results_map[rkey] = {"status": "error", "error": "no cached entry",
                                 "reuse_type": "full_hit"}
            continue

        result_dict: Dict[str, Any] = {
            "status": "cached",
            "execution_id": entry.execution_id,
            "checkpoint_id": entry.checkpoint_id,
            "config_key": w.config.config_key(),
            "question_id": w.question.question_id,
            "reuse_type": "full_hit",
        }

        try:
            execution = bench.get_execution(entry.execution_id)
            result_dict["current_state"] = execution.current_state
            result_dict["metadata"] = execution.metadata
        except Exception as exc:
            logger.debug("Could not read cached state: %s", exc)

        results_map[rkey] = result_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_initial_state(config: RAGConfig, question: BenchmarkQuestion) -> Dict[str, Any]:
    """Build the LangGraph initial state dict for a (config, question) pair."""
    state: Dict[str, Any] = {
        "query": question.question,
        "messages": [],
        "count": 0,
        "result": "",
    }

    if config.frame == "longrag":
        state["test_data_name"] = _benchmark_to_test_data(question.cluster_id)
        state["query_id"] = question.question_id
        state["answers"] = []
        if "answer" in question.target:
            state["answers"] = [question.target["answer"]]
    elif config.frame == "lightrag":
        state["mode"] = "hybrid"

    if question.payload.get("context"):
        state["context"] = question.payload["context"]
    if question.payload.get("docs"):
        state["docs"] = question.payload["docs"]

    return state


def _benchmark_to_test_data(cluster_id: str) -> str:
    return {
        "hotpotqa": "hotpotqa",
        "ultradomain": "ultradomain",
        "alce": "alce",
    }.get(cluster_id, "nq")
