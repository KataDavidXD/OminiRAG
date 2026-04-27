"""
WTB Installation Checker & Smoke Test.

Validates that the ``wtb`` package is correctly installed and that
core SDK operations work end-to-end.  Tests are grouped into tiers:

  Tier 1 (always): import, bench creation, run, checkpoint, rollback, fork, batch
  Tier 2 (if ray): Ray-distributed batch execution
  Tier 3 (if grpc): GrpcEnvironmentProvider (venv service connectivity)

Usage:
    python -m examples.quick_start.install_checker
    python -m examples.quick_start.install_checker --skip-ray
    python -m examples.quick_start.install_checker --grpc-url localhost:50051

Exit code 0 = all attempted checks passed, 1 = at least one failure.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: List[Tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = f"[{status}]"
    msg = f"  {tag:8s} {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Graph factories (SDK-only, minimal LangGraph)
# ---------------------------------------------------------------------------

def _create_linear_graph():
    """3-node linear graph: A -> B -> C."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class St(TypedDict):
        messages: list
        count: int
        result: str

    def node_a(state: Dict[str, Any]) -> dict:
        return {"messages": state.get("messages", []) + ["A"],
                "count": state.get("count", 0) + 1}

    def node_b(state: Dict[str, Any]) -> dict:
        return {"messages": state.get("messages", []) + ["B"],
                "count": state.get("count", 0) + 1}

    def node_c(state: Dict[str, Any]) -> dict:
        msgs = state.get("messages", []) + ["C"]
        return {"messages": msgs, "count": state.get("count", 0) + 1,
                "result": ",".join(msgs)}

    g = StateGraph(St)
    g.add_node("node_a", node_a)
    g.add_node("node_b", node_b)
    g.add_node("node_c", node_c)
    g.add_edge("__start__", "node_a")
    g.add_edge("node_a", "node_b")
    g.add_edge("node_b", "node_c")
    g.add_edge("node_c", END)
    return g


_INIT_STATE: Dict[str, Any] = {"messages": [], "count": 0, "result": ""}


# ---------------------------------------------------------------------------
# Deterministic WTB cache probes
# ---------------------------------------------------------------------------

class _CacheCheckChatModel:
    """Tiny fake LangChain chat model used only on cache misses."""

    def __init__(self, response_prefix: str):
        self.response_prefix = response_prefix
        self.invoke_count = 0

    def invoke(self, messages: list) -> Any:
        self.invoke_count += 1
        rendered = "|".join(str(message) for message in messages)
        return SimpleNamespace(content=f"{self.response_prefix}:{rendered}")


def _probe_ominirag_cache(
    cache_path: str,
    prompt: str,
    response_prefix: str,
    *,
    system_name: str = "checker",
    node_path: str = "checker.llm",
) -> Dict[str, Any]:
    """Run one text generation through OminiRAG's shared WTB cache helper."""
    from rag_contracts import WTBCacheConfig, WTBCachedLLM
    from wtb.infrastructure.llm import reset_service_cache

    reset_service_cache()
    config = WTBCacheConfig(
        enabled=True,
        cache_path=cache_path,
        api_key="cache-check",
        base_url="https://cache-check.invalid/v1",
        text_model="cache-check-text",
        embedding_model="cache-check-embedding",
    )
    llm = WTBCachedLLM(
        config=config,
        system_name=system_name,
        node_path=node_path,
    )
    fake_model = _CacheCheckChatModel(response_prefix=response_prefix)
    llm.service.get_chat_model = lambda **_kwargs: fake_model

    text = llm.complete(
        "cache-check-system",
        prompt,
        model="cache-check-text",
        temperature=0.0,
        max_tokens=32,
    )
    metadata = llm.wtb_cache_metadata()
    stats = llm.cache_stats()
    reset_service_cache()
    return {
        "cache_hit": metadata.get("last_cache_hit"),
        "cache_key": metadata.get("last_cache_key"),
        "text": text,
        "fake_invocations": fake_model.invoke_count,
        "metadata": metadata,
        "stats": stats,
    }


def _ray_cache_probe(cache_path: str, prompt: str, response_prefix: str) -> Dict[str, Any]:
    return _probe_ominirag_cache(cache_path, prompt, response_prefix)


def _record_invariants(name: str, failures: List[str], detail: str) -> None:
    if failures:
        record(name, FAIL, "; ".join(failures))
    else:
        record(name, PASS, detail)


def _expect(condition: bool, failures: List[str], message: str) -> None:
    if not condition:
        failures.append(message)


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1 -- always runs (in-memory, zero external deps beyond langgraph)
# ═══════════════════════════════════════════════════════════════════════════════

def check_import() -> None:
    try:
        import wtb
        record("import wtb", PASS, f"version {wtb.__version__}")
    except Exception as exc:
        record("import wtb", FAIL, str(exc))


def check_sdk_imports() -> None:
    try:
        from wtb.sdk import (  # noqa: F401
            WTBTestBench, WorkflowProject,
            FileTrackingConfig, EnvironmentConfig, ExecutionConfig,
            EnvSpec, RayConfig, NodeResourceConfig,
            WorkspaceIsolationConfig, PauseStrategyConfig,
            RollbackResult, ForkResult,
            BatchRollbackResult, BatchForkResult,
        )
        record("sdk imports", PASS, "14 symbols")
    except Exception as exc:
        record("sdk imports", FAIL, str(exc))


def check_create_bench() -> None:
    try:
        from wtb.sdk import WTBTestBench
        bench = WTBTestBench.create(mode="testing")
        assert bench is not None
        record("create bench", PASS, "mode=testing")
    except Exception as exc:
        record("create bench", FAIL, str(exc))


def check_run_workflow() -> Optional[Any]:
    """Run a workflow end-to-end and return (bench, execution) for later checks."""
    try:
        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = WTBTestBench.create(mode="testing")
        project = WorkflowProject(name="smoke", graph_factory=_create_linear_graph)
        bench.register_project(project)

        execution = bench.run(project="smoke", initial_state=dict(_INIT_STATE))
        assert execution.status.value == "completed", f"status={execution.status}"
        record("run workflow", PASS, f"status={execution.status.value}")
        return bench, execution
    except Exception as exc:
        record("run workflow", FAIL, str(exc))
        traceback.print_exc()
        return None


def check_checkpoints(ctx: Optional[Any]) -> Optional[list]:
    if ctx is None:
        record("checkpoints", SKIP, "workflow run failed")
        return None
    bench, execution = ctx
    try:
        cps = bench.get_checkpoints(execution.id)
        record("checkpoints", PASS,
               f"time_travel={bench.supports_time_travel()}, count={len(cps)}")
        return cps
    except Exception as exc:
        record("checkpoints", FAIL, str(exc))
        return None


def check_rollback(ctx: Optional[Any], cps: Optional[list]) -> None:
    if ctx is None or not cps:
        record("rollback", SKIP, "no context/checkpoints")
        return
    bench, execution = ctx
    try:
        cp_id = str(cps[0].id)
        result = bench.rollback(execution.id, checkpoint_id=cp_id)
        assert result.success, f"rollback error: {result.error}"
        record("rollback", PASS, f"to checkpoint step={cps[0].step}")
    except Exception as exc:
        record("rollback", FAIL, str(exc))


def check_fork(ctx: Optional[Any], cps: Optional[list]) -> None:
    if ctx is None or not cps:
        record("fork", SKIP, "no context/checkpoints")
        return
    bench, execution = ctx
    try:
        cp_id = str(cps[0].id)
        fork_result = bench.fork(
            execution.id,
            checkpoint_id=cp_id,
            new_initial_state={"messages": ["forked"], "count": 99, "result": ""},
        )
        assert fork_result.fork_execution_id, "fork_execution_id is empty"
        assert fork_result.fork_execution_id != execution.id
        record("fork", PASS, f"new_exec={fork_result.fork_execution_id[:12]}...")
    except Exception as exc:
        record("fork", FAIL, str(exc))


def check_batch_sequential() -> None:
    """Batch test via sequential fallback (no Ray)."""
    try:
        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = WTBTestBench.create(mode="testing")
        project = WorkflowProject(name="batch_seq", graph_factory=_create_linear_graph)
        bench.register_project(project)

        batch = bench.run_batch_test(
            project="batch_seq",
            variant_matrix=[
                {"node_b": "default"},
                {"node_b": "alt"},
            ],
            test_cases=[
                dict(_INIT_STATE),
                {"messages": ["x"], "count": 5, "result": ""},
            ],
        )
        ok = sum(1 for r in batch.results if r.success)
        record("batch sequential", PASS,
               f"variants=2, cases=2, passed={ok}/{len(batch.results)}")
    except Exception as exc:
        record("batch sequential", FAIL, str(exc))


def check_batch_rollback_and_fork() -> None:
    """Rollback and fork batch results via the SDK convenience API.

    Uses SQLite (development mode) because batch rollback/fork requires
    shared persistent storage -- the BatchExecutionCoordinator creates its
    own controller/UoW, and in-memory UoWs are not shared across instances.
    """
    import tempfile, os

    tmp = tempfile.mkdtemp(prefix="wtb_check_")
    try:
        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = WTBTestBench.create(mode="development", data_dir=tmp)
        project = WorkflowProject(name="br_test", graph_factory=_create_linear_graph)
        bench.register_project(project)

        batch = bench.run_batch_test(
            project="br_test",
            variant_matrix=[{"node_b": "default"}],
            test_cases=[dict(_INIT_STATE)],
        )
        result = batch.results[0]
        if not result.execution_id:
            record("batch rollback", SKIP, "no execution_id")
            record("batch fork", SKIP, "no execution_id")
            return

        cps = bench.get_batch_result_checkpoints(result)
        if not cps:
            record("batch rollback", SKIP, "no checkpoints")
            record("batch fork", SKIP, "no checkpoints")
            return

        rb = bench.rollback_batch_result(result, checkpoint_id=str(cps[0].id))
        assert rb.success, f"rollback error: {rb.error}"
        record("batch rollback", PASS, f"to step={cps[0].step}")

        fork = bench.fork_batch_result(
            result,
            checkpoint_id=str(cps[0].id),
            new_state={"messages": ["forked"], "count": 42, "result": ""},
        )
        assert fork.fork_execution_id, f"fork error: {fork.error}"
        record("batch fork", PASS,
               f"forked {fork.fork_execution_id[:12]}... from step={cps[0].step}")

        bench.close()
    except Exception as exc:
        record("batch rollback", FAIL, str(exc))
        record("batch fork", SKIP, "blocked by rollback failure")
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cache_repeated_runs() -> None:
    """Verify a second service/run reuses the persistent WTB cache."""
    tmp = tempfile.mkdtemp(prefix="wtb_cache_repeat_")
    try:
        cache_path = str(Path(tmp) / "llm_response_cache.db")
        first = _probe_ominirag_cache(cache_path, "repeatable prompt", "first-run")
        second = _probe_ominirag_cache(cache_path, "repeatable prompt", "second-run")

        failures: List[str] = []
        _expect(not first["cache_hit"], failures, "first run unexpectedly reported cache_hit=True")
        _expect(first["fake_invocations"] == 1, failures,
                f"first run fake model calls={first['fake_invocations']} expected 1")
        _expect(second["cache_hit"], failures, "second run did not report cache_hit=True")
        _expect(second["fake_invocations"] == 0, failures,
                f"second run fake model calls={second['fake_invocations']} expected 0")
        _expect(first["cache_key"] == second["cache_key"], failures,
                "cache_key changed across identical repeated runs")
        _expect(first["text"] == second["text"], failures,
                "second run did not return the cached first-run response text")
        _expect(second["stats"]["entries"] == 1, failures,
                f"cache entries after second run={second['stats']['entries']} expected 1")

        _record_invariants(
            "cache repeated runs",
            failures,
            f"key={second['cache_key'][:12]}..., entries={second['stats']['entries']}",
        )
    except Exception as exc:
        record("cache repeated runs", FAIL, str(exc))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cache_rollback_and_fork() -> None:
    """Validate rollback/fork-style cache reuse and cache-path isolation."""
    tmp = tempfile.mkdtemp(prefix="wtb_cache_branch_")
    try:
        shared_cache_path = str(Path(tmp) / "shared" / "llm_response_cache.db")
        isolated_cache_path = str(Path(tmp) / "isolated" / "llm_response_cache.db")

        base = _probe_ominirag_cache(shared_cache_path, "branch base prompt", "base")
        rollback = _probe_ominirag_cache(shared_cache_path, "branch base prompt", "rollback")
        fork_same = _probe_ominirag_cache(shared_cache_path, "branch base prompt", "fork-same")
        fork_changed = _probe_ominirag_cache(shared_cache_path, "branch changed prompt", "fork-changed")
        isolated = _probe_ominirag_cache(isolated_cache_path, "branch base prompt", "isolated")

        failures: List[str] = []
        _expect(not base["cache_hit"], failures, "base run unexpectedly hit cache")
        _expect(rollback["cache_hit"], failures, "rollback-style rerun did not reuse cache")
        _expect(rollback["fake_invocations"] == 0, failures,
                f"rollback fake model calls={rollback['fake_invocations']} expected 0")
        _expect(fork_same["cache_hit"], failures,
                "fork with unchanged prompt/shared cache did not reuse cache")
        _expect(fork_same["fake_invocations"] == 0, failures,
                f"fork same-prompt fake model calls={fork_same['fake_invocations']} expected 0")
        _expect(not fork_changed["cache_hit"], failures,
                "fork with changed prompt unexpectedly reused the base cache entry")
        _expect(fork_changed["fake_invocations"] == 1, failures,
                f"fork changed-prompt fake model calls={fork_changed['fake_invocations']} expected 1")
        _expect(base["cache_key"] != fork_changed["cache_key"], failures,
                "changed fork prompt produced the same cache_key as the base prompt")
        _expect(fork_changed["stats"]["entries"] == 2, failures,
                f"shared cache entries after changed fork={fork_changed['stats']['entries']} expected 2")
        _expect(not isolated["cache_hit"], failures,
                "isolated cache path unexpectedly reused the shared cache entry")
        _expect(isolated["cache_key"] == base["cache_key"], failures,
                "isolated path changed request cache_key for identical prompt")
        _expect(isolated["fake_invocations"] == 1, failures,
                f"isolated cache fake model calls={isolated['fake_invocations']} expected 1")
        _expect(isolated["stats"]["entries"] == 1, failures,
                f"isolated cache entries={isolated['stats']['entries']} expected 1")

        _record_invariants(
            "cache rollback/fork",
            failures,
            "rollback_hit=True, shared_fork_hit=True, changed_fork_miss=True, isolated_path_miss=True",
        )
    except Exception as exc:
        record("cache rollback/fork", FAIL, str(exc))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_ominirag_cache_helpers() -> None:
    """Validate OminiRAG helper metadata and optional system status reporting."""
    try:
        from rag_contracts import (
            WTBCacheConfig,
            inspect_swappable_system_cache_support,
        )

        config = WTBCacheConfig(
            enabled=True,
            cache_path="/tmp/ominirag-wtb-check/llm_response_cache.db",
            api_key="cache-check",
            base_url="https://cache-check.invalid/v1",
            text_model="cache-check-text",
            embedding_model="cache-check-embedding",
        )
        env = config.as_env()
        statuses = inspect_swappable_system_cache_support()

        failures: List[str] = []
        _expect(env["RAG_USE_WTB_CACHE"] == "true", failures,
                "RAG_USE_WTB_CACHE env value was not true")
        _expect(env["WTB_LLM_CACHE_PATH"].endswith("llm_response_cache.db"), failures,
                "WTB_LLM_CACHE_PATH missing from standardized env")
        _expect({s.name for s in statuses} == {"longrag", "storm", "selfrag"}, failures,
                f"unexpected system status names: {[s.name for s in statuses]}")
        _expect(all(s.cache_surface for s in statuses), failures,
                "one or more system statuses lacked cache_surface")

        ready = [s.name for s in statuses if s.importable]
        blocked = [s.name for s in statuses if not s.importable]
        _record_invariants(
            "ominirag cache helpers",
            failures,
            f"env ok; importable={ready or 'none'}; blocked={blocked or 'none'}",
        )

        for status in statuses:
            record(
                f"{status.name} cache surface",
                PASS if status.importable else SKIP,
                status.detail,
            )
    except Exception as exc:
        record("ominirag cache helpers", FAIL, str(exc))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2 -- Ray distributed batch
# ═══════════════════════════════════════════════════════════════════════════════

def check_ray_batch(skip: bool = False) -> None:
    if skip:
        record("ray batch", SKIP, "skipped via --skip-ray")
        return

    try:
        import ray
    except ImportError:
        record("ray batch", SKIP, "ray not installed")
        return

    import tempfile, shutil

    tmp = tempfile.mkdtemp(prefix="wtb_ray_")
    try:
        from wtb.sdk import WTBTestBench, WorkflowProject, ExecutionConfig, RayConfig

        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True, log_to_driver=False)

        bench = WTBTestBench.create(
            mode="development",
            data_dir=tmp,
            enable_ray=True,
        )
        project = WorkflowProject(
            name="ray_smoke",
            graph_factory=_create_linear_graph,
            execution=ExecutionConfig(
                batch_executor="ray",
                ray_config=RayConfig(address="auto", max_retries=1),
            ),
        )
        bench.register_project(project)

        t0 = time.time()
        batch = bench.run_batch_test(
            project=project.name,
            variant_matrix=[{"node_b": "v0"}, {"node_b": "v1"}],
            test_cases=[dict(_INIT_STATE)],
        )
        elapsed = time.time() - t0
        ok = sum(1 for r in batch.results if r.success)
        record("ray batch", PASS,
               f"results={len(batch.results)}, passed={ok}, {elapsed:.1f}s")
        bench.close()
    except Exception as exc:
        record("ray batch", FAIL, str(exc))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_cache_ray(skip: bool = False) -> None:
    if skip:
        record("cache ray reuse", SKIP, "skipped via --skip-ray")
        return

    try:
        import ray
    except ImportError:
        record("cache ray reuse", SKIP, "ray not installed")
        return

    tmp = tempfile.mkdtemp(prefix="wtb_cache_ray_")
    try:
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True, log_to_driver=False)

        cache_path = str(Path(tmp) / "llm_response_cache.db")
        remote_probe = ray.remote(_ray_cache_probe)
        first = ray.get(remote_probe.remote(cache_path, "ray shared prompt", "ray-first"))
        second = ray.get(remote_probe.remote(cache_path, "ray shared prompt", "ray-second"))

        failures: List[str] = []
        _expect(not first["cache_hit"], failures, "first Ray task unexpectedly hit cache")
        _expect(first["fake_invocations"] == 1, failures,
                f"first Ray task fake model calls={first['fake_invocations']} expected 1")
        _expect(second["cache_hit"], failures,
                "second Ray task did not reuse cache from the first Ray task")
        _expect(second["fake_invocations"] == 0, failures,
                f"second Ray task fake model calls={second['fake_invocations']} expected 0")
        _expect(first["cache_key"] == second["cache_key"], failures,
                "Ray cache_key changed across identical task inputs")
        _expect(first["text"] == second["text"], failures,
                "second Ray task did not return cached first-task response text")
        _expect(second["stats"]["entries"] == 1, failures,
                f"Ray cache entries after second task={second['stats']['entries']} expected 1")

        _record_invariants(
            "cache ray reuse",
            failures,
            f"key={second['cache_key'][:12]}..., entries={second['stats']['entries']}",
        )
    except Exception as exc:
        record("cache ray reuse", FAIL, str(exc))
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)




# ═══════════════════════════════════════════════════════════════════════════════
# Tier 3 -- Venv service (GrpcEnvironmentProvider)
# ═══════════════════════════════════════════════════════════════════════════════

def check_venv_provider(grpc_url: Optional[str]) -> None:
    if grpc_url is None:
        record("venv provider", SKIP, "no --grpc-url provided")
        return

    try:
        import grpc  # noqa: F401
    except ImportError:
        record("venv provider", SKIP, "grpcio not installed")
        return

    try:
        from wtb.infrastructure.environment.providers import GrpcEnvironmentProvider

        provider = GrpcEnvironmentProvider(grpc_address=grpc_url)

        env = provider.create_environment("smoke-variant", {
            "workflow_id": "install_check",
            "node_id": "smoke_node",
            "packages": ["requests"],
            "python_version": "3.12",
        })

        assert env.get("env_path") or env.get("type"), f"unexpected env: {env}"
        record("venv create", PASS,
               f"type={env.get('type')}, path={env.get('env_path', 'n/a')}")

        rt = provider.get_runtime_env("smoke-variant")
        has_path = bool(rt and (rt.get("python_path") or rt.get("py_executable")))
        record("venv runtime_env", PASS if has_path else FAIL,
               f"python_path={'found' if has_path else 'missing'}")

        provider.cleanup_environment("smoke-variant")
        record("venv cleanup", PASS)

        provider.close()
    except Exception as exc:
        record("venv provider", FAIL, str(exc))
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="WTB Installation Checker")
    parser.add_argument("--skip-ray", action="store_true",
                        help="Skip Ray-dependent checks")
    parser.add_argument("--grpc-url", type=str, default=None,
                        help="UV Venv Manager gRPC address (e.g. localhost:50051)")
    args = parser.parse_args()

    print("=" * 64)
    print("  WTB Installation Checker")
    print("=" * 64)

    # ── Tier 1: Core (always) ────────────────────────────────────────────
    print("\n  --- Tier 1: Core SDK ---\n")
    check_import()
    check_sdk_imports()
    check_create_bench()
    ctx = check_run_workflow()
    cps = check_checkpoints(ctx)
    check_rollback(ctx, cps)
    check_fork(ctx, cps)
    check_batch_sequential()
    check_batch_rollback_and_fork()

    # ── Cache behavior ───────────────────────────────────────────────────
    print("\n  --- Tier 1b: WTB Cache Behavior ---\n")
    check_cache_repeated_runs()
    check_cache_rollback_and_fork()
    check_ominirag_cache_helpers()

    # ── Tier 2: Ray ──────────────────────────────────────────────────────
    print("\n  --- Tier 2: Ray Distributed ---\n")
    check_ray_batch(skip=args.skip_ray)
    check_cache_ray(skip=args.skip_ray)

    # ── Tier 3: Venv Service ─────────────────────────────────────────────
    print("\n  --- Tier 3: Venv Service ---\n")
    check_venv_provider(args.grpc_url)

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("-" * 64)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    skipped = sum(1 for _, s, _ in results if s == SKIP)
    print(f"  Total: {len(results)}  |  Passed: {passed}"
          f"  |  Failed: {failed}  |  Skipped: {skipped}")
    print("-" * 64)

    if failed:
        print("\n  Failure details:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail or 'no detail'}")
        print("\n  Some checks FAILED. See details above.\n")
        return 1

    print("\n  All attempted checks PASSED.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
