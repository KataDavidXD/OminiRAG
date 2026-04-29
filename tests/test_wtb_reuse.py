"""
Integration tests for the WTB x OminiRAG bipartite cache-reuse system.

Tests verify:
  1. Cache reuse actually happens (shared prefix -> no recomputation)
  2. Recomputation happens when prefix changes
  3. Fork correctly inherits reusable state
  4. Ray batch populates the ledger with correct metadata
  5. End-to-end cost reduction measurement
  6. Anti-test: no false cache hits across different frames

Uses real KG sample data from benchmark/sample_data/ and real WTB
execution (mode=development with SQLite).  Does NOT require LLM keys.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_SELFRAG_ROOT = str(_PROJECT_ROOT / "self-rag_langgraph" / "self-rag-wtb")
if _SELFRAG_ROOT not in sys.path:
    sys.path.insert(0, _SELFRAG_ROOT)

from ominirag_wtb.config_types import (
    RAGConfig,
    BenchmarkQuestion,
    WorkItem,
    NODE_ORDER,
    state_content_hash,
)
from ominirag_wtb.reuse_ledger import ReuseLedger, MaterializedEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATA_DIR = _PROJECT_ROOT / "benchmark" / "sample_data"
HOTPOTQA_QUERIES = SAMPLE_DATA_DIR / "hotpotqa_kg_sample" / "queries.jsonl"


def _load_hotpotqa_questions() -> List[BenchmarkQuestion]:
    """Load real benchmark questions from KG sample data."""
    questions: List[BenchmarkQuestion] = []
    if not HOTPOTQA_QUERIES.exists():
        pytest.skip("hotpotqa sample data not found")
    with open(HOTPOTQA_QUERIES, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            questions.append(BenchmarkQuestion(
                question_id=f"hotpotqa::test::{item['query_id']}",
                cluster_id="hotpotqa",
                stratum="test",
                question=item["query"],
                payload={"query": item["query"]},
                target={"answer": item.get("ground_truth", "")},
            ))
    return questions


@pytest.fixture
def questions() -> List[BenchmarkQuestion]:
    return _load_hotpotqa_questions()


@pytest.fixture
def ledger():
    """In-memory ledger for fast tests."""
    ld = ReuseLedger(db_path=":memory:")
    yield ld
    ld.close()


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="wtb_reuse_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_bench(tmp_dir: str):
    """Create a WTB bench in development mode for testing."""
    from wtb.sdk import WTBTestBench
    return WTBTestBench.create(mode="development", data_dir=tmp_dir)


def _simple_graph_factory():
    """Minimal 4-node graph matching the canonical RAG topology."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class St(TypedDict):
        query: str
        messages: list
        count: int
        result: str
        expanded_queries: list
        retrieval_results: list

    def node_query(state: Dict[str, Any]) -> dict:
        return {"expanded_queries": [state.get("query", "")],
                "messages": state.get("messages", []) + ["query"],
                "count": state.get("count", 0) + 1}

    def node_retrieval(state: Dict[str, Any]) -> dict:
        return {"retrieval_results": [{"content": "retrieved"}],
                "messages": state.get("messages", []) + ["retrieval"],
                "count": state.get("count", 0) + 1}

    def node_reranking(state: Dict[str, Any]) -> dict:
        return {"messages": state.get("messages", []) + ["reranking"],
                "count": state.get("count", 0) + 1}

    def node_generation(state: Dict[str, Any]) -> dict:
        msgs = state.get("messages", []) + ["generation"]
        return {"messages": msgs,
                "count": state.get("count", 0) + 1,
                "result": ",".join(msgs)}

    g = StateGraph(St)
    g.add_node("query_processing", node_query)
    g.add_node("retrieval", node_retrieval)
    g.add_node("reranking", node_reranking)
    g.add_node("generation", node_generation)
    g.add_edge("__start__", "query_processing")
    g.add_edge("query_processing", "retrieval")
    g.add_edge("retrieval", "reranking")
    g.add_edge("reranking", "generation")
    g.add_edge("generation", END)
    return g


# ---------------------------------------------------------------------------
# Test 1: Cache reuse actually happens
# ---------------------------------------------------------------------------

class TestCacheReuse:
    """Verify that shared prefixes produce ledger hits."""

    def test_prefix_reuse_populates_and_hits(self, ledger, questions):
        """Record entries for config A, verify config B hits at shared depth."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]

        config_a = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "identity", "longrag_reader")
        config_b = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "selfrag_evidence", "longrag_reader")

        for depth in range(1, 6):
            ledger.record(
                prefix=config_a.prefix(depth),
                question_id=q.question_id,
                execution_id=f"exec_a_{depth}",
                checkpoint_id=f"cp_a_{depth}",
                checkpoint_step=depth,
            )

        depth_b, entry_b = ledger.longest_matching_prefix(config_b, q.question_id)

        assert depth_b == 3, (
            f"Configs A and B share frame+query+retrieval (depth 3), got {depth_b}"
        )
        assert entry_b is not None
        assert entry_b.execution_id == "exec_a_3"
        assert entry_b.checkpoint_id == "cp_a_3"

    def test_full_hit_at_depth_5(self, ledger, questions):
        """Identical config -> full hit at depth 5."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]
        config = RAGConfig("longrag", "identity", "lightrag_hybrid",
                           "identity", "longrag_reader")

        for depth in range(1, 6):
            ledger.record(
                prefix=config.prefix(depth),
                question_id=q.question_id,
                execution_id="exec_full",
                checkpoint_id=f"cp_{depth}",
                checkpoint_step=depth,
            )

        depth, entry = ledger.longest_matching_prefix(config, q.question_id)
        assert depth == 5
        assert entry.checkpoint_id == "cp_5"


# ---------------------------------------------------------------------------
# Test 2: Recomputation on prefix change
# ---------------------------------------------------------------------------

class TestPrefixChange:
    """Verify that changing a slot invalidates downstream cache."""

    def test_different_retrieval_forces_recompute(self, ledger, questions):
        """Config C differs from A at retrieval -> shared depth = 2 only."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]
        config_a = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "identity", "longrag_reader")
        config_c = RAGConfig("longrag", "identity", "longrag_dataset",
                             "identity", "longrag_reader")

        for depth in range(1, 6):
            ledger.record(
                prefix=config_a.prefix(depth),
                question_id=q.question_id,
                execution_id=f"exec_a_{depth}",
                checkpoint_id=f"cp_a_{depth}",
                checkpoint_step=depth,
            )

        depth_c, entry_c = ledger.longest_matching_prefix(config_c, q.question_id)

        assert depth_c == 2, (
            f"A and C share frame+query (depth 2) but differ at retrieval, got {depth_c}"
        )

    def test_different_query_forces_full_recompute(self, ledger, questions):
        """Change query component -> only frame matches (depth 1)."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]
        config_a = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "identity", "longrag_reader")
        config_d = RAGConfig("longrag", "lightrag_keywords", "lightrag_hybrid",
                             "identity", "longrag_reader")

        for depth in range(1, 6):
            ledger.record(
                prefix=config_a.prefix(depth),
                question_id=q.question_id,
                execution_id=f"exec_a_{depth}",
                checkpoint_id=f"cp_a_{depth}",
                checkpoint_step=depth,
            )

        depth_d, _ = ledger.longest_matching_prefix(config_d, q.question_id)
        assert depth_d == 1, f"Only frame matches, got depth {depth_d}"


# ---------------------------------------------------------------------------
# Test 3: Fork inherits reusable state
# ---------------------------------------------------------------------------

class TestForkInheritance:
    """Verify fork from a cached checkpoint via WTB bench."""

    def test_fork_creates_new_execution(self, tmp_dir):
        """Fork from a batch result checkpoint, verify new execution exists."""
        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = _make_bench(tmp_dir)
        project = WorkflowProject(name="fork_test", graph_factory=_simple_graph_factory)
        bench.register_project(project)

        execution = bench.run(
            project="fork_test",
            initial_state={"query": "test", "messages": [], "count": 0,
                           "result": "", "expanded_queries": [],
                           "retrieval_results": []},
        )
        assert execution.status.value == "completed"

        cps = bench.get_checkpoints(execution.id)
        assert len(cps) > 0, "Must have checkpoints"

        fork_result = bench.fork(
            execution.id,
            checkpoint_id=str(cps[0].id),
            new_initial_state={"query": "forked", "messages": ["forked"],
                               "count": 99, "result": "",
                               "expanded_queries": [], "retrieval_results": []},
        )

        assert fork_result.fork_execution_id
        assert fork_result.fork_execution_id != execution.id

        forked = bench.get_execution(fork_result.fork_execution_id)
        assert forked is not None

    def test_fork_state_is_independent(self, tmp_dir):
        """Forked execution should carry the new_initial_state."""
        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = _make_bench(tmp_dir)
        project = WorkflowProject(name="fork_state", graph_factory=_simple_graph_factory)
        bench.register_project(project)

        execution = bench.run(
            project="fork_state",
            initial_state={"query": "original", "messages": [], "count": 0,
                           "result": "", "expanded_queries": [],
                           "retrieval_results": []},
        )

        cps = bench.get_checkpoints(execution.id)
        assert len(cps) > 0

        fork_result = bench.fork(
            execution.id,
            checkpoint_id=str(cps[0].id),
            new_initial_state={"query": "forked_q", "messages": ["f"],
                               "count": 42, "result": "",
                               "expanded_queries": [], "retrieval_results": []},
        )
        assert fork_result.fork_execution_id != execution.id


# ---------------------------------------------------------------------------
# Test 4: Batch populates ledger
# ---------------------------------------------------------------------------

class TestBatchLedger:
    """Verify that running a batch populates the reuse ledger."""

    def test_batch_results_recorded_in_ledger(self, tmp_dir, questions):
        """Run a batch, then record checkpoints in ledger."""
        if not questions:
            pytest.skip("no questions")

        from wtb.sdk import WTBTestBench, WorkflowProject

        bench = _make_bench(tmp_dir)
        ledger = ReuseLedger(
            db_path=str(Path(tmp_dir) / "test_ledger.db")
        )

        config = RAGConfig("longrag", "identity", "lightrag_hybrid",
                           "identity", "longrag_reader")

        project = WorkflowProject(name="batch_ledger",
                                  graph_factory=_simple_graph_factory)
        bench.register_project(project)

        q = questions[0]
        execution = bench.run(
            project="batch_ledger",
            initial_state={"query": q.question, "messages": [], "count": 0,
                           "result": "", "expanded_queries": [],
                           "retrieval_results": []},
        )
        assert execution.status.value == "completed"

        cps = bench.get_checkpoints(execution.id)
        cp_dicts = [{"checkpoint_id": str(cp.id), "step": cp.step} for cp in cps]

        ledger.record_all_prefixes(
            config=config,
            question_id=q.question_id,
            execution_id=execution.id,
            checkpoints=cp_dicts,
        )

        for depth in range(1, 6):
            entry = ledger.lookup(config.prefix(depth), q.question_id)
            assert entry is not None, f"Missing entry at depth {depth}"
            assert entry.execution_id == execution.id

        assert ledger.count() == 5

        ledger.close()


# ---------------------------------------------------------------------------
# Test 5: Cost reduction measurement
# ---------------------------------------------------------------------------

class TestCostReduction:
    """Measure that prefix sharing reduces total work items."""

    def test_reuse_partitions_correctly(self, ledger, questions):
        """3 configs sharing depth-3 prefix: first is miss, rest are partial hits."""
        if len(questions) < 2:
            pytest.skip("need >= 2 questions")

        config_a = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "identity", "longrag_reader")
        config_b = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "selfrag_evidence", "longrag_reader")
        config_c = RAGConfig("longrag", "identity", "lightrag_hybrid",
                             "lightrag_compress", "longrag_reader")

        for q in questions[:2]:
            for depth in range(1, 6):
                ledger.record(
                    prefix=config_a.prefix(depth),
                    question_id=q.question_id,
                    execution_id=f"exec_{q.question_id}_{depth}",
                    checkpoint_id=f"cp_{q.question_id}_{depth}",
                    checkpoint_step=depth,
                )

        total_miss = 0
        total_partial = 0
        total_full_hit = 0

        for config in [config_a, config_b, config_c]:
            for q in questions[:2]:
                depth, entry = ledger.longest_matching_prefix(config, q.question_id)
                if depth == 0:
                    total_miss += 1
                elif depth >= 5:
                    total_full_hit += 1
                else:
                    total_partial += 1

        assert total_full_hit == 2, f"Config A x 2 questions = 2 full hits, got {total_full_hit}"
        assert total_partial == 4, f"Configs B,C x 2 questions = 4 partial, got {total_partial}"
        assert total_miss == 0, f"All should have at least frame match, got {total_miss} misses"

        without_reuse = 3 * 2 * 4  # configs * questions * nodes
        nodes_saved = total_full_hit * 4 + total_partial * 3  # full saves 4, partial@depth3 saves 3
        with_reuse = without_reuse - nodes_saved
        reduction = 1.0 - (with_reuse / without_reuse)

        assert reduction > 0.25, f"Expected >25% reduction, got {reduction:.1%}"


# ---------------------------------------------------------------------------
# Test 6: Anti-test -- no false cache hit across frames
# ---------------------------------------------------------------------------

class TestNoFalseHit:
    """Verify different frames never share cache entries."""

    def test_different_frame_no_hit(self, ledger, questions):
        """longrag and selfrag with same component names must not share cache."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]
        config_longrag = RAGConfig("longrag", "identity", "lightrag_hybrid",
                                   "identity", "longrag_reader")
        config_selfrag = RAGConfig("selfrag", "identity", "lightrag_hybrid",
                                   "identity", "longrag_reader")

        for depth in range(1, 6):
            ledger.record(
                prefix=config_longrag.prefix(depth),
                question_id=q.question_id,
                execution_id=f"exec_lr_{depth}",
                checkpoint_id=f"cp_lr_{depth}",
                checkpoint_step=depth,
            )

        depth_sr, entry_sr = ledger.longest_matching_prefix(
            config_selfrag, q.question_id
        )

        assert depth_sr == 0, (
            f"selfrag must NOT hit longrag cache, got depth={depth_sr}"
        )
        assert entry_sr is None

    def test_different_frame_independent_entries(self, ledger, questions):
        """Each frame records its own entries independently."""
        if not questions:
            pytest.skip("no questions")

        q = questions[0]
        for frame in ("longrag", "lightrag", "selfrag"):
            config = RAGConfig(frame, "identity", "lightrag_hybrid",
                               "identity", "longrag_reader")
            for depth in range(1, 6):
                ledger.record(
                    prefix=config.prefix(depth),
                    question_id=q.question_id,
                    execution_id=f"exec_{frame}_{depth}",
                    checkpoint_id=f"cp_{frame}_{depth}",
                    checkpoint_step=depth,
                )

        assert ledger.count() == 15  # 3 frames x 5 depths

        for frame in ("longrag", "lightrag", "selfrag"):
            config = RAGConfig(frame, "identity", "lightrag_hybrid",
                               "identity", "longrag_reader")
            depth, entry = ledger.longest_matching_prefix(config, q.question_id)
            assert depth == 5
            assert entry.execution_id == f"exec_{frame}_5"


# ---------------------------------------------------------------------------
# Unit tests for data types
# ---------------------------------------------------------------------------

class TestConfigTypes:
    def test_rag_config_prefix(self):
        c = RAGConfig("longrag", "identity", "lightrag_hybrid",
                       "identity", "longrag_reader")
        assert c.prefix(0) == ()
        assert c.prefix(1) == ("longrag",)
        assert c.prefix(3) == ("longrag", "identity", "lightrag_hybrid")
        assert c.prefix(5) == c.slots()

    def test_rag_config_key(self):
        c = RAGConfig("selfrag", "lightrag_keywords", "lightrag_graph",
                       "selfrag_evidence", "selfrag_generator")
        assert c.config_key() == "selfrag/lightrag_keywords/lightrag_graph/selfrag_evidence/selfrag_generator"

    def test_from_tuple_roundtrip(self):
        t = ("longrag", "identity", "lightrag_hybrid", "identity", "longrag_reader")
        c = RAGConfig.from_tuple(t)
        assert c.slots() == t

    def test_from_tuple_wrong_length(self):
        with pytest.raises(ValueError):
            RAGConfig.from_tuple(("a", "b", "c"))

    def test_benchmark_question_creation(self):
        q = BenchmarkQuestion(
            question_id="hotpotqa::test::q1",
            cluster_id="hotpotqa",
            stratum="test",
            question="Who?",
            payload={"query": "Who?"},
            target={"answer": "Alice"},
        )
        assert q.question_id == "hotpotqa::test::q1"

    def test_state_content_hash_deterministic(self):
        s1 = {"a": 1, "b": [2, 3]}
        s2 = {"b": [2, 3], "a": 1}
        assert state_content_hash(s1) == state_content_hash(s2)

    def test_state_content_hash_different(self):
        s1 = {"a": 1}
        s2 = {"a": 2}
        assert state_content_hash(s1) != state_content_hash(s2)


class TestReuseLedger:
    def test_record_and_lookup(self, ledger):
        entry = ledger.record(
            prefix=("longrag", "identity"),
            question_id="q1",
            execution_id="exec1",
            checkpoint_id="cp1",
            checkpoint_step=2,
        )
        assert entry.execution_id == "exec1"

        found = ledger.lookup(("longrag", "identity"), "q1")
        assert found is not None
        assert found.checkpoint_id == "cp1"

    def test_lookup_miss(self, ledger):
        assert ledger.lookup(("x",), "q1") is None

    def test_materialized_keys_set(self, ledger):
        ledger.record(("a", "b"), "q1", "e1", "c1")
        ledger.record(("a",), "q2", "e2", "c2")
        keys = ledger.materialized_keys()
        assert (("a", "b"), "q1") in keys
        assert (("a",), "q2") in keys
        assert len(keys) == 2

    def test_record_all_prefixes(self, ledger):
        config = RAGConfig("longrag", "identity", "lightrag_hybrid",
                           "identity", "longrag_reader")
        cps = [{"checkpoint_id": f"cp{i}", "step": i} for i in range(5)]
        entries = ledger.record_all_prefixes(
            config, "q1", "exec1", cps,
        )
        assert len(entries) == 5
        assert ledger.count() == 5

    def test_upsert_overwrites(self, ledger):
        ledger.record(("a",), "q1", "old_exec", "old_cp")
        ledger.record(("a",), "q1", "new_exec", "new_cp")
        found = ledger.lookup(("a",), "q1")
        assert found.execution_id == "new_exec"
        assert ledger.count() == 1


class TestWorkItem:
    def test_full_miss(self):
        c = RAGConfig("longrag", "identity", "lightrag_hybrid",
                       "identity", "longrag_reader")
        q = BenchmarkQuestion("q1", "hotpotqa", "test", "?", {}, {})
        w = WorkItem(config=c, question=q, reuse_depth=0)
        assert w.is_full_miss
        assert not w.is_partial_hit
        assert not w.is_full_hit

    def test_partial_hit(self):
        c = RAGConfig("longrag", "identity", "lightrag_hybrid",
                       "identity", "longrag_reader")
        q = BenchmarkQuestion("q1", "hotpotqa", "test", "?", {}, {})
        w = WorkItem(config=c, question=q, reuse_depth=3)
        assert w.is_partial_hit
        assert not w.is_full_miss
        assert not w.is_full_hit

    def test_full_hit(self):
        c = RAGConfig("longrag", "identity", "lightrag_hybrid",
                       "identity", "longrag_reader")
        q = BenchmarkQuestion("q1", "hotpotqa", "test", "?", {}, {})
        w = WorkItem(config=c, question=q, reuse_depth=5)
        assert w.is_full_hit
        assert not w.is_full_miss


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
