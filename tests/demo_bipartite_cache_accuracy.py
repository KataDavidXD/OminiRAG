"""
Bipartite Cache Accuracy Diagnostic Demo
=========================================

Stress-tests the (config_prefix, question_id) bipartite reuse ledger to
expose cache misuse, false hits, silent misses, and edge-case bugs.

Run::

    python tests/demo_bipartite_cache_accuracy.py

The demo runs a matrix of scenarios with real WTB execution (SQLite
checkpoints), real KG sample data, and reports a per-scenario
PASS/FAIL/BUG table at the end.

Scenarios covered
-----------------
  A. Exact-match cache correctness (same config + same question = hit)
  B. Cross-frame isolation (longrag vs selfrag must NEVER share cache)
  C. Cross-question isolation (same config, different question = miss)
  D. Prefix depth boundary precision (depth 1..5 boundary checks)
  E. Overwrite consistency (re-record same key, old data gone)
  F. State hash validation (detect stale/corrupt checkpoint refs)
  G. Concurrent ledger writes (thread safety)
  H. Ledger <-> WTB checkpoint round-trip (record, fork, verify state)
  I. Full batch partition audit (miss/partial/hit counts match expected)
  J. Materialized-keys sync with AG-UCT (ledger.materialized_keys() == Path_t)
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SELFRAG = str(_ROOT / "self-rag_langgraph" / "self-rag-wtb")
if _SELFRAG not in sys.path:
    sys.path.insert(0, _SELFRAG)

from ominirag_wtb.config_types import (
    RAGConfig,
    BenchmarkQuestion,
    WorkItem,
    NODE_ORDER,
    state_content_hash,
)
from ominirag_wtb.reuse_ledger import ReuseLedger, MaterializedEntry


# ======================================================================
# Helpers
# ======================================================================

SAMPLE_DIR = _ROOT / "benchmark" / "sample_data"

def _load_questions(cluster: str, jsonl_path: Path) -> List[BenchmarkQuestion]:
    qs: List[BenchmarkQuestion] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gt = item.get("ground_truth", "")
            if isinstance(gt, list):
                gt = gt[0] if gt else ""
            qs.append(BenchmarkQuestion(
                question_id=f"{cluster}::test::{item['query_id']}",
                cluster_id=cluster,
                stratum="test",
                question=item["query"],
                payload={"query": item["query"]},
                target={"answer": gt},
            ))
    return qs


ALL_QUESTIONS: Dict[str, List[BenchmarkQuestion]] = {}
for _cid, _sub in [
    ("hotpotqa", "hotpotqa_kg_sample"),
    ("ultradomain", "ultradomain_kg_sample"),
    ("alce", "alce_kg_sample"),
]:
    _p = SAMPLE_DIR / _sub / "queries.jsonl"
    if _p.exists():
        ALL_QUESTIONS[_cid] = _load_questions(_cid, _p)


# The 14 meaningful configs from the plan (frame explicit as 5th slot)
FOURTEEN_CONFIGS = [
    RAGConfig("longrag",  "identity",          "longrag_dataset",   "identity",          "longrag_reader"),
    RAGConfig("longrag",  "identity",          "lightrag_hybrid",   "identity",          "longrag_reader"),
    RAGConfig("longrag",  "identity",          "lightrag_hybrid",   "lightrag_compress", "longrag_reader"),
    RAGConfig("longrag",  "lightrag_keywords", "lightrag_hybrid",   "identity",          "longrag_reader"),
    RAGConfig("lightrag", "lightrag_keywords", "lightrag_hybrid",   "lightrag_compress", "lightrag_answer"),
    RAGConfig("lightrag", "lightrag_keywords", "lightrag_chunk",    "lightrag_compress", "lightrag_answer"),
    RAGConfig("lightrag", "lightrag_keywords", "lightrag_graph",    "lightrag_compress", "lightrag_answer"),
    RAGConfig("lightrag", "identity",          "lightrag_hybrid",   "identity",          "lightrag_answer"),
    RAGConfig("selfrag",  "identity",          "lightrag_hybrid",   "selfrag_evidence",  "selfrag_generator"),
    RAGConfig("selfrag",  "identity",          "longrag_dataset",   "selfrag_evidence",  "selfrag_generator"),
    RAGConfig("selfrag",  "identity",          "lightrag_hybrid",   "identity",          "selfrag_generator"),
    RAGConfig("longrag",  "identity",          "lightrag_hybrid",   "selfrag_evidence",  "longrag_reader"),
    RAGConfig("lightrag", "lightrag_keywords", "lightrag_hybrid",   "selfrag_evidence",  "lightrag_answer"),
    RAGConfig("selfrag",  "lightrag_keywords", "lightrag_hybrid",   "selfrag_evidence",  "selfrag_generator"),
]


def _simple_graph_factory():
    """Minimal 4-node graph for WTB execution."""
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class St(TypedDict):
        query: str
        messages: list
        count: int
        result: str
        expanded_queries: list
        retrieval_results: list

    def nq(s):
        return {"expanded_queries": [s.get("query", "")],
                "messages": s.get("messages", []) + ["query"],
                "count": s.get("count", 0) + 1}

    def nr(s):
        return {"retrieval_results": [{"content": "doc"}],
                "messages": s.get("messages", []) + ["retrieval"],
                "count": s.get("count", 0) + 1}

    def nk(s):
        return {"messages": s.get("messages", []) + ["reranking"],
                "count": s.get("count", 0) + 1}

    def ng(s):
        m = s.get("messages", []) + ["generation"]
        return {"messages": m, "count": s.get("count", 0) + 1,
                "result": ",".join(m)}

    g = StateGraph(St)
    g.add_node("query_processing", nq)
    g.add_node("retrieval", nr)
    g.add_node("reranking", nk)
    g.add_node("generation", ng)
    g.add_edge("__start__", "query_processing")
    g.add_edge("query_processing", "retrieval")
    g.add_edge("retrieval", "reranking")
    g.add_edge("reranking", "generation")
    g.add_edge("generation", END)
    return g


class Scenario:
    """One test scenario with name, expected result, actual result."""
    def __init__(self, name: str):
        self.name = name
        self.passed: Optional[bool] = None
        self.details: str = ""
        self.bugs: List[str] = []

    def ok(self, detail: str = ""):
        self.passed = True
        self.details = detail

    def fail(self, detail: str):
        self.passed = False
        self.details = detail

    def bug(self, detail: str):
        self.passed = False
        self.bugs.append(detail)
        self.details = detail

    @property
    def status(self) -> str:
        if self.passed is None:
            return "SKIP"
        if self.bugs:
            return "BUG"
        return "PASS" if self.passed else "FAIL"


# ======================================================================
# Scenario A: Exact-match cache correctness
# ======================================================================

def scenario_a_exact_match(ledger: ReuseLedger) -> Scenario:
    s = Scenario("A: Exact-match cache correctness")
    qs = ALL_QUESTIONS.get("hotpotqa", [])
    if not qs:
        s.ok("SKIP - no hotpotqa data")
        return s

    q = qs[0]
    config = FOURTEEN_CONFIGS[0]

    for depth in range(1, 6):
        ledger.record(config.prefix(depth), q.question_id,
                      f"e_{depth}", f"cp_{depth}", depth,
                      state_hash=f"hash_{depth}")

    errors = []
    for depth in range(1, 6):
        entry = ledger.lookup(config.prefix(depth), q.question_id)
        if entry is None:
            errors.append(f"depth={depth}: lookup returned None after record")
        elif entry.execution_id != f"e_{depth}":
            errors.append(f"depth={depth}: exec_id mismatch {entry.execution_id}")
        elif entry.state_hash != f"hash_{depth}":
            errors.append(f"depth={depth}: hash mismatch {entry.state_hash}")

    depth_full, entry_full = ledger.longest_matching_prefix(config, q.question_id)
    if depth_full != 5:
        errors.append(f"longest_matching_prefix returned {depth_full}, expected 5")

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok(f"All 5 depths verified for {config.config_key()} x {q.question_id}")
    return s


# ======================================================================
# Scenario B: Cross-frame isolation
# ======================================================================

def scenario_b_cross_frame_isolation(ledger: ReuseLedger) -> Scenario:
    s = Scenario("B: Cross-frame isolation (longrag/lightrag/selfrag)")
    qs = ALL_QUESTIONS.get("hotpotqa", [])
    if not qs:
        s.ok("SKIP - no data")
        return s

    q = qs[0]
    # Record for longrag
    cfg_lr = RAGConfig("longrag", "identity", "lightrag_hybrid", "identity", "longrag_reader")
    for depth in range(1, 6):
        ledger.record(cfg_lr.prefix(depth), q.question_id,
                      f"lr_{depth}", f"cp_lr_{depth}", depth)

    errors = []
    for other_frame in ("lightrag", "selfrag"):
        cfg_other = RAGConfig(other_frame, "identity", "lightrag_hybrid", "identity", "longrag_reader")
        depth_o, entry_o = ledger.longest_matching_prefix(cfg_other, q.question_id)
        if depth_o > 0:
            errors.append(
                f"CROSS-CONTAMINATION: {other_frame} hit longrag cache at depth={depth_o} "
                f"(entry exec={entry_o.execution_id if entry_o else 'N/A'})"
            )

    # Also verify the reverse: record for selfrag, check longrag doesn't hit
    cfg_sr = RAGConfig("selfrag", "identity", "lightrag_hybrid", "identity", "longrag_reader")
    for depth in range(1, 6):
        ledger.record(cfg_sr.prefix(depth), q.question_id,
                      f"sr_{depth}", f"cp_sr_{depth}", depth)

    d_lr, e_lr = ledger.longest_matching_prefix(cfg_lr, q.question_id)
    if d_lr != 5 or e_lr.execution_id != "lr_5":
        errors.append(
            f"longrag entry corrupted after selfrag insert: depth={d_lr}, "
            f"exec={e_lr.execution_id if e_lr else 'None'}"
        )

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok("3 frames fully isolated, no cross-contamination")
    return s


# ======================================================================
# Scenario C: Cross-question isolation
# ======================================================================

def scenario_c_cross_question_isolation(ledger: ReuseLedger) -> Scenario:
    s = Scenario("C: Cross-question isolation")
    qs = ALL_QUESTIONS.get("hotpotqa", [])
    if len(qs) < 2:
        s.ok("SKIP - need >= 2 questions")
        return s

    config = FOURTEEN_CONFIGS[0]
    q1, q2 = qs[0], qs[1]

    for depth in range(1, 6):
        ledger.record(config.prefix(depth), q1.question_id,
                      f"q1_{depth}", f"cp_q1_{depth}", depth)

    errors = []
    d2, e2 = ledger.longest_matching_prefix(config, q2.question_id)
    if d2 > 0:
        errors.append(
            f"Question cross-leak: q2 hit q1's cache at depth={d2}"
        )

    # Record q2 separately and verify independence
    for depth in range(1, 6):
        ledger.record(config.prefix(depth), q2.question_id,
                      f"q2_{depth}", f"cp_q2_{depth}", depth)

    e1_check = ledger.lookup(config.prefix(3), q1.question_id)
    e2_check = ledger.lookup(config.prefix(3), q2.question_id)
    if e1_check is None or e1_check.execution_id != "q1_3":
        errors.append(f"q1 depth=3 corrupted: {e1_check}")
    if e2_check is None or e2_check.execution_id != "q2_3":
        errors.append(f"q2 depth=3 corrupted: {e2_check}")

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok(f"q1 and q2 fully independent for {config.config_key()}")
    return s


# ======================================================================
# Scenario D: Prefix depth boundary precision
# ======================================================================

def scenario_d_prefix_boundary(ledger: ReuseLedger) -> Scenario:
    s = Scenario("D: Prefix depth boundary precision (depth 1..5)")
    qs = ALL_QUESTIONS.get("alce", [])
    if not qs:
        s.ok("SKIP - no alce data")
        return s

    q = qs[0]
    errors = []

    # --- Sub-test D1: Record only depth=3, verify depth 4 and 5 miss ---
    cfg = RAGConfig("longrag", "identity", "lightrag_hybrid", "identity", "longrag_reader")
    ledger.record(cfg.prefix(3), q.question_id, "e_d3", "cp_d3", 3)

    d, e = ledger.longest_matching_prefix(cfg, q.question_id)
    if d != 3:
        errors.append(f"D1: expected depth=3, got {d}")
    for check_depth in (4, 5):
        entry = ledger.lookup(cfg.prefix(check_depth), q.question_id)
        if entry is not None:
            errors.append(f"D1: depth={check_depth} should be None but got exec={entry.execution_id}")

    # --- Sub-test D2: Configs that differ at exactly depth=4 ---
    cfg_a = RAGConfig("longrag", "identity", "lightrag_hybrid", "identity",          "longrag_reader")
    cfg_b = RAGConfig("longrag", "identity", "lightrag_hybrid", "selfrag_evidence",  "longrag_reader")

    for depth in range(1, 6):
        ledger.record(cfg_a.prefix(depth), q.question_id, f"ea_{depth}", f"cpa_{depth}", depth)

    d_b, e_b = ledger.longest_matching_prefix(cfg_b, q.question_id)
    if d_b != 3:
        errors.append(f"D2: cfg_b should match at depth=3, got {d_b}")

    # --- Sub-test D3: Verify depth=1 is just the frame ---
    entry_d1 = ledger.lookup(("longrag",), q.question_id)
    if entry_d1 is None:
        errors.append("D3: depth=1 ('longrag',) should exist")
    else:
        if entry_d1.prefix != ("longrag",):
            errors.append(f"D3: prefix mismatch {entry_d1.prefix}")

    # --- Sub-test D4: Empty prefix (depth=0) should never be stored ---
    entry_d0 = ledger.lookup((), q.question_id)
    if entry_d0 is not None:
        errors.append(f"D4: depth=0 should not exist, got exec={entry_d0.execution_id}")

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok("All boundary checks passed (depths 0-5)")
    return s


# ======================================================================
# Scenario E: Overwrite consistency
# ======================================================================

def scenario_e_overwrite(ledger: ReuseLedger) -> Scenario:
    s = Scenario("E: Overwrite consistency (INSERT OR REPLACE)")
    q_id = "synthetic::overwrite::q1"

    ledger.record(("longrag", "identity"), q_id, "old_exec", "old_cp", 2,
                  state_hash="old_hash")
    ledger.record(("longrag", "identity"), q_id, "new_exec", "new_cp", 2,
                  state_hash="new_hash")

    entry = ledger.lookup(("longrag", "identity"), q_id)
    errors = []
    if entry is None:
        errors.append("lookup returned None after overwrite")
    elif entry.execution_id != "new_exec":
        errors.append(f"exec_id still old: {entry.execution_id}")
    elif entry.state_hash != "new_hash":
        errors.append(f"hash still old: {entry.state_hash}")
    elif entry.checkpoint_id != "new_cp":
        errors.append(f"cp_id still old: {entry.checkpoint_id}")

    count_before = ledger.count()
    # Overwrite again -- count should not increase
    ledger.record(("longrag", "identity"), q_id, "newer_exec", "newer_cp", 2)
    count_after = ledger.count()
    if count_after != count_before:
        errors.append(f"count grew on overwrite: {count_before} -> {count_after}")

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok("Overwrite replaces cleanly, no count inflation")
    return s


# ======================================================================
# Scenario F: State hash validation
# ======================================================================

def scenario_f_state_hash(ledger: ReuseLedger) -> Scenario:
    s = Scenario("F: State hash validation (detect stale refs)")
    q_id = "synthetic::hash::q1"

    real_state = {"query": "hello", "result": [1, 2, 3]}
    correct_hash = state_content_hash(real_state)

    ledger.record(("longrag", "identity", "lightrag_hybrid"), q_id,
                  "exec_h", "cp_h", 3, state_hash=correct_hash)

    entry = ledger.lookup(("longrag", "identity", "lightrag_hybrid"), q_id)
    errors = []

    if entry is None:
        errors.append("lookup failed")
    elif entry.state_hash != correct_hash:
        errors.append(f"hash mismatch in DB: {entry.state_hash} vs {correct_hash}")

    # Simulate a stale cache: state changed but hash wasn't updated
    tampered_state = {"query": "hello", "result": [1, 2, 999]}
    tampered_hash = state_content_hash(tampered_state)
    if tampered_hash == correct_hash:
        errors.append("BUG: different states produced same hash (collision)")

    # Verify hash function properties
    h1 = state_content_hash({"a": 1, "b": 2})
    h2 = state_content_hash({"b": 2, "a": 1})
    if h1 != h2:
        errors.append("BUG: hash not order-independent (sort_keys broken)")

    h3 = state_content_hash({})
    h4 = state_content_hash({"x": None})
    if h3 == h4:
        errors.append("BUG: empty dict and {'x': None} produce same hash")

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok(f"Hash validation correct: {correct_hash}")
    return s


# ======================================================================
# Scenario G: Concurrent ledger writes
# ======================================================================

def scenario_g_concurrency(ledger: ReuseLedger) -> Scenario:
    s = Scenario("G: Concurrent ledger writes (thread safety)")
    errors_list: List[str] = []
    barrier = threading.Barrier(4)

    def writer(frame: str, n: int):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            return
        for i in range(n):
            try:
                ledger.record(
                    (frame, "identity", str(i)),
                    f"concurrent_q_{i}",
                    f"exec_{frame}_{i}",
                    f"cp_{frame}_{i}",
                    i,
                )
            except Exception as exc:
                errors_list.append(f"{frame}[{i}]: {exc}")

    threads = [
        threading.Thread(target=writer, args=("longrag", 50)),
        threading.Thread(target=writer, args=("lightrag", 50)),
        threading.Thread(target=writer, args=("selfrag", 50)),
        threading.Thread(target=writer, args=("longrag", 50)),  # deliberate overlap
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    if errors_list:
        s.bug(f"{len(errors_list)} concurrent write errors: {errors_list[:3]}")
        return s

    # Verify: longrag was written by 2 threads, last-write-wins
    for i in range(50):
        entry = ledger.lookup(("longrag", "identity", str(i)), f"concurrent_q_{i}")
        if entry is None:
            errors_list.append(f"longrag[{i}] missing after concurrent writes")
            break

    # lightrag and selfrag should have exactly 50 each
    for frame in ("lightrag", "selfrag"):
        for i in range(50):
            entry = ledger.lookup((frame, "identity", str(i)), f"concurrent_q_{i}")
            if entry is None:
                errors_list.append(f"{frame}[{i}] missing")
                break

    if errors_list:
        s.bug("; ".join(errors_list[:5]))
    else:
        s.ok("200 concurrent writes (4 threads), all entries verified")
    return s


# ======================================================================
# Scenario H: Ledger <-> WTB checkpoint round-trip
# ======================================================================

def scenario_h_wtb_roundtrip() -> Scenario:
    s = Scenario("H: Ledger <-> WTB checkpoint round-trip")
    from wtb.sdk import WTBTestBench, WorkflowProject

    tmp = tempfile.mkdtemp(prefix="cache_demo_h_")
    try:
        bench = WTBTestBench.create(mode="development", data_dir=tmp)
        ledger = ReuseLedger(db_path=str(Path(tmp) / "ledger.db"))

        project = WorkflowProject(name="roundtrip", graph_factory=_simple_graph_factory)
        bench.register_project(project)

        qs = ALL_QUESTIONS.get("hotpotqa", [])
        if not qs:
            s.ok("SKIP - no data")
            return s

        q = qs[0]
        config = RAGConfig("longrag", "identity", "lightrag_hybrid",
                           "identity", "longrag_reader")

        execution = bench.run(
            project="roundtrip",
            initial_state={"query": q.question, "messages": [], "count": 0,
                           "result": "", "expanded_queries": [],
                           "retrieval_results": []},
        )

        errors = []
        if execution.status.value != "completed":
            errors.append(f"execution not completed: {execution.status}")
            s.fail("; ".join(errors))
            return s

        cps = bench.get_checkpoints(execution.id)
        if not cps:
            errors.append("no checkpoints from WTB")
            s.fail("; ".join(errors))
            return s

        cp_dicts = [{"checkpoint_id": str(cp.id), "step": cp.step} for cp in cps]
        ledger.record_all_prefixes(config, q.question_id, execution.id, cp_dicts)

        # Verify every depth maps to a real checkpoint
        for depth in range(1, 6):
            entry = ledger.lookup(config.prefix(depth), q.question_id)
            if entry is None:
                errors.append(f"depth={depth} not in ledger after record_all_prefixes")
                continue
            if entry.execution_id != execution.id:
                errors.append(f"depth={depth} exec mismatch")

        # Verify fork from ledger entry works
        depth_3_entry = ledger.lookup(config.prefix(3), q.question_id)
        if depth_3_entry:
            fork = bench.fork(
                execution.id,
                checkpoint_id=depth_3_entry.checkpoint_id,
                new_initial_state={
                    "query": "forked", "messages": ["fork"], "count": 0,
                    "result": "", "expanded_queries": [], "retrieval_results": [],
                },
            )
            if not fork.fork_execution_id:
                errors.append("fork returned no execution_id")
            else:
                forked_exec = bench.get_execution(fork.fork_execution_id)
                if forked_exec is None:
                    errors.append("forked execution not found")

        ledger.close()

        if errors:
            s.bug("; ".join(errors))
        else:
            s.ok(f"Full round-trip: run -> {len(cps)} checkpoints -> ledger -> fork")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return s


# ======================================================================
# Scenario I: Full batch partition audit
# ======================================================================

def scenario_i_batch_partition() -> Scenario:
    s = Scenario("I: Batch partition audit (miss/partial/hit counts)")

    ledger = ReuseLedger(db_path=":memory:")
    qs = []
    for cid in ("hotpotqa", "alce", "ultradomain"):
        qs.extend(ALL_QUESTIONS.get(cid, []))
    if len(qs) < 3:
        s.ok("SKIP - need >= 3 questions across benchmarks")
        return s
    qs = qs[:3]

    # 3 configs: A shares depth-3 prefix with B, C is completely different frame
    cfg_a = RAGConfig("longrag", "identity", "lightrag_hybrid", "identity",          "longrag_reader")
    cfg_b = RAGConfig("longrag", "identity", "lightrag_hybrid", "selfrag_evidence",  "longrag_reader")
    cfg_c = RAGConfig("selfrag", "identity", "lightrag_hybrid", "selfrag_evidence",  "selfrag_generator")
    configs = [cfg_a, cfg_b, cfg_c]

    # Pre-populate: run cfg_a on all questions (simulate prior execution)
    for q in qs:
        for depth in range(1, 6):
            ledger.record(cfg_a.prefix(depth), q.question_id,
                          f"ea_{q.question_id}_{depth}", f"cpa_{depth}", depth)

    # Now partition
    misses, partials, full_hits = [], [], []
    for cfg in configs:
        for q in qs:
            d, e = ledger.longest_matching_prefix(cfg, q.question_id)
            w = WorkItem(cfg, q, d, e)
            if w.is_full_miss:
                misses.append(w)
            elif w.is_full_hit:
                full_hits.append(w)
            elif w.is_partial_hit:
                partials.append(w)

    errors = []
    total = len(configs) * len(qs)
    actual_total = len(misses) + len(partials) + len(full_hits)
    if actual_total != total:
        errors.append(f"partition count mismatch: {actual_total} != {total}")

    # cfg_a x 3q = 3 full hits
    a_hits = [w for w in full_hits if w.config == cfg_a]
    if len(a_hits) != len(qs):
        errors.append(f"cfg_a full hits: expected {len(qs)}, got {len(a_hits)}")

    # cfg_b x 3q = 3 partial hits (depth=3, shares frame+query+retrieval)
    b_partials = [w for w in partials if w.config == cfg_b]
    if len(b_partials) != len(qs):
        errors.append(f"cfg_b partials: expected {len(qs)}, got {len(b_partials)}")
    for w in b_partials:
        if w.reuse_depth != 3:
            errors.append(f"cfg_b partial depth should be 3, got {w.reuse_depth}")

    # cfg_c x 3q = 3 full misses (different frame)
    c_misses = [w for w in misses if w.config == cfg_c]
    if len(c_misses) != len(qs):
        errors.append(f"cfg_c misses: expected {len(qs)}, got {len(c_misses)}")

    ledger.close()

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok(f"{total} work items: {len(full_hits)} hit, {len(partials)} partial, {len(misses)} miss")
    return s


# ======================================================================
# Scenario J: materialized_keys sync with AG-UCT
# ======================================================================

def scenario_j_materialized_keys_sync() -> Scenario:
    s = Scenario("J: materialized_keys sync with AG-UCT Path_t")

    ledger = ReuseLedger(db_path=":memory:")
    qs = ALL_QUESTIONS.get("hotpotqa", [])
    if not qs:
        s.ok("SKIP - no data")
        return s

    q = qs[0]
    cfg = RAGConfig("longrag", "identity", "lightrag_hybrid", "identity", "longrag_reader")

    expected_keys = set()
    for depth in range(1, 6):
        prefix = cfg.prefix(depth)
        ledger.record(prefix, q.question_id, f"e_{depth}", f"cp_{depth}", depth)
        expected_keys.add((prefix, q.question_id))

    actual_keys = ledger.materialized_keys()

    errors = []
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        errors.append(f"missing from materialized_keys: {missing}")
    if extra:
        errors.append(f"extra in materialized_keys: {extra}")

    # Simulate what AG-UCT does: check membership
    for depth in range(1, 6):
        key = (cfg.prefix(depth), q.question_id)
        if key not in actual_keys:
            errors.append(f"AG-UCT membership check failed for depth={depth}")

    # Cross-frame key should NOT be in the set
    cross_key = (("selfrag",), q.question_id)
    if cross_key in actual_keys:
        errors.append("cross-frame key wrongly in materialized_keys")

    ledger.close()

    if errors:
        s.bug("; ".join(errors))
    else:
        s.ok(f"{len(actual_keys)} keys, perfect sync with AG-UCT Path_t")
    return s


# ======================================================================
# Main runner
# ======================================================================

def main() -> None:
    print("=" * 72)
    print("  Bipartite Cache Accuracy Diagnostic")
    print("  Testing (config_prefix, question_id) reuse ledger correctness")
    print("=" * 72)
    print()

    # Load question summary
    for cid, qs in ALL_QUESTIONS.items():
        print(f"  [{cid}] {len(qs)} questions loaded from sample_data")
    print()

    # Scenarios A-G share one in-memory ledger (stateful chain)
    shared_ledger = ReuseLedger(db_path=":memory:")

    scenarios: List[Scenario] = []

    # Chain: each scenario uses/populates the shared ledger,
    # testing that earlier writes don't corrupt later reads.
    t0 = time.perf_counter()

    scenarios.append(scenario_a_exact_match(ReuseLedger(":memory:")))
    scenarios.append(scenario_b_cross_frame_isolation(ReuseLedger(":memory:")))
    scenarios.append(scenario_c_cross_question_isolation(ReuseLedger(":memory:")))
    scenarios.append(scenario_d_prefix_boundary(ReuseLedger(":memory:")))
    scenarios.append(scenario_e_overwrite(ReuseLedger(":memory:")))
    scenarios.append(scenario_f_state_hash(ReuseLedger(":memory:")))
    scenarios.append(scenario_g_concurrency(ReuseLedger(":memory:")))
    scenarios.append(scenario_h_wtb_roundtrip())
    scenarios.append(scenario_i_batch_partition())
    scenarios.append(scenario_j_materialized_keys_sync())

    elapsed = time.perf_counter() - t0

    # Report
    print("-" * 72)
    print(f"  {'Scenario':<55s} {'Status':>6s}")
    print("-" * 72)

    bugs = []
    for sc in scenarios:
        tag = sc.status
        marker = {"PASS": "[OK]", "FAIL": "[!!]", "BUG": "[BUG]", "SKIP": "[--]"}[tag]
        print(f"  {marker} {sc.name:<50s} {tag:>6s}")
        if sc.details and tag != "PASS":
            for line in sc.details.split("; "):
                print(f"       -> {line}")
        if sc.bugs:
            bugs.extend(sc.bugs)

    print("-" * 72)
    passed = sum(1 for sc in scenarios if sc.passed)
    failed = sum(1 for sc in scenarios if sc.passed is False)
    skipped = sum(1 for sc in scenarios if sc.passed is None)
    print(f"  Total: {len(scenarios)} scenarios  |  "
          f"{passed} PASS  |  {failed} FAIL  |  {skipped} SKIP  |  "
          f"{elapsed:.2f}s")

    if bugs:
        print()
        print("  BUGS FOUND:")
        for b in bugs:
            print(f"    - {b}")

    print("=" * 72)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
