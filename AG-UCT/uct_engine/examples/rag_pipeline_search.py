"""AG-UCT SearchState + Evaluator for the 3-framework RAG configuration space.

Search space
------------
  query      : identity | lightrag_keywords
  retrieval  : longrag_dataset | lightrag_hybrid | lightrag_chunk | lightrag_graph
  reranking  : identity | lightrag_compress | selfrag_evidence
  generation : longrag_reader | lightrag_answer | selfrag_generator

The evaluator uses simulated rewards by default.  When ``--real`` is passed,
``build_pipeline_from_config()`` maps slot values to actual adapter instances
and evaluates against real benchmark adapters.

Run::

    python -m uct_engine.examples.rag_pipeline_search           # simulated
    python -m uct_engine.examples.rag_pipeline_search --real    # real pipeline
"""

from __future__ import annotations

import sys
from typing import Any, Hashable

from uct_engine import (
    BenchmarkClusterResult,
    ClusterDef,
    CostAwareUCTScorer,
    EvaluationResult,
    Evaluator,
    ReuseAwareCostModel,
    SearchContext,
    SearchState,
    UCTSearchEngine,
)


# ---------------------------------------------------------------------------
# Configuration space: 4 slots x multiple options per slot
# ---------------------------------------------------------------------------

SLOT_NAMES: list[str] = ["query", "retrieval", "reranking", "generation"]

SLOT_OPTIONS: list[list[str]] = [
    # Query
    ["identity", "lightrag_keywords"],
    # Retrieval
    ["longrag_dataset", "lightrag_hybrid", "lightrag_chunk", "lightrag_graph"],
    # Reranking
    ["identity", "lightrag_compress", "selfrag_evidence"],
    # Generation
    ["longrag_reader", "lightrag_answer", "selfrag_generator"],
]

# Component costs: approximate LLM call counts per query
COMPONENT_COSTS: dict[str, float] = {
    "identity":           0.0,
    "lightrag_keywords":  1.0,   # 1 LLM call for keyword extraction
    "longrag_dataset":    0.0,   # pre-computed lookup
    "lightrag_hybrid":    1.0,   # embedding call
    "lightrag_chunk":     1.0,
    "lightrag_graph":     1.0,
    "lightrag_compress":  1.0,   # 1 LLM call for compression
    "selfrag_evidence":   0.5,   # scoring is lighter than generation
    "longrag_reader":     1.0,   # 1 LLM call
    "lightrag_answer":    1.0,   # 1 LLM call
    "selfrag_generator":  1.5,   # generate + score
}

# Reward lookup: simulated quality scores for key configurations.
# Keys are (query, retrieval, reranking, generation) tuples.
# Missing configs get a default baseline.
REWARD_TABLE: dict[tuple[str, ...], float] = {
    # Strong: LightRAG full pipeline
    ("lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "lightrag_answer"): 0.88,
    # Strong: LightRAG retrieval + SelfRAG reranking + LightRAG gen
    ("lightrag_keywords", "lightrag_hybrid", "selfrag_evidence", "lightrag_answer"):  0.91,
    # Strong: LightRAG retrieval + SelfRAG reranking + SelfRAG gen
    ("lightrag_keywords", "lightrag_hybrid", "selfrag_evidence", "selfrag_generator"): 0.89,
    # LongRAG baseline with identity
    ("identity", "longrag_dataset", "identity", "longrag_reader"):                     0.72,
    # LongRAG + SelfRAG reranking
    ("identity", "longrag_dataset", "selfrag_evidence", "longrag_reader"):             0.78,
    # Cross: LightRAG retrieval + LongRAG generation
    ("lightrag_keywords", "lightrag_hybrid", "identity", "longrag_reader"):            0.76,
    # Cross: LongRAG retrieval + LightRAG generation
    ("identity", "longrag_dataset", "identity", "lightrag_answer"):                    0.68,
    # Cross: LightRAG retrieval + LightRAG compress + SelfRAG gen
    ("lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "selfrag_generator"): 0.86,
    # Chunk-only retrieval
    ("lightrag_keywords", "lightrag_chunk", "lightrag_compress", "lightrag_answer"):   0.80,
    # Graph-only retrieval
    ("lightrag_keywords", "lightrag_graph", "lightrag_compress", "lightrag_answer"):   0.82,
    # LongRAG + LightRAG compress + LightRAG gen
    ("identity", "longrag_dataset", "lightrag_compress", "lightrag_answer"):           0.74,
    # SelfRAG evidence + LongRAG reader
    ("lightrag_keywords", "lightrag_graph", "selfrag_evidence", "longrag_reader"):     0.79,
}

# Default reward for configs not in the table
DEFAULT_REWARD = 0.55

# Incompatible combinations that get penalized
INCOMPATIBLE: set[tuple[str, str]] = set()


def _compute_reward(choices: tuple[str, ...]) -> float:
    if choices in REWARD_TABLE:
        return REWARD_TABLE[choices]
    base = DEFAULT_REWARD
    for component in choices:
        base += COMPONENT_COSTS.get(component, 0.0) * 0.02
    return min(base, 0.70)


def _compute_cost(choices: tuple[str, ...]) -> float:
    return sum(COMPONENT_COSTS.get(c, 0.5) for c in choices)


# ---------------------------------------------------------------------------
# Pipeline builder: maps slot values to real adapter instances
# ---------------------------------------------------------------------------

def _build_longrag_generation():
    """Build a LongRAGGeneration with a SimpleLLM-backed inference shim.

    ``LongRAGGeneration`` requires an ``llm_inference`` object exposing
    ``predict_nq()`` and ``predict_hotpotqa()``.  When the full LongRAG
    inference module is unavailable, this creates a lightweight shim that
    delegates to ``SimpleLLMGeneration`` from ``rag_contracts``.
    """
    import os

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        from rag_contracts import IdentityGeneration
        return IdentityGeneration()

    class _LLMInferenceShim:
        """Mimics LongRAG's inference API using OpenAI chat completions."""

        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def _ask(self, context, query, titles):
            titles_str = ", ".join(titles) if titles else "N/A"
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                max_tokens=300,
                messages=[
                    {"role": "system",
                     "content": "You are an expert reader. Extract the answer "
                     "from the provided context. Be concise and precise."},
                    {"role": "user",
                     "content": f"Titles: {titles_str}\n\n"
                     f"Context:\n{context[:4000]}\n\n"
                     f"Question: {query}\n\nAnswer:"},
                ],
            )
            ans = (resp.choices[0].message.content or "").strip()
            return ans, ans

        def predict_nq(self, context, query, titles):
            return self._ask(context, query, titles)

        def predict_hotpotqa(self, context, query, titles):
            return self._ask(context, query, titles)

    from longRAG_example.longrag_langgraph.adapters import LongRAGGeneration
    return LongRAGGeneration(llm_inference=_LLMInferenceShim())


def _build_selfrag_components() -> tuple:
    """Build SelfRAG reranking and generation adapters using an OpenAI vLLM shim.

    Uses the same approach as ``real_selfrag_swap_demo.py``: wraps OpenAI
    chat completions behind the vLLM ``generate()`` interface so that
    ``SelfRAGReranking`` / ``SelfRAGGeneration`` can score passages with
    synthetic logprobs.

    Returns ``(reranking, generation)`` or ``(None, None)`` on failure.
    """
    import os
    import types
    from unittest.mock import MagicMock

    if "vllm" not in sys.modules:
        from dataclasses import dataclass as _dc

        _fv = types.ModuleType("vllm")

        @_dc
        class _SP:
            temperature: float = 0.0
            top_p: float = 1.0
            max_tokens: int = 100
            logprobs: int = 0

        _fv.SamplingParams = _SP
        sys.modules["vllm"] = _fv

    if "torch" not in sys.modules:
        _ft = types.ModuleType("torch")
        _ft.no_grad = lambda: MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: None
        )
        sys.modules["torch"] = _ft

    _sr_root = str(
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent
        / "self-rag_langgraph" / "self-rag-wtb"
    )
    if _sr_root not in sys.path:
        sys.path.insert(0, _sr_root)

    try:
        from selfrag.adapters import SelfRAGGeneration, SelfRAGReranking
        from selfrag.constants import (
            ground_tokens_names,
            load_special_tokens,
            rel_tokens_names,
            retrieval_tokens_names,
            utility_tokens_names,
        )
    except ImportError:
        return None, None

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, None

    ALL_SPECIAL = (
        retrieval_tokens_names + rel_tokens_names
        + ground_tokens_names + utility_tokens_names
    )
    TOKEN_MAP = {tok: 2000 + i for i, tok in enumerate(ALL_SPECIAL)}

    class _Tok:
        def convert_tokens_to_ids(self, token: str) -> int:
            return TOKEN_MAP.get(token, 0)

    class _Out:
        def __init__(self, text):
            self.text = text
            from selfrag.constants import control_tokens as _ct
            found_rel = next((t for t in rel_tokens_names if t in text), "[Relevant]")
            found_grd = next((t for t in ground_tokens_names if t in text), "[Fully supported]")
            found_ut = next((t for t in utility_tokens_names if t in text), "[Utility:5]")
            n = max(5, len(text.split()) // 2)
            self.token_ids = [TOKEN_MAP.get(found_rel, 888)]
            self.token_ids.extend([888] * n)
            gp = len(self.token_ids)
            self.token_ids.append(TOKEN_MAP.get(found_grd, 888))
            up = len(self.token_ids)
            self.token_ids.append(TOKEN_MAP.get(found_ut, 888))
            self.logprobs = []
            for pos in range(len(self.token_ids)):
                e = {}
                for t in rel_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == 0 and t == found_rel) else -5.0
                for t in retrieval_tokens_names:
                    e[TOKEN_MAP[t]] = -5.0
                for t in ground_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == gp and t == found_grd) else -5.0
                for t in utility_tokens_names:
                    e[TOKEN_MAP[t]] = -0.1 if (pos == up and t == found_ut) else -5.0
                self.logprobs.append(e)
            self.cumulative_logprob = -2.0

    class _Pred:
        def __init__(self, o):
            self.outputs = [o]

    class _Model:
        def __init__(self):
            from openai import OpenAI
            base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE", "")
            self.client = OpenAI(api_key=api_key, base_url=base or None)
            self.model = os.environ.get("DEFAULT_LLM", "gpt-4o-mini")

        def generate(self, prompts, sp=None):
            out = []
            for p in prompts:
                try:
                    r = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system",
                             "content": "You are a Self-RAG assistant. Given a question and "
                             "optionally a paragraph of evidence, produce a SHORT factual "
                             "answer wrapped in Self-RAG control tokens.\n"
                             "Format: [Relevant]<answer>[Fully supported][Utility:5]"},
                            {"role": "user", "content": p},
                        ],
                        temperature=getattr(sp, "temperature", 0.0),
                        max_tokens=getattr(sp, "max_tokens", 150),
                    )
                    txt = r.choices[0].message.content or ""
                except Exception:
                    txt = "[Relevant]Error[Partially supported][Utility:3]"
                out.append(_Pred(_Out(txt)))
            return out

    tok = _Tok()
    _, rel, grd, ut = load_special_tokens(tok, use_grounding=True, use_utility=True)
    model = _Model()

    reranking = SelfRAGReranking(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    generation = SelfRAGGeneration(model=model, rel_tokens=rel, grd_tokens=grd, ut_tokens=ut)
    return reranking, generation


def build_pipeline_from_config(
    choices: tuple[str, ...],
    benchmark: str = "hotpotqa",
) -> dict[str, Any]:
    """Build a dict of rag_contracts components from a UCT config tuple.

    Returns ``{"query": ..., "retrieval": ..., "reranking": ..., "generation": ...}``.
    Raises ImportError if required packages are not installed.
    """
    from rag_contracts import IdentityQuery, IdentityReranking, IdentityGeneration, SimpleLLMGeneration

    query_name, retrieval_name, reranking_name, generation_name = choices
    components: dict[str, Any] = {}

    if query_name == "identity":
        components["query"] = IdentityQuery()
    elif query_name == "lightrag_keywords":
        from lightrag_langgraph.adapters import LightRAGQuery
        components["query"] = LightRAGQuery()

    if retrieval_name == "longrag_dataset":
        from longRAG_example.longrag_langgraph.adapters import HFDatasetRetrieval
        components["retrieval"] = HFDatasetRetrieval()
    elif retrieval_name in ("lightrag_hybrid", "lightrag_chunk", "lightrag_graph"):
        from lightrag_langgraph.adapters import LightRAGRetrieval
        mode_map = {
            "lightrag_hybrid": "hybrid",
            "lightrag_chunk": "chunk",
            "lightrag_graph": "graph",
        }
        components["retrieval"] = LightRAGRetrieval(mode=mode_map[retrieval_name])

    if reranking_name == "identity":
        components["reranking"] = IdentityReranking()
    elif reranking_name == "lightrag_compress":
        from lightrag_langgraph.adapters import LightRAGReranking
        components["reranking"] = LightRAGReranking()
    elif reranking_name == "selfrag_evidence":
        selfrag_rr, _ = _build_selfrag_components()
        components["reranking"] = selfrag_rr if selfrag_rr else IdentityReranking()

    if generation_name == "longrag_reader":
        components["generation"] = _build_longrag_generation()
    elif generation_name == "lightrag_answer":
        from lightrag_langgraph.adapters import LightRAGGeneration
        components["generation"] = LightRAGGeneration()
    elif generation_name == "selfrag_generator":
        _, selfrag_gen = _build_selfrag_components()
        components["generation"] = selfrag_gen if selfrag_gen else IdentityGeneration()

    return components


def evaluate_config_real(
    choices: tuple[str, ...],
    benchmark: str = "hotpotqa",
) -> float:
    """Build a real pipeline from config, run on sample data, return score.

    This is the bridge between AG-UCT and real rag_contracts evaluation.
    Returns a normalized reward in [0, 1].
    """
    try:
        components = build_pipeline_from_config(choices, benchmark)
    except (ImportError, Exception):
        return DEFAULT_REWARD

    generation = components.get("generation")
    if generation is None:
        return DEFAULT_REWARD

    try:
        if benchmark == "hotpotqa":
            from benchmark.hotpotqa_adapter import (
                HotpotQABenchmarkAdapter,
                load_hotpotqa_sample,
            )
            data = load_hotpotqa_sample("benchmark/sample_data/hotpotqa_kg_sample")
            adapter = HotpotQABenchmarkAdapter()
            result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "ultradomain":
            from benchmark.ultradomain_adapter import (
                UltraDomainBenchmarkAdapter,
                load_ultradomain_sample,
            )
            data = load_ultradomain_sample("benchmark/sample_data/ultradomain_kg_sample")
            adapter = UltraDomainBenchmarkAdapter()
            result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "alce":
            from benchmark.alce_adapter import ALCEBenchmarkAdapter, load_alce_data
            import json
            from pathlib import Path
            docs_path = Path("benchmark/sample_data/alce_kg_sample/alce_docs.json")
            if docs_path.exists():
                with open(docs_path, encoding="utf-8") as f:
                    all_docs = json.load(f)
                queries_path = Path("benchmark/sample_data/alce_kg_sample/queries.jsonl")
                data = []
                with open(queries_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            q = json.loads(line)
                            data.append({
                                "question": q["query"],
                                "answer": q.get("ground_truth", ""),
                                "docs": all_docs.get(q["query_id"], []),
                            })
                adapter = ALCEBenchmarkAdapter()
                result = adapter.evaluate_generation(data, generation)
                return result.avg_f1 / 100.0
    except Exception:
        pass

    return DEFAULT_REWARD


# ---------------------------------------------------------------------------
# SearchState
# ---------------------------------------------------------------------------

class RAGPipelineSearchState:
    """Partial RAG pipeline config being assembled slot by slot."""

    def __init__(self, choices: tuple[str, ...] = ()) -> None:
        self.choices = choices

    def is_terminal(self) -> bool:
        return len(self.choices) == len(SLOT_NAMES)

    def available_actions(self) -> list[Hashable]:
        depth = len(self.choices)
        if depth >= len(SLOT_OPTIONS):
            return []
        return list(SLOT_OPTIONS[depth])

    def child(self, action: Hashable) -> "RAGPipelineSearchState":
        return RAGPipelineSearchState(self.choices + (str(action),))

    def state_key(self) -> Hashable:
        return self.choices

    def pretty(self) -> str:
        parts = [f"{SLOT_NAMES[i]}={v}" for i, v in enumerate(self.choices)]
        return "(" + ", ".join(parts) + ")"

    def path_key_for_action(self, action: Hashable) -> Hashable:
        return self.choices + (str(action),)


# ---------------------------------------------------------------------------
# Benchmark clusters (simulating datasets)
# ---------------------------------------------------------------------------

CLUSTER_IDS = ["hotpotqa", "ultradomain", "alce"]

CLUSTER_NOISE: dict[str, float] = {
    "hotpotqa":     0.02,
    "ultradomain": -0.01,
    "alce":         0.00,
}

CLUSTER_COST: dict[str, float] = {
    "hotpotqa":     1.0,
    "ultradomain":  1.2,
    "alce":         1.5,
}

# Per-dataset reward modifiers (some configs work better on some datasets)
DATASET_AFFINITY: dict[tuple[str, str], float] = {
    # LightRAG hybrid excels on UltraDomain (domain-specific KG)
    ("lightrag_hybrid", "ultradomain"): 0.05,
    ("lightrag_graph", "ultradomain"):  0.04,
    # SelfRAG evidence scoring helps with multi-hop (HotpotQA)
    ("selfrag_evidence", "hotpotqa"):   0.03,
    # LightRAG compression helps ALCE (citation quality)
    ("lightrag_compress", "alce"):      0.03,
    # LongRAG dataset retrieval is tuned for HotpotQA
    ("longrag_dataset", "hotpotqa"):    0.04,
}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGPipelineEvaluator(Evaluator):
    """Evaluates a complete RAG pipeline configuration across 3 benchmark datasets.

    When ``use_real=False`` (default), uses the simulated reward table.
    When ``use_real=True``, builds actual adapter instances via
    ``build_pipeline_from_config()`` and evaluates against real benchmark
    adapters with sample data.

    For each dataset cluster:
    1. Computes reward (simulated or real)
    2. Applies dataset-specific affinity bonuses (simulated only)
    3. Tracks path-prefix reuse for cost optimization
    """

    def __init__(self, use_real: bool = False) -> None:
        self.use_real = use_real

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        assert state.is_terminal(), "evaluator expects a terminal state"
        choices: tuple[str, ...] = state.state_key()  # type: ignore[assignment]

        cluster_results: list[BenchmarkClusterResult] = []
        total_reward = 0.0
        total_cost = 0.0
        weight = 1.0 / len(CLUSTER_IDS)

        for cid in CLUSTER_IDS:
            if self.use_real:
                reward = evaluate_config_real(choices, benchmark=cid)
            else:
                reward = _compute_reward(choices) + CLUSTER_NOISE[cid]
                for component in choices:
                    reward += DATASET_AFFINITY.get((component, cid), 0.0)

            new_keys: list[Hashable] = []
            cluster_cost = 0.0
            for depth in range(1, len(choices) + 1):
                prefix_key: Hashable = (choices[:depth], cid)
                if prefix_key not in context.materialized_keys:
                    new_keys.append(prefix_key)
                    cluster_cost += CLUSTER_COST[cid] / len(choices)

            cluster_results.append(BenchmarkClusterResult(
                cluster_id=cid,
                reward=reward,
                cost=cluster_cost,
                materialized_keys=new_keys,
                metadata={
                    "config": {SLOT_NAMES[i]: v for i, v in enumerate(choices)},
                    "dataset": cid,
                    "real_eval": self.use_real,
                },
            ))
            total_reward += weight * reward
            total_cost += cluster_cost

        return EvaluationResult(
            reward=total_reward,
            total_cost=total_cost,
            cluster_results=cluster_results,
            metadata={
                "config_pretty": state.pretty(),
                "real_eval": self.use_real,
            },
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    use_real = "--real" in sys.argv

    evaluator = RAGPipelineEvaluator(use_real=use_real)
    scorer = CostAwareUCTScorer(lambda_t=0.05)
    clusters = [
        ClusterDef(cid, weight=1.0, base_cost=CLUSTER_COST[cid])
        for cid in CLUSTER_IDS
    ]
    cost_model = ReuseAwareCostModel(clusters=clusters)

    engine = UCTSearchEngine(
        evaluator=evaluator,
        scorer=scorer,
        cost_model=cost_model,
        exploration_constant=1.4,
        random_seed=42,
    )

    root = RAGPipelineSearchState()
    result = engine.search(root, max_iterations=500)

    print("=" * 70, flush=True)
    print("  RAG Pipeline UCT Search Complete", flush=True)
    print("=" * 70, flush=True)
    print(f"  Best config : {result.best_state.pretty()}", flush=True)
    print(f"  Best reward : {result.best_reward:.4f}", flush=True)
    print(f"  Iterations  : {result.iterations}", flush=True)
    print(f"  Evaluations : {result.total_evaluations}", flush=True)
    print(f"  Total cost  : {result.total_cost:.2f}", flush=True)
    print(f"  Materialized keys: {len(result.context.materialized_keys)}", flush=True)
    print("=" * 70, flush=True)

    total_configs = 1
    for opts in SLOT_OPTIONS:
        total_configs *= len(opts)
    print(f"\n  Search space : {total_configs} total configurations", flush=True)
    print(f"  Explored     : {result.total_evaluations} evaluations", flush=True)
    print(f"  Efficiency   : {result.total_evaluations / total_configs:.1%} of space", flush=True)

    print("\n  Slot breakdown:", flush=True)
    for action, child in result.root_node.children.items():
        print(
            f"    {SLOT_NAMES[0]}={action!s:20s}  visits={child.visit_count:4d}  "
            f"Q={child.q_value:.4f}  best={child.best_value:.4f}",
            flush=True,
        )

    print("\n  Top-5 configs by Q-value:", flush=True)
    configs_visited = []

    def _collect(node, depth=0, prefix=()):
        if depth == len(SLOT_NAMES):
            if node.visit_count > 0:
                configs_visited.append((prefix, node.q_value, node.visit_count))
            return
        for act, ch in node.children.items():
            _collect(ch, depth + 1, prefix + (str(act),))

    _collect(result.root_node)
    configs_visited.sort(key=lambda x: x[1], reverse=True)
    for cfg, q, visits in configs_visited[:5]:
        label = ", ".join(f"{SLOT_NAMES[i]}={v}" for i, v in enumerate(cfg))
        print(f"    Q={q:.4f}  visits={visits:3d}  ({label})", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
