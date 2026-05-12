"""AG-UCT SearchState + Evaluator for the OminiRAG 5-dimension configuration space.

Search space (5 dimensions -- aligned with RAG survey taxonomy)
----------------------------------------------------------------
  chunking       : standard_passage | longrag_4k | kg_extraction
  query          : identity | lightrag_keywords
  retrieval      : bm25 | dense_e5 | bm25_dense_hybrid | lightrag_hybrid | lightrag_graph
  post_retrieval : identity | cross_encoder | lightrag_compress | selfrag_critique
  generation     : longrag_reader | lightrag_answer | selfrag_generator | simple_llm

Hard constraint: lightrag_hybrid / lightrag_graph require kg_extraction chunking.

Framework presets:
  LongRAG  = longrag_4k      + identity          + bm25/dense     + identity          + longrag_reader
  LightRAG = kg_extraction   + lightrag_keywords + lightrag_hybrid + lightrag_compress + lightrag_answer
  Self-RAG = standard_passage + identity          + dense_e5        + selfrag_critique  + selfrag_generator

The evaluator uses simulated rewards by default.  When ``--real`` is passed,
``build_pipeline_from_config()`` maps slot values to actual adapter instances
and evaluates against real benchmark adapters.

When ``--wtb-reuse`` is passed, the search uses the ``RAGCacheAwareEvaluator``
from ``ominirag_wtb`` which physically caches intermediate LangGraph states
via WTB and forks from shared prefixes to avoid redundant computation.

Run::

    python -m uct_engine.examples.rag_pipeline_search                     # simulated
    python -m uct_engine.examples.rag_pipeline_search --real              # real pipeline (sample_data)
    python -m uct_engine.examples.rag_pipeline_search --data-dir /data1/ragworkspace/dataset  # real datasets
    python -m uct_engine.examples.rag_pipeline_search --wtb-reuse         # with WTB reuse
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
# Configuration space: 5 dimensions aligned with RAG survey taxonomy
# ---------------------------------------------------------------------------

SLOT_NAMES: list[str] = [
    "chunking", "query", "retrieval", "post_retrieval", "generation",
]

SLOT_OPTIONS: list[list[str]] = [
    # Chunking (offline corpus preparation)
    ["standard_passage", "longrag_4k", "kg_extraction"],
    # Query (expansion / decomposition)
    ["identity", "lightrag_keywords"],
    # Retrieval (real corpus search)
    ["bm25", "dense_e5", "bm25_dense_hybrid", "lightrag_hybrid", "lightrag_graph"],
    # Post-Retrieval (reranking / compression / critique)
    ["identity", "cross_encoder", "lightrag_compress", "selfrag_critique"],
    # Generation
    ["longrag_reader", "lightrag_answer", "selfrag_generator", "simple_llm"],
]

# Backward-compatible old slot names for migration
SLOT_NAMES_OLD: list[str] = ["frame", "query", "retrieval", "reranking", "generation"]

# Hard constraint: lightrag_hybrid / lightrag_graph require kg_extraction chunking
CHUNKING_RETRIEVAL_CONSTRAINTS: dict[str, str] = {
    "lightrag_hybrid": "kg_extraction",
    "lightrag_graph": "kg_extraction",
}

# Component costs: approximate compute units per query
COMPONENT_COSTS: dict[str, float] = {
    # Chunking (offline, amortised to near-zero at query time)
    "standard_passage":   0.0,
    "longrag_4k":         0.0,
    "kg_extraction":      0.0,
    # Query
    "identity":           0.0,
    "lightrag_keywords":  1.0,
    # Retrieval
    "bm25":               0.1,   # CPU-only lexical search
    "dense_e5":           0.3,   # one embedding inference
    "bm25_dense_hybrid":  0.4,   # BM25 + embedding + RRF merge
    "lightrag_hybrid":    1.0,   # embedding + KG traversal
    "lightrag_graph":     1.0,   # KG-only traversal
    # Post-Retrieval
    "cross_encoder":      0.5,   # N forward passes on cross-encoder
    "lightrag_compress":  1.0,   # 1 LLM call for compression
    "selfrag_critique":   0.5,   # scoring is lighter than generation
    # Generation
    "longrag_reader":     1.0,
    "lightrag_answer":    1.0,
    "selfrag_generator":  1.5,   # generate + score
    "simple_llm":         1.0,
}

# Reward lookup: simulated quality scores for key configurations.
# Keys are 5-tuples (chunking, query, retrieval, post_retrieval, generation).
REWARD_TABLE: dict[tuple[str, ...], float] = {
    # -- Framework preset baselines --
    # LongRAG preset: longrag_4k + identity + bm25 + identity + longrag_reader
    ("longrag_4k", "identity", "bm25", "identity", "longrag_reader"):                     0.72,
    # LightRAG preset: kg_extraction + lightrag_keywords + lightrag_hybrid + lightrag_compress + lightrag_answer
    ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "lightrag_compress", "lightrag_answer"): 0.88,
    # Self-RAG preset: standard_passage + identity + dense_e5 + selfrag_critique + selfrag_generator
    ("standard_passage", "identity", "dense_e5", "selfrag_critique", "selfrag_generator"):  0.85,

    # -- Cross-framework combinations --
    # BM25 + cross-encoder reranking (classic IR pipeline)
    ("standard_passage", "identity", "bm25", "cross_encoder", "longrag_reader"):           0.79,
    ("standard_passage", "identity", "bm25", "cross_encoder", "simple_llm"):               0.78,
    # Dense + cross-encoder
    ("standard_passage", "identity", "dense_e5", "cross_encoder", "longrag_reader"):       0.82,
    ("standard_passage", "identity", "dense_e5", "cross_encoder", "simple_llm"):           0.81,
    # Hybrid BM25+Dense + cross-encoder (expected strongest standard pipeline)
    ("standard_passage", "identity", "bm25_dense_hybrid", "cross_encoder", "longrag_reader"):  0.86,
    ("standard_passage", "identity", "bm25_dense_hybrid", "cross_encoder", "simple_llm"):      0.85,
    # Hybrid + selfrag critique
    ("standard_passage", "identity", "bm25_dense_hybrid", "selfrag_critique", "selfrag_generator"): 0.87,
    # LightRAG retrieval + cross-encoder
    ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "cross_encoder", "lightrag_answer"):  0.90,
    # LightRAG retrieval + selfrag critique + selfrag gen
    ("kg_extraction", "lightrag_keywords", "lightrag_hybrid", "selfrag_critique", "selfrag_generator"): 0.89,
    # LongRAG 4k chunks + dense retrieval
    ("longrag_4k", "identity", "dense_e5", "identity", "longrag_reader"):                  0.74,
    ("longrag_4k", "identity", "dense_e5", "cross_encoder", "longrag_reader"):             0.80,
    # LightRAG graph-only retrieval
    ("kg_extraction", "lightrag_keywords", "lightrag_graph", "lightrag_compress", "lightrag_answer"): 0.82,
    # Hybrid + lightrag compress
    ("standard_passage", "identity", "bm25_dense_hybrid", "lightrag_compress", "lightrag_answer"):    0.83,
}

# Default reward for configs not in the table
DEFAULT_REWARD = 0.55

# Incompatible: lightrag retrieval without kg_extraction gets penalised
INCOMPATIBLE_PENALTY = 0.20


def _check_constraints(choices: tuple[str, ...]) -> bool:
    """Return True if the configuration satisfies hard constraints."""
    if len(choices) >= 3:
        chunking, _, retrieval = choices[0], choices[1], choices[2]
        required = CHUNKING_RETRIEVAL_CONSTRAINTS.get(retrieval)
        if required and chunking != required:
            return False
    return True


def _compute_reward(choices: tuple[str, ...]) -> float:
    if not _check_constraints(choices):
        return DEFAULT_REWARD - INCOMPATIBLE_PENALTY

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

from rag_contracts.component_registry import (  # noqa: E402
    build_longrag_generation as _build_longrag_generation,
    build_pipeline_from_config,
    build_selfrag_components as _build_selfrag_components,
    build_simple_llm as _build_simple_llm,
)


_CORPUS_CACHE: dict[str, "CorpusIndex"] = {}


def _load_corpus_for_benchmark(benchmark: str, chunks: dict | None = None):
    """Load a CorpusIndex for the given benchmark.

    Priority: inline chunks > real corpus index > KG sample fallback.
    Real corpus indexes are loaded once and cached for the process lifetime.
    """
    from rag_contracts import CorpusIndex
    from pathlib import Path
    import os

    if chunks:
        return CorpusIndex.from_chunks_dict(chunks)

    if benchmark in _CORPUS_CACHE:
        return _CORPUS_CACHE[benchmark]

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_root = Path(os.environ.get("OMINIRAG_DATA_ROOT", "/data1/ragworkspace/train"))

    real_paths = {
        "hotpotqa": [
            project_root / "data" / "corpus_indexes" / "fullwiki_corpus_index.json",
            data_root / "fullwiki" / "fullwiki_corpus_index.json",
        ],
        "ultradomain": [
            project_root / "data" / "corpus_indexes" / "ultradomain_mix_corpus_index.json",
            data_root / "ultradomain" / "ultradomain_corpus_index.json",
        ],
    }
    sample_paths = {
        "hotpotqa": project_root / "benchmark" / "sample_data" / "hotpotqa_kg_sample" / "chunks.json",
        "ultradomain": project_root / "benchmark" / "sample_data" / "ultradomain_kg_sample" / "chunks.json",
    }

    for p in real_paths.get(benchmark, []):
        if p.exists():
            corpus = CorpusIndex.from_json_file(p)
            _CORPUS_CACHE[benchmark] = corpus
            print(f"  [corpus] Loaded real index for {benchmark}: {len(corpus)} chunks from {p}", flush=True)
            return corpus

    sample = sample_paths.get(benchmark)
    if sample and sample.exists():
        corpus = CorpusIndex.from_json_file(sample)
        _CORPUS_CACHE[benchmark] = corpus
        print(f"  [corpus] Loaded sample index for {benchmark}: {len(corpus)} chunks", flush=True)
        return corpus

    return None


def evaluate_config_real(
    choices: tuple[str, ...],
    benchmark: str = "hotpotqa",
    frozen_sample: list | None = None,
) -> float:
    """Build a real pipeline from config, run on sample data, return score.

    This is the bridge between AG-UCT and real rag_contracts evaluation.
    Returns a normalized reward in [0, 1].

    The full pipeline (retrieval + post-retrieval + generation) is exercised
    when a corpus is available.  For ALCE, the graph_factory pattern with
    ALCEDocRetrieval is used.  Falls back to generation-only evaluation
    when corpus loading or pipeline construction fails.

    Parameters
    ----------
    frozen_sample:
        Optional list of ``BenchmarkItem`` pre-drawn via ``SamplingEngine``.
        When provided, items are converted to eval dicts on-the-fly instead
        of loading from ``sample_data/``.
    """
    if frozen_sample is not None:
        return _evaluate_frozen(frozen_sample, benchmark, choices)

    try:
        corpus = _load_corpus_for_benchmark(benchmark)
        components = build_pipeline_from_config(choices, benchmark, corpus=corpus)
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
            graph = _build_eval_graph(choices, components)
            if graph is not None:
                result = adapter.evaluate_pipeline(data, graph)
            else:
                result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "ultradomain":
            from benchmark.ultradomain_adapter import (
                UltraDomainBenchmarkAdapter,
                load_ultradomain_sample,
            )
            data = load_ultradomain_sample("benchmark/sample_data/ultradomain_kg_sample")
            adapter = UltraDomainBenchmarkAdapter()
            graph = _build_eval_graph(choices, components)
            if graph is not None:
                result = adapter.evaluate_pipeline(data, graph)
            else:
                result = adapter.evaluate_generation(data, generation)
            return result.avg_f1 / 100.0
        elif benchmark == "alce":
            from benchmark.alce_adapter import ALCEBenchmarkAdapter
            import json
            from pathlib import Path as _P
            docs_path = _P("benchmark/sample_data/alce_kg_sample/alce_docs.json")
            if docs_path.exists():
                with open(docs_path, encoding="utf-8") as f:
                    all_docs = json.load(f)
                queries_path = _P("benchmark/sample_data/alce_kg_sample/queries.jsonl")
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
                factory = _build_alce_graph_factory(choices, components)
                if factory is not None:
                    result = adapter.evaluate_pipeline(data, factory)
                else:
                    result = adapter.evaluate_generation(data, generation)
                return result.avg_f1 / 100.0
    except Exception:
        pass

    return DEFAULT_REWARD


def _build_eval_graph(choices: tuple[str, ...], components: dict):
    """Try to build a compiled LangGraph from components. Returns None on failure."""
    try:
        from ominirag_wtb.config_types import RAGConfig
        from ominirag_wtb.graph_factories import config_to_graph_factory
        config = RAGConfig.from_tuple(choices if len(choices) == 5 else ("standard_passage",) + choices)
        factory = config_to_graph_factory(config)
        return factory()
    except Exception:
        return None


def _build_alce_graph_factory(choices: tuple[str, ...], components: dict):
    """Build a graph_factory callable for ALCE (injects ALCEDocRetrieval per item)."""
    try:
        from ominirag_wtb.config_types import RAGConfig
        from ominirag_wtb.graph_factories import _infer_frame, _get_frame_builder
        from rag_contracts import ALCEDocRetrieval

        config = RAGConfig.from_tuple(choices if len(choices) == 5 else ("standard_passage",) + choices)
        frame = _infer_frame(config)
        builder = _get_frame_builder(frame)

        def factory(retrieval=None):
            ret = retrieval if retrieval is not None else components["retrieval"]
            return builder(
                retrieval=ret,
                generation=components["generation"],
                reranking=components.get("post_retrieval"),
                query=components.get("query"),
            )
        return factory
    except Exception:
        return None


def _evaluate_frozen(frozen_sample: list, benchmark: str, choices: tuple[str, ...]) -> float:
    """Evaluate a frozen BenchmarkItem sample with the full pipeline when possible.

    Supports two item formats:

    * **BenchmarkItem objects** (from ``SamplingEngine`` / HuggingFace) --
      converted to eval dicts using ``payload`` / ``target`` / ``metadata``.
    * **Plain eval dicts** (from ``build_frozen_samples_real``) --
      already carry ``question``, ``answer``, ``context_results``, etc.
      These are passed through directly.
    """
    try:
        if benchmark == "hotpotqa":
            from benchmark.hotpotqa_adapter import HotpotQABenchmarkAdapter
            eval_items = _frozen_to_eval_items_hotpotqa(frozen_sample)
            generation, graph = _try_build_pipeline(eval_items, choices, benchmark)
            if generation is None:
                return DEFAULT_REWARD

            adapter = HotpotQABenchmarkAdapter()
            if graph is not None:
                result = adapter.evaluate_pipeline(eval_items, graph)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0

        elif benchmark == "ultradomain":
            from benchmark.ultradomain_adapter import UltraDomainBenchmarkAdapter
            eval_items = _frozen_to_eval_items_ultradomain(frozen_sample)
            generation, graph = _try_build_pipeline(eval_items, choices, benchmark)
            if generation is None:
                return DEFAULT_REWARD

            adapter = UltraDomainBenchmarkAdapter()
            if graph is not None:
                result = adapter.evaluate_pipeline(eval_items, graph)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0

        elif benchmark == "alce":
            from benchmark.alce_adapter import ALCEBenchmarkAdapter
            eval_items = []
            for bitem in frozen_sample:
                eval_items.append({
                    "question": bitem.payload["question"],
                    "answer": bitem.target["answer"],
                    "docs": bitem.payload.get("docs", [])[:5],
                    "qa_pairs": bitem.target.get("qa_pairs", []),
                    "query_id": bitem.item_id,
                })

            components = build_pipeline_from_config(choices, benchmark)
            generation = components.get("generation")
            if generation is None:
                return DEFAULT_REWARD

            adapter = ALCEBenchmarkAdapter()
            factory = _build_alce_graph_factory(choices, components)
            if factory is not None:
                result = adapter.evaluate_pipeline(eval_items, factory)
            else:
                result = adapter.evaluate_generation(eval_items, generation)
            return result.avg_f1 / 100.0

    except Exception:
        pass
    return DEFAULT_REWARD


def _try_build_pipeline(eval_items, choices, benchmark):
    """Build pipeline components, falling back to generation-only on error.

    When real data carries ``context_results``, retrieval is unnecessary --
    ``get_context_for_item()`` in the adapter will use those directly.
    Returns ``(generation, graph)``; *graph* may be ``None``.
    """
    try:
        all_chunks = _collect_chunks(eval_items)
        corpus = _load_corpus_for_benchmark(benchmark, all_chunks or None)
        components = build_pipeline_from_config(choices, benchmark, corpus=corpus)
        generation = components.get("generation")
        graph = _build_eval_graph(choices, components)
        return generation, graph
    except Exception:
        pass
    # Retrieval / reranking may fail (missing deps), but generation might work
    try:
        gen_name = choices[-1] if choices else "simple_llm"
        if gen_name == "longrag_reader":
            generation = _build_longrag_generation()
        elif gen_name == "simple_llm":
            from rag_contracts.common_components import SimpleLLMGeneration
            generation = SimpleLLMGeneration(llm=_build_simple_llm())
        elif gen_name == "selfrag_generator":
            _, generation = _build_selfrag_components()
        else:
            from rag_contracts.identity import IdentityGeneration
            generation = IdentityGeneration()
        return generation, None
    except Exception:
        return None, None


def _is_plain_dict(item) -> bool:
    """True when *item* is a plain eval dict (from ``load_*_real``)."""
    return isinstance(item, dict) and "question" in item


def _frozen_to_eval_items_hotpotqa(frozen_sample: list) -> list[dict]:
    """Convert frozen sample to HotpotQA eval dicts.

    Plain dicts (from ``load_hotpotqa_real``) are passed through unchanged.
    BenchmarkItem objects are converted using their ``payload`` / ``target``.
    """
    eval_items: list[dict] = []
    for bitem in frozen_sample:
        if _is_plain_dict(bitem):
            eval_items.append(bitem)
            continue
        p = bitem.payload
        chunks: dict = {}
        for idx, (title, sents) in enumerate(
            zip(p.get("context_titles", []), p.get("context_sentences", []))
        ):
            chunk_id = f"c{bitem.item_id}_{idx}"
            chunk_data = {"content": f"[{title}] " + " ".join(sents), "doc_ids": [title]}
            chunks[chunk_id] = chunk_data
        eval_items.append({
            "question": p["question"], "answer": bitem.target["answer"],
            "query_id": bitem.item_id, "chunks": chunks,
        })
    return eval_items


def _frozen_to_eval_items_ultradomain(frozen_sample: list) -> list[dict]:
    """Convert frozen sample to UltraDomain eval dicts."""
    eval_items: list[dict] = []
    for bitem in frozen_sample:
        if _is_plain_dict(bitem):
            eval_items.append(bitem)
            continue
        ctx = (bitem.payload.get("context") or "")[:3000]
        domain = bitem.metadata.get("domain", "unknown")
        chunk_id = f"c{bitem.item_id}"
        if ctx:
            chunk_data = {"content": ctx, "doc_ids": [domain]}
            chunks = {chunk_id: chunk_data}
        else:
            chunks = {}
        eval_items.append({
            "question": bitem.payload.get("query") or "",
            "answer": bitem.target.get("answer") or "",
            "domain": domain, "query_id": bitem.item_id, "chunks": chunks,
        })
    return eval_items


def _collect_chunks(eval_items: list[dict]) -> dict:
    """Aggregate all ``chunks`` dicts from eval items (for corpus building)."""
    all_chunks: dict = {}
    for item in eval_items:
        for k, v in item.get("chunks", {}).items():
            all_chunks[k] = v
    return all_chunks


def build_frozen_samples(budget: int = 30, seed: int = 42) -> dict[str, list]:
    """Pre-draw a frozen sample per cluster using SamplingEngine.

    Returns ``{cluster_id: list[BenchmarkItem]}``.
    Only builds samples for clusters whose HF data is available locally.
    """
    import os
    from pathlib import Path
    from bsamp.sampling.engine import SamplingEngine

    _hf = Path(os.environ.get("HF_HUB_DIR", str(Path.home() / ".cache" / "huggingface" / "hub")))
    samples: dict[str, list] = {}

    hq_root = _hf / "datasets--hotpotqa--hotpot_qa" / "snapshots" / "1908d6afbbead072334abe2965f91bd2709910ab"
    if hq_root.exists():
        from bsamp.sampling.adapters.hotpotqa import HotpotQAAdapter
        adapter = HotpotQAAdapter(str(hq_root))
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["hotpotqa"] = result.items

    ud_root = _hf / "datasets--TommyChien--UltraDomain" / "snapshots" / "aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"
    if ud_root.exists():
        from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
        adapter = UltraDomainAdapter(str(ud_root), target_domains=["physics", "cs", "legal"])
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["ultradomain"] = result.items

    alce_root = _hf / "datasets--princeton-nlp--ALCE-data" / "snapshots" / "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
    if alce_root.exists():
        from bsamp.sampling.adapters.alce import ALCEAdapter
        adapter = ALCEAdapter(str(alce_root), subsets=["asqa"])
        result = SamplingEngine(adapter=adapter, method="proportional", budget=budget, seed=seed).run()
        samples["alce"] = result.items

    return samples


def build_frozen_samples_real(
    data_dir: str,
    max_items: int = 30,
    ud_domain: str = "mix",
) -> dict[str, list]:
    """Load real datasets from ``data_dir`` and wrap them as frozen eval dicts.

    Unlike ``build_frozen_samples()`` (which uses HuggingFace parquet via
    ``SamplingEngine``), this reads the raw JSON/JSONL files produced by the
    new ``load_hotpotqa_real`` / ``load_ultradomain_real`` loaders and
    returns plain eval-item dicts that ``_evaluate_frozen_real()`` consumes
    directly.

    Supported layouts::

        data_dir/
          all_data/hotpotqa/hotpotqa.json    (original)
          UltraDomain/mix.jsonl

        OR (fullwiki layout):

        data_dir/
          fullwiki/fullwiki_sample_500_uniform.parquet
          ultradomain/mix.jsonl
    """
    from pathlib import Path
    samples: dict[str, list] = {}

    hotpotqa_dir = Path(data_dir) / "all_data" / "hotpotqa"
    fullwiki_parquet = Path(data_dir) / "fullwiki" / "fullwiki_sample_500_uniform.parquet"

    if hotpotqa_dir.exists():
        from benchmark.hotpotqa_adapter import load_hotpotqa_real
        items = load_hotpotqa_real(hotpotqa_dir, max_items=max_items)
        samples["hotpotqa"] = items
    elif fullwiki_parquet.exists():
        items = _load_hotpotqa_parquet(fullwiki_parquet, max_items=max_items)
        samples["hotpotqa"] = items

    ud_dir = Path(data_dir) / "UltraDomain"
    ud_dir_alt = Path(data_dir) / "ultradomain"
    if ud_dir.exists():
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ud_dir, domain=ud_domain, max_items=max_items)
        samples["ultradomain"] = items
    elif ud_dir_alt.exists():
        from benchmark.ultradomain_adapter import load_ultradomain_real
        items = load_ultradomain_real(ud_dir_alt, domain=ud_domain, max_items=max_items)
        samples["ultradomain"] = items

    return samples


def _load_hotpotqa_parquet(path, max_items: int | None = None) -> list[dict]:
    """Load HotpotQA items from the fullwiki parquet file."""
    import pandas as pd
    from benchmark.base_adapter import hotpotqa_context_to_retrieval_results

    df = pd.read_parquet(path)
    if max_items is not None:
        df = df.head(max_items)

    items: list[dict] = []
    for _, row in df.iterrows():
        context_raw = row.get("context", [])
        if isinstance(context_raw, list) and context_raw:
            context_results = hotpotqa_context_to_retrieval_results(context_raw)
        else:
            context_results = []
        items.append({
            "question": row["question"],
            "answer": row["answer"],
            "query_id": row.get("id", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "context_results": context_results,
        })
    return items


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
        options = list(SLOT_OPTIONS[depth])

        # Prune retrieval options that violate chunking constraint
        if depth == 2 and len(self.choices) >= 1:
            chunking = self.choices[0]
            options = [
                o for o in options
                if CHUNKING_RETRIEVAL_CONSTRAINTS.get(o, chunking) == chunking
            ]

        return options

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
    # SelfRAG critique scoring helps with multi-hop (HotpotQA)
    ("selfrag_critique", "hotpotqa"):   0.03,
    # Cross-encoder helps with precision (HotpotQA multi-hop)
    ("cross_encoder", "hotpotqa"):      0.02,
    # LightRAG compression helps ALCE (citation quality)
    ("lightrag_compress", "alce"):      0.03,
    # BM25 is strong for factoid lookup
    ("bm25", "hotpotqa"):               0.02,
    # Hybrid retrieval benefits domain-specific queries
    ("bm25_dense_hybrid", "ultradomain"): 0.03,
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
    When ``frozen_samples`` is provided, uses pre-drawn BenchmarkItem lists
    instead of loading from ``sample_data/``.

    For each dataset cluster:
    1. Computes reward (simulated or real)
    2. Applies dataset-specific affinity bonuses (simulated only)
    3. Tracks path-prefix reuse for cost optimization
    """

    def __init__(self, use_real: bool = False,
                 frozen_samples: dict[str, list] | None = None) -> None:
        self.use_real = use_real
        self.frozen_samples = frozen_samples or {}

    def evaluate(self, state: SearchState, context: SearchContext) -> EvaluationResult:
        assert state.is_terminal(), "evaluator expects a terminal state"
        choices: tuple[str, ...] = state.state_key()  # type: ignore[assignment]

        cluster_results: list[BenchmarkClusterResult] = []
        total_reward = 0.0
        total_cost = 0.0
        weight = 1.0 / len(CLUSTER_IDS)

        for cid in CLUSTER_IDS:
            if self.use_real:
                frozen = self.frozen_samples.get(cid)
                reward = evaluate_config_real(choices, benchmark=cid, frozen_sample=frozen)
            else:
                reward = _compute_reward(choices) + CLUSTER_NOISE.get(cid, 0.0)
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
# WTB Cache-Aware Evaluator factory
# ---------------------------------------------------------------------------

def _build_wtb_evaluator(
    use_real: bool = False,
    frozen_samples: dict[str, list] | None = None,
) -> "Evaluator":
    """Build a ``RAGCacheAwareEvaluator`` backed by a WTB bench + ledger."""
    import tempfile
    from ominirag_wtb import RAGCacheAwareEvaluator, ReuseLedger

    tmp = tempfile.mkdtemp(prefix="wtb_reuse_")
    ledger = ReuseLedger(db_path=str(__import__("pathlib").Path(tmp) / "reuse_ledger.db"))

    bench = None
    if use_real:
        from wtb.sdk import WTBTestBench
        bench = WTBTestBench.create(mode="development", data_dir=tmp)

    bq_samples: dict[str, list] = {}
    if frozen_samples:
        from ominirag_wtb.config_types import BenchmarkQuestion
        for cid, items in frozen_samples.items():
            bq_samples[cid] = [
                BenchmarkQuestion.from_benchmark_item(it, cid) for it in items
            ]

    return RAGCacheAwareEvaluator(
        ledger=ledger,
        bench=bench,
        cluster_ids=CLUSTER_IDS,
        frozen_samples=bq_samples,
        cluster_costs=CLUSTER_COST,
        use_real=use_real,
        reward_table=REWARD_TABLE,
        default_reward=DEFAULT_REWARD,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_search_set(path: str) -> dict[str, list]:
    """Load a pre-built stratified search set JSON into frozen_samples format.

    The JSON is a list of dicts with ``question``, ``answer``, ``query_id``,
    ``type``, ``level`` (HotpotQA) or ``_domain``, ``input``, ``answers`` (UltraDomain).
    Auto-detects the benchmark from the data.
    """
    import json as _json
    from pathlib import Path as _P

    path_obj = _P(path)
    with open(path_obj, "r", encoding="utf-8") as f:
        items = _json.load(f)

    if not items:
        return {}

    if "question" in items[0]:
        eval_items = []
        for entry in items:
            eval_items.append({
                "question": entry["question"],
                "answer": entry["answer"],
                "query_id": entry.get("id", ""),
                "type": entry.get("type", ""),
                "level": entry.get("level", ""),
            })
        return {"hotpotqa": eval_items}
    elif "input" in items[0]:
        eval_items = []
        for entry in items:
            eval_items.append({
                "question": entry.get("input", ""),
                "answer": (entry.get("answers", [""])[0]
                           if isinstance(entry.get("answers"), list)
                           else entry.get("answers", "")),
                "query_id": entry.get("_id", ""),
                "domain": entry.get("_domain", "unknown"),
            })
        return {"ultradomain": eval_items}

    return {}


def _serialize_tree(root_node, slot_names: list[str]) -> dict:
    """Recursively serialize a TreeNode into a JSON-safe dict."""
    configs_visited: list[dict] = []

    def _collect(node, depth=0, prefix=()):
        if depth == len(slot_names):
            if node.visit_count > 0:
                configs_visited.append({
                    "config": {slot_names[i]: v for i, v in enumerate(prefix)},
                    "config_tuple": list(prefix),
                    "q_value": node.q_value,
                    "visit_count": node.visit_count,
                    "value_sum": node.value_sum,
                    "best_value": node.best_value,
                })
            return
        for act, ch in node.children.items():
            _collect(ch, depth + 1, prefix + (str(act),))

    _collect(root_node)
    configs_visited.sort(key=lambda x: x["q_value"], reverse=True)

    slot_breakdown = {}
    for action, child in root_node.children.items():
        slot_breakdown[str(action)] = {
            "visit_count": child.visit_count,
            "q_value": child.q_value,
            "best_value": child.best_value,
        }

    return {
        "slot_breakdown": slot_breakdown,
        "configs_visited": configs_visited,
    }


def main() -> None:
    from pathlib import Path as _PP
    _env = _PP(__file__).resolve().parents[3] / ".env"
    if _env.exists():
        from dotenv import load_dotenv
        load_dotenv(_env)

    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="AG-UCT RAG pipeline search")
    parser.add_argument("--real", action="store_true", help="Use real pipeline evaluation (sample_data)")
    parser.add_argument("--use-hf", action="store_true",
                        help="Use HuggingFace datasets via SamplingEngine (implies --real)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to real dataset root (e.g. /data1/ragworkspace/dataset). "
                             "Loads HotpotQA and UltraDomain from raw JSON/JSONL. Implies --real.")
    parser.add_argument("--search-set", type=str, default=None,
                        help="Path to pre-built stratified search set JSON "
                             "(from build_stratified_search_set.py). Implies --real.")
    parser.add_argument("--ud-domain", type=str, default="mix",
                        help="UltraDomain domain file to load when --data-dir is set (default: mix)")
    parser.add_argument("--wtb-reuse", action="store_true",
                        help="Use WTB cache-aware evaluator with bipartite reuse ledger")
    parser.add_argument("--budget", type=int, default=30,
                        help="Per-cluster sample budget (default: 30)")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-tree", type=str, default=None,
                        help="Save search tree (node visits, Q-values, configs) to JSON file")
    parser.add_argument("--output-stats", type=str, default=None,
                        help="Save SearchResult summary to JSON file")
    args = parser.parse_args()

    use_real = args.real or args.use_hf or args.data_dir is not None or args.search_set is not None
    frozen_samples: dict[str, list] | None = None

    if args.search_set is not None:
        print(f"Loading stratified search set from {args.search_set} ...", flush=True)
        frozen_samples = load_search_set(args.search_set)
        for cid, items in frozen_samples.items():
            print(f"  {cid}: {len(items)} items", flush=True)
        if not frozen_samples:
            print("  WARNING: Empty search set. Falling back to sample_data.", flush=True)
    elif args.data_dir is not None:
        print(f"Loading real datasets from {args.data_dir} ...", flush=True)
        frozen_samples = build_frozen_samples_real(
            data_dir=args.data_dir,
            max_items=args.budget,
            ud_domain=args.ud_domain,
        )
        for cid, items in frozen_samples.items():
            print(f"  {cid}: {len(items)} items", flush=True)
        if not frozen_samples:
            print("  WARNING: No real datasets found. Falling back to sample_data.", flush=True)
    elif args.use_hf:
        print("Building frozen HF samples ...", flush=True)
        frozen_samples = build_frozen_samples(budget=args.budget, seed=args.seed)
        for cid, items in frozen_samples.items():
            print(f"  {cid}: {len(items)} items", flush=True)
        if not frozen_samples:
            print("  WARNING: No HF datasets found locally. Falling back to sample_data.", flush=True)

    if args.wtb_reuse:
        evaluator = _build_wtb_evaluator(
            use_real=use_real, frozen_samples=frozen_samples,
        )
    else:
        evaluator = RAGPipelineEvaluator(use_real=use_real, frozen_samples=frozen_samples)
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
        random_seed=args.seed,
    )

    root = RAGPipelineSearchState()
    result = engine.search(root, max_iterations=args.max_iterations)

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

    if args.save_tree:
        tree_path = _PP(args.save_tree)
        tree_path.parent.mkdir(parents=True, exist_ok=True)
        tree_data = _serialize_tree(result.root_node, SLOT_NAMES)
        tree_data["best_config"] = list(result.best_state.state_key())
        tree_data["best_reward"] = result.best_reward
        tree_data["iterations"] = result.iterations
        tree_data["total_evaluations"] = result.total_evaluations
        tree_data["total_cost"] = result.total_cost
        tree_data["materialized_keys_count"] = len(result.context.materialized_keys)
        tree_data["search_space_size"] = total_configs
        with open(tree_path, "w", encoding="utf-8") as f:
            _json.dump(tree_data, f, ensure_ascii=False, indent=2)
        print(f"\n  Search tree saved to {tree_path}", flush=True)

    if args.output_stats:
        stats_path = _PP(args.output_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "best_config": list(result.best_state.state_key()),
            "best_config_pretty": result.best_state.pretty(),
            "best_reward": result.best_reward,
            "iterations": result.iterations,
            "total_evaluations": result.total_evaluations,
            "total_cost": result.total_cost,
            "materialized_keys_count": len(result.context.materialized_keys),
            "search_space_size": total_configs,
            "efficiency": result.total_evaluations / total_configs,
        }
        with open(stats_path, "w", encoding="utf-8") as f:
            _json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"  Search stats saved to {stats_path}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
