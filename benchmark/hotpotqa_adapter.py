"""HotpotQA benchmark evaluation adapter for rag_contracts pipelines.

Loads HotpotQA data (from HuggingFace, KG sample, or local JSONL), runs it
through any pipeline conforming to canonical protocols, and evaluates outputs
using EM and token-F1 metrics (the standard HotpotQA evaluation).

Scoring logic lives in ``bsamp.scoring``; this module is the thin glue
between OminiRAG's ``Generation`` protocol and the scoring SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult

from benchmark.base_adapter import (
    get_context_for_item,
    hotpotqa_context_to_retrieval_results,
    invoke_graph_sync,
    sample_chunks_to_retrieval_results,
)
from bsamp.scoring import (
    HotpotQAEvaluator,
    compute_exact,
    compute_f1,
    normalize_answer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Re-exports kept for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "normalize_answer",
    "compute_f1",
    "compute_exact",
    "load_hotpotqa_sample",
    "load_hotpotqa_jsonl",
    "load_hotpotqa_real",
    "sample_chunks_to_retrieval_results",
    "HotpotQAEvaluationResult",
    "HotpotQABenchmarkAdapter",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_hotpotqa_sample(sample_dir: str | Path) -> list[dict]:
    """Load from the KG sample data directory.

    Reads ``queries.jsonl`` and ``chunks.json`` to produce items with
    ``question``, ``answer``, and ``context_chunks``.
    """
    sample_dir = Path(sample_dir)
    queries = []
    with open(sample_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    chunks: dict = {}
    chunks_path = sample_dir / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

    items = []
    for q in queries:
        items.append({
            "question": q["query"],
            "answer": q["ground_truth"],
            "query_id": q.get("query_id", ""),
            "chunks": chunks,
        })
    return items


def load_hotpotqa_jsonl(path: str | Path) -> list[dict]:
    """Load HotpotQA from a JSONL file with ``question``, ``answer``, ``context`` fields."""
    items = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_hotpotqa_real(
    data_dir: str | Path,
    *,
    split: str = "train",
    max_items: int | None = None,
) -> list[dict]:
    """Load real HotpotQA dataset from ``/data1/ragworkspace/dataset/all_data/hotpotqa``.

    Expects ``hotpotqa_<split>.json`` — a JSON file containing a list of dicts,
    each with ``question``, ``answer``, ``context`` (list of [title, [sents]]),
    ``type``, ``level``, and ``_id``.

    Each item is normalised into the format expected by ``evaluate_generation``,
    with ``context_results`` pre-built via ``hotpotqa_context_to_retrieval_results``.
    """
    data_dir = Path(data_dir)

    candidates = [
        data_dir / f"hotpotqa_{split}.json",
        data_dir / f"{split}.json",
        data_dir / "hotpotqa.json",
    ]
    src: Path | None = None
    for c in candidates:
        if c.exists():
            src = c
            break
    if src is None:
        raise FileNotFoundError(
            f"No HotpotQA JSON found in {data_dir}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    with open(src, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    if max_items is not None:
        raw = raw[:max_items]

    items: list[dict] = []
    for entry in raw:
        context_raw = entry.get("context", [])
        items.append({
            "question": entry["question"],
            "answer": entry.get("answer", ""),
            "query_id": entry.get("_id", ""),
            "type": entry.get("type", ""),
            "level": entry.get("level", ""),
            "context_results": hotpotqa_context_to_retrieval_results(context_raw),
        })
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HotpotQAEvaluationResult:
    """Aggregated evaluation results across all items."""
    avg_em: float = 0.0
    avg_f1: float = 0.0
    num_items: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class HotpotQABenchmarkAdapter:
    """Evaluate any rag_contracts Generation or pipeline against HotpotQA data.

    Usage::

        adapter = HotpotQABenchmarkAdapter()
        data = load_hotpotqa_sample("benchmark/sample_data/hotpotqa_kg_sample")
        results = adapter.evaluate_generation(data, my_generation)
    """

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
    ) -> HotpotQAEvaluationResult:
        """Run generation on each HotpotQA item and compute EM/F1."""
        evaluator = HotpotQAEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item["question"]
            context = get_context_for_item(item)

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
            )
            output = gen_result.output.strip()
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            scored_items.append({
                "prediction": output,
                "answer": answer,
                "question": question,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "answer": si["answer"],
                "output": si["prediction"],
                "em": score_obj.metrics["em"],
                "f1": score_obj.metrics["f1"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return HotpotQAEvaluationResult(
            avg_em=result.aggregate["avg_em"],
            avg_f1=result.aggregate["avg_f1"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph: Any,
    ) -> HotpotQAEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute EM/F1."""
        evaluator = HotpotQAEvaluator()
        scored_items: list[dict] = []

        for item in data:
            question = item["question"]
            state = invoke_graph_sync(graph, {
                "query": question,
                "query_id": item.get("query_id", ""),
                "answers": [item.get("answer", "")],
                "test_data_name": "hotpotqa",
            })

            gen_result: GenerationResult = state.get(
                "generation_result", GenerationResult(output="")
            )
            output = gen_result.output.strip()
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            scored_items.append({
                "prediction": output,
                "answer": answer,
                "question": question,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "answer": si["answer"],
                "output": si["prediction"],
                "em": score_obj.metrics["em"],
                "f1": score_obj.metrics["f1"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return HotpotQAEvaluationResult(
            avg_em=result.aggregate["avg_em"],
            avg_f1=result.aggregate["avg_f1"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def save_results(
        self,
        results: HotpotQAEvaluationResult,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_em": results.avg_em,
                    "avg_f1": results.avg_f1,
                    "num_items": results.num_items,
                },
                "per_item": results.per_item,
            }, f, ensure_ascii=False, indent=2)
