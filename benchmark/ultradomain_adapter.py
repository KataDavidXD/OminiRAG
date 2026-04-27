"""UltraDomain benchmark evaluation adapter for rag_contracts pipelines.

Scoring logic (LLM-as-judge + pairwise comparison) lives in
``bsamp.scoring``; this module is the thin glue between OminiRAG's
``Generation`` protocol and the scoring SDK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rag_contracts import GenerationResult, RetrievalResult

from bsamp.scoring import (
    UltraDomainEvaluator,
    compute_f1,
    normalize_answer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Re-exports for backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "normalize_answer",
    "compute_f1",
    "load_ultradomain_sample",
    "load_ultradomain_jsonl",
    "sample_chunks_to_retrieval_results",
    "UltraDomainEvaluationResult",
    "UltraDomainBenchmarkAdapter",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_ultradomain_sample(sample_dir: str | Path) -> list[dict]:
    """Load from the KG sample data directory."""
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
            "domain": q.get("domain", "general"),
            "query_id": q.get("query_id", ""),
            "chunks": chunks,
        })
    return items


def load_ultradomain_jsonl(path: str | Path) -> list[dict]:
    """Load UltraDomain from a JSONL file."""
    items = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_chunks_to_retrieval_results(chunks: dict) -> list[RetrievalResult]:
    """Convert KG sample chunks dict to RetrievalResult list."""
    results = []
    for cid, info in chunks.items():
        doc_ids = info.get("doc_ids", [])
        results.append(RetrievalResult(
            source_id=cid,
            content=info.get("content", ""),
            score=1.0,
            title=doc_ids[0] if doc_ids else cid,
            metadata={"doc_ids": doc_ids},
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UltraDomainEvaluationResult:
    """Aggregated evaluation results across all items."""
    avg_comprehensiveness: float = 0.0
    avg_diversity: float = 0.0
    avg_empowerment: float = 0.0
    avg_f1: float = 0.0
    avg_length: float = 0.0
    num_items: int = 0
    per_item: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UltraDomainBenchmarkAdapter:
    """Evaluate any rag_contracts Generation against UltraDomain data.

    If ``llm_complete`` is provided, uses LLM-as-judge for the three domain
    metrics. Otherwise only computes token-F1 (useful for quick testing).

    Usage::

        adapter = UltraDomainBenchmarkAdapter(llm_complete=my_llm.complete)
        data = load_ultradomain_sample("benchmark/sample_data/ultradomain_kg_sample")
        results = adapter.evaluate_generation(data, my_generation)
    """

    llm_complete: Optional[Callable[..., str]] = None

    def evaluate_generation(
        self,
        data: list[dict],
        generation: Any,
    ) -> UltraDomainEvaluationResult:
        """Run generation on each UltraDomain item and compute metrics."""
        evaluator = UltraDomainEvaluator(llm_fn=self.llm_complete)
        scored_items: list[dict] = []

        for item in data:
            question = item["question"]
            domain = item.get("domain", "general")
            chunks = item.get("chunks", {})
            context = sample_chunks_to_retrieval_results(chunks)

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
            )
            output = gen_result.output.strip()

            ref_answer = item.get("answer", "")
            if isinstance(ref_answer, list):
                ref_answer = ref_answer[0] if ref_answer else ""

            scored_items.append({
                "prediction": output,
                "answer": ref_answer,
                "question": question,
                "domain": domain,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "domain": si["domain"],
                "output": si["prediction"],
                "comprehensiveness": score_obj.metrics["comprehensiveness"],
                "diversity": score_obj.metrics["diversity"],
                "empowerment": score_obj.metrics["empowerment"],
                "f1": score_obj.metrics["f1"],
                "length": score_obj.metrics["length"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return UltraDomainEvaluationResult(
            avg_comprehensiveness=result.aggregate["avg_comprehensiveness"],
            avg_diversity=result.aggregate["avg_diversity"],
            avg_empowerment=result.aggregate["avg_empowerment"],
            avg_f1=result.aggregate["avg_f1"],
            avg_length=result.aggregate["avg_length"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph: Any,
    ) -> UltraDomainEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute metrics."""
        import asyncio

        evaluator = UltraDomainEvaluator(llm_fn=self.llm_complete)
        scored_items: list[dict] = []

        for item in data:
            question = item["question"]
            domain = item.get("domain", "general")
            state = asyncio.run(graph.ainvoke({"query": question}))

            gen_result: GenerationResult = state.get(
                "generation_result", GenerationResult(output="")
            )
            output = gen_result.output.strip()

            ref_answer = item.get("answer", "")
            if isinstance(ref_answer, list):
                ref_answer = ref_answer[0] if ref_answer else ""

            scored_items.append({
                "prediction": output,
                "answer": ref_answer,
                "question": question,
                "domain": domain,
                "query_id": item.get("query_id", ""),
                "_citations": gen_result.citations,
            })

        result = evaluator.score_batch(scored_items)

        per_item = []
        for si, score_obj in zip(scored_items, result.per_item):
            per_item.append({
                "question": si["question"],
                "domain": si["domain"],
                "output": si["prediction"],
                "comprehensiveness": score_obj.metrics["comprehensiveness"],
                "diversity": score_obj.metrics["diversity"],
                "empowerment": score_obj.metrics["empowerment"],
                "f1": score_obj.metrics["f1"],
                "length": score_obj.metrics["length"],
                "query_id": si["query_id"],
                "citations": si["_citations"],
            })

        return UltraDomainEvaluationResult(
            avg_comprehensiveness=result.aggregate["avg_comprehensiveness"],
            avg_diversity=result.aggregate["avg_diversity"],
            avg_empowerment=result.aggregate["avg_empowerment"],
            avg_f1=result.aggregate["avg_f1"],
            avg_length=result.aggregate["avg_length"],
            num_items=result.num_items,
            per_item=per_item,
        )

    def save_results(
        self,
        results: UltraDomainEvaluationResult,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "avg_comprehensiveness": results.avg_comprehensiveness,
                    "avg_diversity": results.avg_diversity,
                    "avg_empowerment": results.avg_empowerment,
                    "avg_f1": results.avg_f1,
                    "avg_length": results.avg_length,
                    "num_items": results.num_items,
                },
                "per_item": results.per_item,
            }, f, ensure_ascii=False, indent=2)
