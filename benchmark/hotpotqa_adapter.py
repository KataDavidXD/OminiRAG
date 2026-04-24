"""HotpotQA benchmark evaluation adapter for rag_contracts pipelines.

Loads HotpotQA data (from HuggingFace, KG sample, or local JSONL), runs it
through any pipeline conforming to canonical protocols, and evaluates outputs
using EM and token-F1 metrics (the standard HotpotQA evaluation).
"""

from __future__ import annotations

import collections
import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_contracts import GenerationResult, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════════════
# Text normalization (HotpotQA standard)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_answer(gold).split()
    pred_toks = normalize_answer(pred).split()

    if not gold_toks or not pred_toks:
        return float(gold_toks == pred_toks)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def compute_exact(gold: str, pred: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(pred))


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

    chunks = {}
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
        per_item = []
        em_scores = []
        f1_scores = []

        for item in data:
            question = item["question"]
            chunks = item.get("chunks", {})
            context = sample_chunks_to_retrieval_results(chunks)

            gen_result: GenerationResult = generation.generate(
                query=question,
                context=context,
            )

            output = gen_result.output.strip()
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            em = compute_exact(answer, output)
            f1 = compute_f1(answer, output)

            em_scores.append(em)
            f1_scores.append(f1)
            per_item.append({
                "question": question,
                "answer": answer,
                "output": output,
                "em": em,
                "f1": f1,
                "query_id": item.get("query_id", ""),
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return HotpotQAEvaluationResult(
            avg_em=100 * sum(em_scores) / n,
            avg_f1=100 * sum(f1_scores) / n,
            num_items=len(data),
            per_item=per_item,
        )

    def evaluate_pipeline(
        self,
        data: list[dict],
        graph: Any,
    ) -> HotpotQAEvaluationResult:
        """Run an entire LangGraph pipeline per item and compute EM/F1.

        The pipeline must produce a ``generation_result`` field in its output state.
        """
        import asyncio

        per_item = []
        em_scores = []
        f1_scores = []

        for item in data:
            question = item["question"]
            state = asyncio.run(graph.ainvoke({
                "query": question,
                "query_id": item.get("query_id", ""),
                "answers": [item.get("answer", "")],
                "test_data_name": "hotpotqa",
            }))

            gen_result: GenerationResult = state.get(
                "generation_result", GenerationResult(output="")
            )
            output = gen_result.output.strip()
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            em = compute_exact(answer, output)
            f1 = compute_f1(answer, output)

            em_scores.append(em)
            f1_scores.append(f1)
            per_item.append({
                "question": question,
                "answer": answer,
                "output": output,
                "em": em,
                "f1": f1,
                "query_id": item.get("query_id", ""),
                "citations": gen_result.citations,
            })

        n = len(data) or 1
        return HotpotQAEvaluationResult(
            avg_em=100 * sum(em_scores) / n,
            avg_f1=100 * sum(f1_scores) / n,
            num_items=len(data),
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
