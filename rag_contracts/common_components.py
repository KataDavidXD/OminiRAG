"""Reusable retrieval and generation components for rag_contracts pipelines.

These are corpus-agnostic components that work with any pipeline:
- LLMRetrieval: generates background context via LLM
- DuckDuckGoRetrieval: web search via DuckDuckGo
- FallbackRetrieval: chains a primary + fallback retrieval
- ALCEDocRetrieval: wraps pre-retrieved ALCE documents as a Retrieval component
- SimpleLLMGeneration: LLM-based answer extraction (LongRAG reader style)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import GenerationResult, RetrievalResult


@dataclass
class LLMRetrieval:
    """Asks an LLM to produce background context for each query.

    The ``llm`` object must have a ``complete(system, user, **kwargs)`` method
    that returns a string.  No corpus or index required.
    """

    llm: Any

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for i, q in enumerate(queries[:top_k]):
            ctx = self.llm.complete(
                "You are a research assistant. Provide concise, factual background "
                "information that would help answer the following question. "
                "Include key facts, dates, and names.",
                q,
                temperature=0.3,
                max_tokens=400,
            )
            results.append(
                RetrievalResult(
                    source_id=f"llm-context://{i}",
                    content=ctx,
                    score=1.0,
                    title=f"LLM-generated context for: {q}",
                )
            )
        return results


@dataclass
class DuckDuckGoRetrieval:
    """Web search retrieval via DuckDuckGo (``ddgs`` package)."""

    k: int = 5

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        from ddgs import DDGS

        ddgs = DDGS()
        seen: set[str] = set()
        results: list[RetrievalResult] = []
        for q in queries:
            try:
                hits = ddgs.text(q, max_results=self.k)
            except Exception:
                hits = []
            for hit in hits or []:
                url = hit.get("href", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                results.append(
                    RetrievalResult(
                        source_id=url,
                        content=hit.get("body", ""),
                        score=0.9,
                        title=hit.get("title", ""),
                    )
                )
        return results[:top_k]


@dataclass
class FallbackRetrieval:
    """Chains two retrieval components: falls back to the second when the
    primary returns zero results."""

    primary: Any
    fallback: Any

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = self.primary.retrieve(queries, top_k=top_k)
        if results:
            return results
        return self.fallback.retrieve(queries, top_k=top_k)


@dataclass
class ALCEDocRetrieval:
    """Wraps ALCE pre-retrieved documents as a ``rag_contracts.Retrieval``.

    ALCE benchmark items provide N documents per question.  This adapter
    converts them into ``RetrievalResult`` objects so the documents can
    flow through any standard pipeline's retrieval slot.

    Usage::

        adapter = ALCEDocRetrieval(docs=alce_item["docs"])
        results = adapter.retrieve(["any query"])
    """

    docs: list[dict] = field(default_factory=list)

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for idx, doc in enumerate(self.docs[:top_k]):
            results.append(RetrievalResult(
                source_id=str(idx + 1),
                content=doc.get("text", ""),
                score=1.0 - idx * 0.01,
                title=doc.get("title", ""),
                metadata={
                    "doc_index": idx,
                    "summary": doc.get("summary", ""),
                    "extraction": doc.get("extraction", ""),
                },
            ))
        return results


@dataclass
class SimpleLLMGeneration:
    """Extracts a concise answer from context via LLM (LongRAG reader style).

    The ``llm`` object must have a ``complete(system, user, **kwargs)`` method.
    """

    llm: Any

    def generate(
        self, query: str, context: list[RetrievalResult], instruction: str = ""
    ) -> GenerationResult:
        ctx_text = "\n\n---\n\n".join(
            f"[{i+1}] {r.title}\n{r.content}" for i, r in enumerate(context[:5])
        )
        answer = self.llm.complete(
            "You are an expert reader. Extract the answer from the provided context. "
            "Be concise and precise. If the answer is a short entity, return just that.",
            f"Context:\n{ctx_text}\n\nQuestion: {query}\n\nAnswer:",
            temperature=0.1,
            max_tokens=300,
        )
        return GenerationResult(
            output=answer.strip(),
            citations=[r.source_id for r in context[:5]],
            metadata={"style": "llm-reader"},
        )
