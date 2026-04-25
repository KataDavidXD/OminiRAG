"""FreshWiki dataset adapter.

Maps FreshWikiAPI._build_document() output to BenchmarkItem:
  - payload:  {"topic", "text", "content"}       -- RAG pipeline input
  - target:   {"summary", "url"}                  -- reference answers for scoring
  - metadata: {"predicted_class", "quality_bucket", "text_length", "n_sections", ...}

quality_bucket collapses Wikipedia's six quality classes into three tiers:
  Stub/Start -> low,  C/B -> mid,  GA/FA -> high
"""

from __future__ import annotations

from typing import Any

from bsamp.loader.FreshWiki import FreshWikiAPI
from bsamp.sampling.adapters.base import BenchmarkAdapter
from bsamp.sampling.types import BenchmarkItem

_QUALITY_BUCKET = {
    "Stub": "low",
    "Start": "low",
    "C": "mid",
    "B": "mid",
    "GA": "high",
    "FA": "high",
}


class FreshWikiAdapter(BenchmarkAdapter):
    """Wraps FreshWikiAPI and emits BenchmarkItem with payload/target/metadata."""

    def __init__(self, root_dir: str) -> None:
        self._api = FreshWikiAPI(root_dir)
        self._items: list[BenchmarkItem] | None = None

    @property
    def name(self) -> str:
        return "freshwiki"

    def population_size(self) -> int:
        return len(self.load_items())

    def available_strata_keys(self) -> list[str]:
        return ["predicted_class", "quality_bucket", "text_length"]

    def load_items(self) -> list[BenchmarkItem]:
        if self._items is not None:
            return self._items

        docs = self._api.load_documents()
        items: list[BenchmarkItem] = []

        for doc in docs:
            predicted_class = doc.get("predicted_class") or "unknown"
            quality_bucket = _QUALITY_BUCKET.get(predicted_class, "unknown")
            text_length = len(doc.get("text", ""))
            n_sections = len(doc.get("content", []))

            item = BenchmarkItem(
                item_id=f"freshwiki::{doc['id']}",
                benchmark="freshwiki",
                stratum="",
                payload={
                    "topic": doc.get("topic"),
                    "text": doc.get("text"),
                    "content": doc.get("content"),
                },
                target={
                    "summary": doc.get("summary"),
                    "url": doc.get("url"),
                },
                metadata={
                    "predicted_class": predicted_class,
                    "quality_bucket": quality_bucket,
                    "predicted_scores": doc.get("predicted_scores"),
                    "text_length": text_length,
                    "n_sections": n_sections,
                    "title": doc.get("title"),
                },
            )
            items.append(item)

        self._items = items
        return items
