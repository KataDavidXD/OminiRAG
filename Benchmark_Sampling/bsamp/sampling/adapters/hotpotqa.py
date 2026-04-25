"""HotpotQA dataset adapter for the sampling subsystem.

Maps HotpotQAAPI output to BenchmarkItem:
  - payload:  {"question", "context_text"}    -- RAG pipeline input
  - target:   {"answer", "supporting_facts_titles", "supporting_facts_sent_ids"}
  - metadata: {"type", "level", "context_length"}

Stratification: ``type`` (comparison|bridge) x ``level`` (easy|medium|hard)
"""

from __future__ import annotations

from bsamp.loader.hotpot_qa import HotpotQAAPI
from bsamp.sampling.adapters.base import BenchmarkAdapter
from bsamp.sampling.types import BenchmarkItem


class HotpotQAAdapter(BenchmarkAdapter):
    """Wraps HotpotQAAPI and emits BenchmarkItem."""

    def __init__(self, root_dir: str, split: str = "distractor") -> None:
        self._api = HotpotQAAPI(root_dir, split=split)
        self._items: list[BenchmarkItem] | None = None

    @property
    def name(self) -> str:
        return "hotpotqa"

    def population_size(self) -> int:
        return len(self.load_items())

    def available_strata_keys(self) -> list[str]:
        return ["type", "level", "context_length"]

    def load_items(self) -> list[BenchmarkItem]:
        if self._items is not None:
            return self._items

        raw_items = self._api.load_items()
        items: list[BenchmarkItem] = []

        for raw in raw_items:
            item = BenchmarkItem(
                item_id=f"hotpotqa::{raw['id']}",
                benchmark="hotpotqa",
                stratum="",
                payload={
                    "question": raw["question"],
                    "context_text": raw["context_text"],
                    "context_titles": raw["context_titles"],
                    "context_sentences": raw["context_sentences"],
                },
                target={
                    "answer": raw["answer"],
                    "supporting_facts_titles": raw["supporting_facts_titles"],
                    "supporting_facts_sent_ids": raw["supporting_facts_sent_ids"],
                },
                metadata={
                    "type": raw["type"],
                    "level": raw["level"],
                    "context_length": raw["context_length"],
                },
            )
            items.append(item)

        self._items = items
        return items
