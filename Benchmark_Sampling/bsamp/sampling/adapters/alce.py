"""ALCE dataset adapter for the sampling subsystem.

Maps ALCEAPI output to BenchmarkItem:
  - payload:  {"question", "docs"}             -- RAG pipeline input
  - target:   {"answer", "short_answers", "qa_pairs"}
  - metadata: {"subset", "n_docs", "n_qa_pairs"}

Stratification: ``subset`` (asqa|qampari|eli5).  No secondary length axis
because all items have 100 pre-retrieved docs.
"""

from __future__ import annotations

from typing import Sequence

from bsamp.loader.ALCE import ALCEAPI
from bsamp.sampling.adapters.base import BenchmarkAdapter
from bsamp.sampling.types import BenchmarkItem


class ALCEAdapter(BenchmarkAdapter):
    """Wraps ALCEAPI and emits BenchmarkItem."""

    def __init__(
        self,
        root_dir: str,
        subsets: Sequence[str] | None = None,
    ) -> None:
        self._api = ALCEAPI(root_dir)
        self._subsets = list(subsets) if subsets else self._api.available_subsets()
        self._items: list[BenchmarkItem] | None = None

    @property
    def name(self) -> str:
        return "alce"

    def population_size(self) -> int:
        return len(self.load_items())

    def available_strata_keys(self) -> list[str]:
        return ["subset", "n_docs", "n_qa_pairs"]

    def load_items(self) -> list[BenchmarkItem]:
        if self._items is not None:
            return self._items

        items: list[BenchmarkItem] = []
        for subset in self._subsets:
            raw_items = self._api.load_subset(subset)
            for raw in raw_items:
                item = BenchmarkItem(
                    item_id=f"alce::{raw['id']}",
                    benchmark="alce",
                    stratum="",
                    payload={
                        "question": raw["question"],
                        "docs": raw["docs"],
                    },
                    target={
                        "answer": raw["answer"],
                        "short_answers": raw["short_answers"],
                        "qa_pairs": raw["qa_pairs"],
                    },
                    metadata={
                        "subset": raw["subset"],
                        "n_docs": raw["n_docs"],
                        "n_qa_pairs": raw["n_qa_pairs"],
                    },
                )
                items.append(item)

        self._items = items
        return items
