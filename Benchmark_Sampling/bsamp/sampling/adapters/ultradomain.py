from __future__ import annotations

from typing import Any, Sequence

from bsamp.loader.UltraDomain import UltraDomainAPI
from bsamp.sampling.adapters.base import BenchmarkAdapter
from bsamp.sampling.types import BenchmarkItem


class UltraDomainAdapter(BenchmarkAdapter):
    """Wraps UltraDomainAPI and emits BenchmarkItem with payload/target/metadata."""

    def __init__(
        self,
        root_dir: str,
        target_domains: Sequence[str] | None = None,
    ) -> None:
        self._api = UltraDomainAPI(root_dir)
        self._target_domains = target_domains
        self._items: list[BenchmarkItem] | None = None

    @property
    def name(self) -> str:
        return "ultradomain"

    @property
    def available_domains(self) -> list[str]:
        return self._api.available_domains

    def population_size(self) -> int:
        return len(self.load_items())

    def available_strata_keys(self) -> list[str]:
        return ["domain", "length", "length_bucket"]

    def load_items(self) -> list[BenchmarkItem]:
        if self._items is not None:
            return self._items

        domain_map = self._api.load_domains(self._target_domains)

        items: list[BenchmarkItem] = []
        for domain, raw_rows in domain_map.items():
            for raw in raw_rows:
                std = self._api._to_standard_item(raw)
                item = BenchmarkItem(
                    item_id=std.get("id", f"ultradomain::{domain}::{id(raw)}"),
                    benchmark="ultradomain",
                    stratum="",
                    payload={
                        "query": std.get("query"),
                        "context": std.get("context"),
                    },
                    target={
                        "answers": std.get("answers"),
                        "answer": std.get("answer"),
                        "label": std.get("label"),
                    },
                    metadata={
                        "domain": std.get("domain", domain),
                        "length": std.get("length"),
                        "context_id": std.get("context_id"),
                        "title": std.get("title"),
                        "authors": std.get("authors"),
                    },
                )
                items.append(item)

        self._items = items
        return items
