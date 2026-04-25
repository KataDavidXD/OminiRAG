"""Abstract base class for benchmark dataset adapters.

Adapters are responsible for:
  1. Wrapping existing Loader APIs (FreshWikiAPI / UltraDomainAPI)
  2. Converting raw rows into BenchmarkItem (payload / target / metadata split)
  3. NOT re-implementing data-loading logic

To add a new benchmark, subclass and implement load_items() / population_size() /
available_strata_keys().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from bsamp.sampling.types import BenchmarkItem


class BenchmarkAdapter(ABC):
    """Loads a benchmark dataset and converts rows to BenchmarkItem instances.

    Subclasses wrap an existing loader API (FreshWikiAPI / UltraDomainAPI)
    and are responsible for the payload / target / metadata split.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'freshwiki' or 'ultradomain'."""

    @abstractmethod
    def load_items(self) -> list[BenchmarkItem]:
        """Return *all* benchmark items (full population)."""

    @abstractmethod
    def population_size(self) -> int:
        """Total N for this benchmark."""

    @abstractmethod
    def available_strata_keys(self) -> list[str]:
        """Metadata keys that can be used for stratification."""
