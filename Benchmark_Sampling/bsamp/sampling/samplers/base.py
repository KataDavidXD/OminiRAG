"""Abstract base class for all samplers.

Every sampler (StratifiedSampler, MetropolisHastingsSampler, future HMCSampler)
implements this interface. Core methods:
  - ``select()``:    given strata, stats, and budget, return an ``ItemRealization``
  - ``get_state()``: return JSON-serialisable internal state (for checkpointing)
  - ``set_state()``: restore from serialised state (for rollback / fork)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bsamp.sampling.types import BenchmarkItem, ItemRealization, StratumStats


class BaseSampler(ABC):
    """Interface that all samplers (stratified, MH, HMC) implement."""

    @abstractmethod
    def select(
        self,
        strata: dict[str, list[BenchmarkItem]],
        strata_stats: dict[str, StratumStats],
        budget: int,
    ) -> ItemRealization:
        """Choose the next batch of items to evaluate.

        Returns an ``ItemRealization`` containing the allocation vector and
        the concrete realised item ids.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Serialisable sampler-specific state for checkpointing."""

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore sampler-specific state from checkpoint."""
