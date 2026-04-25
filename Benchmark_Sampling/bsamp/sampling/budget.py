"""Budget controller for evaluation call limits.

Tracks total and used budget. Can operate standalone or as a live view
over a ``SamplingState`` for a single source of truth:
  - Standalone:  ``BudgetController(total=200, used=0)``
  - State-bound: ``BudgetController.from_state(state)`` -- reads/writes ``state.budget_used``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bsamp.sampling.types import SamplingState


class BudgetController:
    """Tracks evaluation budget and enforces limits.

    Can operate standalone (total + used) or as a live view over a
    ``SamplingState`` so that ``budget_used`` has a single source of truth.
    """

    def __init__(self, total: int, *, used: int = 0, state: SamplingState | None = None) -> None:
        self._total = total
        self._state = state
        self._standalone_used = used

    @classmethod
    def from_state(cls, state: SamplingState) -> BudgetController:
        return cls(total=state.budget_total, state=state)

    @property
    def total(self) -> int:
        return self._total

    @property
    def used(self) -> int:
        if self._state is not None:
            return self._state.budget_used
        return self._standalone_used

    @used.setter
    def used(self, value: int) -> None:
        if self._state is not None:
            self._state.budget_used = value
        else:
            self._standalone_used = value

    @property
    def remaining(self) -> int:
        return max(self._total - self.used, 0)

    @property
    def fraction_used(self) -> float:
        if self._total <= 0:
            return 1.0
        return self.used / self._total

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def consume(self, n: int = 1) -> None:
        self.used = self.used + n

    def can_afford(self, n: int) -> bool:
        return self.remaining >= n

    def clamp(self, requested: int) -> int:
        """Return min(requested, remaining)."""
        return min(requested, self.remaining)
