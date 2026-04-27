"""Unified result types for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ItemScore:
    """Per-item evaluation result."""

    item_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    prediction: str = ""
    gold: str | list[str] = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregated evaluation result across a scored batch."""

    benchmark: str = ""
    num_items: int = 0
    aggregate: dict[str, float] = field(default_factory=dict)
    per_item: list[ItemScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
