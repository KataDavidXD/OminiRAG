"""Core data types for the sampling subsystem.

Defines the full data model:
  - BenchmarkItem:    single benchmark evaluation unit (payload / target / metadata split)
  - EvalRecord:       one evaluated observation (item_id + reward + provenance)
  - StratumStats:     running statistics for a single stratum (online mean / variance / SE)
  - Estimate:         stratified estimator output with confidence interval
  - ItemRealization:  allocation vector + concrete realised item_ids (Layer 1 vs Layer 2)
  - SamplingState:    full checkpointable session state (JSON-serialisable, supports rollback)
  - CacheKey:         cache lookup key (config_hash, item_id)

All types implement to_dict / from_dict round-trip serialisation.
SamplingState additionally provides to_json / from_json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, NamedTuple


@dataclass(frozen=True)
class BenchmarkItem:
    """A single benchmark evaluation unit.

    The three-way split keeps a clean contract:
      - payload  -> input to run_rag()
      - target   -> ground truth for compute_reward()
      - metadata -> stratification / logging only
    """

    item_id: str
    benchmark: str
    stratum: str
    payload: dict[str, Any]
    target: dict[str, Any] | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "stratum": self.stratum,
            "payload": self.payload,
            "target": self.target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkItem:
        return cls(
            item_id=d["item_id"],
            benchmark=d["benchmark"],
            stratum=d["stratum"],
            payload=d["payload"],
            target=d["target"],
            metadata=d["metadata"],
        )


@dataclass
class EvalRecord:
    """One evaluated (item, reward) observation with provenance."""

    item_id: str
    stratum: str
    reward: float
    step: int
    cached: bool
    wall_time_ms: float
    allocation_snapshot: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRecord:
        return cls(**d)


@dataclass
class StratumStats:
    """Running statistics for a single stratum."""

    stratum: str
    population_size: int
    sample_size: int = 0
    running_sum: float = 0.0
    running_sum_sq: float = 0.0

    def update(self, reward: float) -> None:
        self.sample_size += 1
        self.running_sum += reward
        self.running_sum_sq += reward * reward

    @property
    def mean(self) -> float:
        if self.sample_size == 0:
            return 0.0
        return self.running_sum / self.sample_size

    @property
    def variance(self) -> float:
        if self.sample_size < 2:
            return 0.0
        mean = self.mean
        return (self.running_sum_sq / self.sample_size) - mean * mean

    @property
    def sample_variance(self) -> float:
        """Bessel-corrected sample variance s^2 = sum(x-xbar)^2 / (n-1)."""
        if self.sample_size < 2:
            return 0.0
        mean = self.mean
        ss = self.running_sum_sq - 2 * mean * self.running_sum + self.sample_size * mean * mean
        return ss / (self.sample_size - 1)

    @property
    def std_error(self) -> float:
        if self.sample_size < 2:
            return float("inf")
        fpc = 1.0 - self.sample_size / self.population_size
        return math.sqrt(self.sample_variance / self.sample_size * max(fpc, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StratumStats:
        return cls(**d)


@dataclass
class Estimate:
    """Point estimate with confidence interval from stratified sampling."""

    mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_evaluated: int
    n_total: int
    strata: dict[str, StratumStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std_error": self.std_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
            "n_evaluated": self.n_evaluated,
            "n_total": self.n_total,
            "strata": {k: v.to_dict() for k, v in self.strata.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Estimate:
        strata = {k: StratumStats.from_dict(v) for k, v in d["strata"].items()}
        return cls(
            mean=d["mean"],
            std_error=d["std_error"],
            ci_lower=d["ci_lower"],
            ci_upper=d["ci_upper"],
            confidence=d["confidence"],
            n_evaluated=d["n_evaluated"],
            n_total=d["n_total"],
            strata=strata,
        )


@dataclass
class ItemRealization:
    """Concrete item draw for a given allocation.

    Allocation (Layer 1) determines *how many* per stratum.
    Realization (Layer 2) determines *which ones*.
    """

    allocation: list[int]
    realized_items: list[str]
    rng_state_before: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation": self.allocation,
            "realized_items": self.realized_items,
            "rng_state_before": _serialise_rng_state(self.rng_state_before),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ItemRealization:
        return cls(
            allocation=d["allocation"],
            realized_items=d["realized_items"],
            rng_state_before=_deserialise_rng_state(d["rng_state_before"]),
        )


@dataclass
class SamplingState:
    """Full checkpointable state of a sampling session."""

    config_id: str
    benchmark: str
    sampler_type: str
    budget_total: int
    budget_used: int
    strata_stats: dict[str, StratumStats]
    sampler_state: dict[str, Any]
    rng_state: Any
    history: list[EvalRecord]
    realizations: list[ItemRealization]
    stopped: bool = False
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "benchmark": self.benchmark,
            "sampler_type": self.sampler_type,
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
            "strata_stats": {k: v.to_dict() for k, v in self.strata_stats.items()},
            "sampler_state": self.sampler_state,
            "rng_state": _serialise_rng_state(self.rng_state),
            "history": [r.to_dict() for r in self.history],
            "realizations": [r.to_dict() for r in self.realizations],
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SamplingState:
        return cls(
            config_id=d["config_id"],
            benchmark=d["benchmark"],
            sampler_type=d["sampler_type"],
            budget_total=d["budget_total"],
            budget_used=d["budget_used"],
            strata_stats={k: StratumStats.from_dict(v) for k, v in d["strata_stats"].items()},
            sampler_state=d["sampler_state"],
            rng_state=_deserialise_rng_state(d["rng_state"]),
            history=[EvalRecord.from_dict(r) for r in d["history"]],
            realizations=[ItemRealization.from_dict(r) for r in d["realizations"]],
            stopped=d.get("stopped", False),
            stop_reason=d.get("stop_reason"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> SamplingState:
        return cls.from_dict(json.loads(s))


class CacheKey(NamedTuple):
    config_hash: str
    item_id: str


# ---------------------------------------------------------------------------
# RNG state (de)serialisation helpers for Python's random.Random
# ---------------------------------------------------------------------------

def _serialise_rng_state(state: Any) -> Any:
    """Convert random.getstate() tuple to JSON-safe form."""
    if state is None:
        return None
    if isinstance(state, (list, dict, str, int, float, bool)):
        return state
    if isinstance(state, tuple):
        return [_serialise_rng_state(e) for e in state]
    return str(state)


def _deserialise_rng_state(state: Any) -> Any:
    """Reverse of _serialise_rng_state. Returns tuple tree."""
    if state is None:
        return None
    if isinstance(state, list):
        inner = tuple(_deserialise_rng_state(e) for e in state)
        return inner
    return state
