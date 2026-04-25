"""Stratification module -- assigns BenchmarkItems to strata.

Partitions items by configurable features:
  - primary axis:   e.g. ``domain`` (UltraDomain) or ``quality_bucket`` (FreshWiki)
  - secondary axis: optional ``length`` binning (tercile / median split)
  - small-strata collapse: strata with population < min_stratum_size are merged
    into a catch-all ``{primary}::other`` bucket

Returns ``(stratum_name -> list[BenchmarkItem], stratum_name -> StratumStats)``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Sequence

from bsamp.sampling.types import BenchmarkItem, StratumStats


@dataclass
class StratificationConfig:
    primary: str = "domain"
    secondary: str | None = "length"
    length_bins: int = 3
    min_stratum_size: int = 5


def stratify(
    items: list[BenchmarkItem],
    config: StratificationConfig,
) -> tuple[dict[str, list[BenchmarkItem]], dict[str, StratumStats]]:
    """Assign each item to a stratum and return (stratum->items, stratum->stats).

    Items are returned with their ``stratum`` field set.
    Strata smaller than ``config.min_stratum_size`` are collapsed into
    a catch-all stratum labelled ``{primary}::other``.
    """
    length_edges = _compute_length_edges(items, config) if config.secondary == "length" else None

    buckets: dict[str, list[BenchmarkItem]] = defaultdict(list)
    for item in items:
        label = _assign_stratum(item, config, length_edges)
        updated = replace(item, stratum=label)
        buckets[label].append(updated)

    buckets, collapsed = _collapse_small_strata(buckets, config.min_stratum_size, config.primary)

    strata_stats: dict[str, StratumStats] = {}
    for label, members in buckets.items():
        strata_stats[label] = StratumStats(
            stratum=label,
            population_size=len(members),
        )

    return dict(buckets), strata_stats


def _assign_stratum(
    item: BenchmarkItem,
    config: StratificationConfig,
    length_edges: dict[str, list[float]] | None,
) -> str:
    meta = item.metadata
    primary_val = str(meta.get(config.primary, "unknown"))

    if config.secondary is None or config.secondary != "length":
        return primary_val

    raw_length = meta.get("length") or meta.get("text_length") or 0
    length_val = int(raw_length) if raw_length else 0

    if length_edges is None:
        return primary_val

    edges = length_edges.get(primary_val)
    if edges is None:
        bucket = "all"
    else:
        bucket = _length_bucket(length_val, edges)

    return f"{primary_val}::{bucket}"


def _length_bucket(length: int, edges: list[float]) -> str:
    for i, edge in enumerate(edges):
        if length <= edge:
            return f"len{i}"
    return f"len{len(edges)}"


def _compute_length_edges(
    items: list[BenchmarkItem],
    config: StratificationConfig,
) -> dict[str, list[float]]:
    """Compute within-group length tercile (or n-tile) edges."""
    groups: dict[str, list[int]] = defaultdict(list)
    for item in items:
        meta = item.metadata
        primary_val = str(meta.get(config.primary, "unknown"))
        raw_length = meta.get("length") or meta.get("text_length") or 0
        groups[primary_val].append(int(raw_length) if raw_length else 0)

    n_bins = config.length_bins
    edges: dict[str, list[float]] = {}
    for key, lengths in groups.items():
        if len(lengths) < n_bins:
            continue
        sorted_lens = sorted(lengths)
        bin_edges: list[float] = []
        for q in range(1, n_bins):
            idx = int(len(sorted_lens) * q / n_bins)
            bin_edges.append(float(sorted_lens[min(idx, len(sorted_lens) - 1)]))
        edges[key] = bin_edges

    return edges


def _collapse_small_strata(
    buckets: dict[str, list[BenchmarkItem]],
    min_size: int,
    primary_key: str,
) -> tuple[dict[str, list[BenchmarkItem]], list[str]]:
    """Merge strata with fewer than *min_size* items into an 'other' bucket."""
    collapsed_labels: list[str] = []
    overflow: dict[str, list[BenchmarkItem]] = defaultdict(list)
    final: dict[str, list[BenchmarkItem]] = {}

    for label, members in buckets.items():
        if len(members) < min_size:
            collapsed_labels.append(label)
            primary_val = label.split("::")[0] if "::" in label else label
            overflow_label = f"{primary_val}::other"
            overflow[overflow_label].extend(members)
        else:
            final[label] = members

    for label, members in overflow.items():
        relabelled = [replace(item, stratum=label) for item in members]
        if label in final:
            final[label].extend(relabelled)
        else:
            final[label] = relabelled

    return final, collapsed_labels


def build_freshwiki_config() -> StratificationConfig:
    return StratificationConfig(
        primary="quality_bucket",
        secondary="length",
        length_bins=2,
        min_stratum_size=5,
    )


def build_ultradomain_config() -> StratificationConfig:
    return StratificationConfig(
        primary="domain",
        secondary="length",
        length_bins=3,
        min_stratum_size=5,
    )


def build_hotpotqa_config() -> StratificationConfig:
    """HotpotQA: stratify by ``type`` (comparison|bridge) x context length bins."""
    return StratificationConfig(
        primary="type",
        secondary="length",
        length_bins=2,
        min_stratum_size=5,
    )


def build_alce_config() -> StratificationConfig:
    """ALCE: stratify by ``subset`` (asqa|qampari|eli5).  No length axis."""
    return StratificationConfig(
        primary="subset",
        secondary=None,
        length_bins=1,
        min_stratum_size=5,
    )
