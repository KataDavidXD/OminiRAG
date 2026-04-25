from __future__ import annotations

from bsamp.sampling.types import BenchmarkItem
from bsamp.sampling.stratification import (
    StratificationConfig,
    stratify,
    build_freshwiki_config,
    build_ultradomain_config,
)


def _make_items(n: int, domain: str = "physics", length_range: tuple[int, int] = (100, 10000)) -> list[BenchmarkItem]:
    import random
    rng = random.Random(42)
    items = []
    for i in range(n):
        items.append(BenchmarkItem(
            item_id=f"test::{domain}::{i}",
            benchmark="test",
            stratum="",
            payload={"query": f"q{i}"},
            target={"answer": f"a{i}"},
            metadata={
                "domain": domain,
                "length": rng.randint(*length_range),
                "quality_bucket": rng.choice(["low", "mid", "high"]),
                "predicted_class": rng.choice(["Stub", "Start", "C", "B", "GA", "FA"]),
                "text_length": rng.randint(*length_range),
            },
        ))
    return items


class TestStratify:
    def test_all_items_assigned(self):
        items = _make_items(50, "physics") + _make_items(50, "cs")
        cfg = StratificationConfig(primary="domain", secondary="length", length_bins=3, min_stratum_size=5)
        strata, stats = stratify(items, cfg)

        all_ids = set()
        for members in strata.values():
            for item in members:
                assert item.stratum != "", "stratum must be set"
                all_ids.add(item.item_id)

        assert len(all_ids) == 100, "all items must appear exactly once"

    def test_no_overlap(self):
        items = _make_items(80, "math") + _make_items(80, "bio")
        cfg = StratificationConfig(primary="domain", secondary="length", length_bins=2, min_stratum_size=5)
        strata, stats = stratify(items, cfg)

        seen = set()
        for members in strata.values():
            for item in members:
                assert item.item_id not in seen, f"Duplicate: {item.item_id}"
                seen.add(item.item_id)

    def test_stats_population_matches(self):
        items = _make_items(60, "a") + _make_items(40, "b")
        cfg = StratificationConfig(primary="domain", secondary=None, min_stratum_size=1)
        strata, stats = stratify(items, cfg)

        for label, members in strata.items():
            assert stats[label].population_size == len(members)

    def test_small_strata_collapsed(self):
        items = _make_items(50, "big") + _make_items(3, "tiny")
        cfg = StratificationConfig(primary="domain", secondary=None, min_stratum_size=5)
        strata, stats = stratify(items, cfg)

        for label, members in strata.items():
            assert len(members) >= 5 or "other" in label, f"Stratum {label} too small: {len(members)}"

    def test_freshwiki_config(self):
        items = _make_items(100, "wiki")
        cfg = build_freshwiki_config()
        strata, stats = stratify(items, cfg)
        total = sum(s.population_size for s in stats.values())
        assert total == 100

    def test_ultradomain_config(self):
        items = _make_items(200, "physics") + _make_items(200, "cs")
        cfg = build_ultradomain_config()
        strata, stats = stratify(items, cfg)
        total = sum(s.population_size for s in stats.values())
        assert total == 400
