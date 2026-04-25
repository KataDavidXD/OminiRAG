"""
真实数据集集成测试 -- 使用本地 HuggingFace 缓存中的 UltraDomain 和 FreshWiki。

测试策略:
  - 使用确定性 eval_fn（基于 item_id 的哈希），不依赖 LLM
  - 覆盖 proportional / neyman / mh 三种方法
  - 验证分层覆盖、预算约束、估计量收敛、状态序列化
  - 验证真实数据的 payload/target/metadata 字段完整性

运行方式:
    pytest tests/test_real_data_sampling.py -v
    pytest tests/test_real_data_sampling.py -v -k ultradomain
    pytest tests/test_real_data_sampling.py -v -k freshwiki

跳过条件: 本地无数据集时自动 skip。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from bsamp.sampling.engine import SamplingEngine, SamplingResult
from bsamp.sampling.types import (
    BenchmarkItem,
    Estimate,
    SamplingState,
)

# ---------------------------------------------------------------------------
# Cross-platform dataset path resolution (env-var override or HF default)
# ---------------------------------------------------------------------------

_HF_HUB = Path(os.environ.get("HF_HUB_DIR", str(Path.home() / ".cache" / "huggingface" / "hub")))
_UD_ROOT = str(os.environ.get("BENCHMARK_UD_ROOT",
    _HF_HUB / "datasets--TommyChien--UltraDomain" / "snapshots" / "aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"))
_FW_ROOT = str(os.environ.get("BENCHMARK_FW_ROOT",
    _HF_HUB / "datasets--EchoShao8899--FreshWiki" / "snapshots" / "03f2f8abbe54c78e834f70783de105129c07e18e"))
_HQ_ROOT = str(os.environ.get("BENCHMARK_HQ_ROOT",
    _HF_HUB / "datasets--hotpotqa--hotpot_qa" / "snapshots" / "1908d6afbbead072334abe2965f91bd2709910ab"))
_ALCE_ROOT = str(os.environ.get("BENCHMARK_ALCE_ROOT",
    _HF_HUB / "datasets--princeton-nlp--ALCE-data" / "snapshots" / "334fa2e7dd32040c3fef931a123c4be1a81e91a0"))

_HAS_UD = Path(_UD_ROOT).exists()
_HAS_FW = Path(_FW_ROOT).exists()
_HAS_HQ = Path(_HQ_ROOT).exists()
_HAS_ALCE = Path(_ALCE_ROOT).exists()

_UD_SUBSET = ["physics", "cs", "mathematics", "biology", "cooking"]


def _eval_fn(config: dict[str, Any], item_id: str) -> float:
    """确定性评分函数: 用 item_id 哈希生成 [0, 1] 伪奖励。"""
    h = hash(item_id) % 10000
    return h / 10000.0


# ---------------------------------------------------------------------------
# Session-scoped fixtures (一个进程只加载一次, 避免 MemoryError)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ud_adapter():
    if not _HAS_UD:
        pytest.skip("UltraDomain dataset not found locally")
    from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
    return UltraDomainAdapter(_UD_ROOT, target_domains=_UD_SUBSET)


@pytest.fixture(scope="module")
def fw_adapter():
    if not _HAS_FW:
        pytest.skip("FreshWiki dataset not found locally")
    from bsamp.sampling.adapters.freshwiki import FreshWikiAdapter
    return FreshWikiAdapter(_FW_ROOT)


# ===================================================================
# UltraDomain 真实数据测试
# ===================================================================

@pytest.mark.skipif(not _HAS_UD, reason="UltraDomain dataset not found locally")
class TestUltraDomainReal:

    # -- 数据加载 --

    def test_load_population(self, ud_adapter):
        """加载 5 domain 子集, 验证 item 数量和字段完整性。"""
        items = ud_adapter.load_items()
        assert len(items) > 500, f"5 domains should have ~700+ items, got {len(items)}"

        for item in items[:20]:
            assert item.benchmark == "ultradomain"
            assert item.payload["query"], "query should not be empty"
            assert item.payload["context"] is not None
            assert item.target["answers"] is not None or item.target["answer"] is not None
            assert item.metadata["domain"], "domain should not be empty"
            assert isinstance(item.metadata["length"], (int, float))

    def test_domain_coverage(self, ud_adapter):
        """验证 5 个 domain 都被加载。"""
        items = ud_adapter.load_items()
        domains = {item.metadata["domain"] for item in items}
        assert domains == set(_UD_SUBSET)

    # -- Proportional 采样 --

    def test_proportional_200(self, ud_adapter):
        """Proportional 抽样 200 条, 验证分层覆盖。"""
        engine = SamplingEngine(
            adapter=ud_adapter,
            method="proportional",
            budget=200,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert len(result.items) == 200
        assert result.estimate is not None
        assert 0.0 <= result.estimate.mean <= 1.0
        assert result.estimate.ci_lower < result.estimate.ci_upper

        domains = {item.metadata["domain"] for item in result.items}
        assert len(domains) >= 3, f"200 items should cover most domains, got {len(domains)}"

        for stratum, count in result.strata_summary.items():
            assert count > 0, f"Stratum {stratum} has 0 items"

    # -- Neyman 采样 --

    def test_neyman_300(self, ud_adapter):
        """Neyman 两阶段抽样 (pilot + optimal), 验证方差估计和 CI。"""
        engine = SamplingEngine(
            adapter=ud_adapter,
            method="neyman",
            budget=300,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert result.state.sampler_type == "neyman"
        assert len(result.state.realizations) == 2, "pilot + main"
        assert result.estimate is not None
        assert result.estimate.n_evaluated <= 300

        ci_width = result.estimate.ci_upper - result.estimate.ci_lower
        assert ci_width < 0.5, f"CI width {ci_width} too large for 300 samples"

    # -- MH 采样 --

    def test_mh_200_iterations(self, ud_adapter):
        """MH 采样 200 预算, 20 次迭代, 验证 energy trace。"""
        engine = SamplingEngine(
            adapter=ud_adapter,
            method="mh",
            budget=200,
            seed=42,
            eval_fn=_eval_fn,
            mh_iterations=20,
            mh_temperature=1.0,
            mh_anneal_rate=0.95,
        )
        result = engine.run()

        assert result.state.sampler_type == "mh"
        assert result.state.budget_used <= 200

        ss = result.state.sampler_state
        assert ss["step_count"] > 0
        assert len(ss["energy_trace"]) > 0
        assert "sorted_labels" in ss

    # -- 3-domain 子集引擎 --

    def test_subset_domains(self):
        """仅加载 3 个 domain 子集的引擎。"""
        if not _HAS_UD:
            pytest.skip("UltraDomain dataset not found locally")
        from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
        adapter = UltraDomainAdapter(_UD_ROOT, target_domains=["physics", "cs", "mathematics"])
        items = adapter.load_items()

        domains = {item.metadata["domain"] for item in items}
        assert domains == {"physics", "cs", "mathematics"}

        engine = SamplingEngine(
            adapter=adapter,
            method="proportional",
            budget=50,
            seed=42,
        )
        result = engine.run()
        assert len(result.items) == 50

    # -- 状态序列化 / 反序列化 --

    def test_state_roundtrip_real_data(self, ud_adapter):
        """完整状态 JSON 序列化往返。"""
        engine = SamplingEngine(
            adapter=ud_adapter,
            method="proportional",
            budget=100,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        json_str = result.state.to_json()
        restored = SamplingState.from_json(json_str)

        assert restored.benchmark == "ultradomain"
        assert restored.budget_total == 100
        assert restored.budget_used == result.state.budget_used
        assert len(restored.history) == len(result.state.history)
        for orig, rest in zip(result.state.history, restored.history):
            assert orig.item_id == rest.item_id
            assert abs(orig.reward - rest.reward) < 1e-12

    # -- SamplingResult.save() --

    def test_save_result(self, ud_adapter, tmp_path):
        """验证 SamplingResult.save() 输出合法 JSON。"""
        engine = SamplingEngine(
            adapter=ud_adapter,
            method="proportional",
            budget=50,
            seed=42,
        )
        result = engine.run()

        out_file = tmp_path / "result.json"
        result.save(str(out_file))

        loaded = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(loaded["items"]) == 50
        assert "strata_summary" in loaded
        assert "state" in loaded


# ===================================================================
# FreshWiki 真实数据测试
# ===================================================================

@pytest.mark.skipif(not _HAS_FW, reason="FreshWiki dataset not found locally")
class TestFreshWikiReal:

    def test_load_population(self, fw_adapter):
        """加载全部 FreshWiki 文档, 验证字段。"""
        items = fw_adapter.load_items()
        assert len(items) >= 80, f"FreshWiki should have ~100 docs, got {len(items)}"

        for item in items[:10]:
            assert item.benchmark == "freshwiki"
            assert item.payload["topic"], "topic should not be empty"
            assert item.payload["text"], "text should not be empty"
            assert item.metadata["predicted_class"] in {"Stub", "Start", "C", "B", "GA", "FA", "unknown"}
            assert item.metadata["quality_bucket"] in {"low", "mid", "high", "unknown"}
            assert isinstance(item.metadata["text_length"], int)
            assert isinstance(item.metadata["n_sections"], int)

    def test_proportional_30(self, fw_adapter):
        """Proportional 抽样 30 篇。"""
        engine = SamplingEngine(
            adapter=fw_adapter,
            method="proportional",
            budget=30,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert len(result.items) == 30
        assert result.estimate is not None

        quality_buckets = {item.metadata["quality_bucket"] for item in result.items}
        assert len(quality_buckets) >= 2, f"Expected multiple quality buckets, got {quality_buckets}"

    def test_neyman_50(self, fw_adapter):
        """Neyman 抽样 50 篇 (pilot + optimal)。"""
        engine = SamplingEngine(
            adapter=fw_adapter,
            method="neyman",
            budget=50,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert result.state.sampler_type == "neyman"
        assert result.estimate is not None
        assert result.state.budget_used <= 50

    def test_mh_40(self, fw_adapter):
        """MH 抽样 40 篇, 10 次迭代。"""
        engine = SamplingEngine(
            adapter=fw_adapter,
            method="mh",
            budget=40,
            seed=42,
            eval_fn=_eval_fn,
            mh_iterations=10,
        )
        result = engine.run()

        assert result.state.sampler_type == "mh"
        assert result.state.budget_used <= 40

    def test_full_population_sampling(self, fw_adapter):
        """预算超过 population 时, 自动 clamp。"""
        items = fw_adapter.load_items()
        n = len(items)

        engine = SamplingEngine(
            adapter=fw_adapter,
            method="proportional",
            budget=n + 100,
            seed=42,
        )
        result = engine.run()
        assert len(result.items) <= n


# ===================================================================
# 跨 Benchmark 对比测试
# ===================================================================

@pytest.mark.skipif(not (_HAS_UD and _HAS_FW), reason="Both datasets required")
class TestCrossBenchmark:

    def test_compare_ud_and_fw(self, ud_adapter, fw_adapter):
        """依次对 UltraDomain 和 FreshWiki 做抽样, 比较估计。"""
        ud_engine = SamplingEngine(adapter=ud_adapter, method="neyman", budget=100, seed=42, eval_fn=_eval_fn)
        fw_engine = SamplingEngine(adapter=fw_adapter, method="neyman", budget=30, seed=42, eval_fn=_eval_fn)

        ud_result = ud_engine.run()
        fw_result = fw_engine.run()

        assert ud_result.state.benchmark == "ultradomain"
        assert fw_result.state.benchmark == "freshwiki"
        assert ud_result.estimate is not None
        assert fw_result.estimate is not None

        ud_ids = {item.item_id for item in ud_result.items}
        fw_ids = {item.item_id for item in fw_result.items}
        assert ud_ids.isdisjoint(fw_ids)


# ===================================================================
# HotpotQA fixtures and real-data tests
# ===================================================================

@pytest.fixture(scope="module")
def hq_adapter():
    if not _HAS_HQ:
        pytest.skip("HotpotQA dataset not found locally")
    from bsamp.sampling.adapters.hotpotqa import HotpotQAAdapter
    return HotpotQAAdapter(_HQ_ROOT)


@pytest.mark.skipif(not _HAS_HQ, reason="HotpotQA dataset not found locally")
class TestHotpotQAReal:

    def test_load_population(self, hq_adapter):
        items = hq_adapter.load_items()
        assert len(items) > 1000, f"HotpotQA should have many items, got {len(items)}"

        for item in items[:20]:
            assert item.benchmark == "hotpotqa"
            assert item.payload["question"], "question should not be empty"
            assert item.target["answer"], "answer should not be empty"
            assert item.metadata["type"] in {"comparison", "bridge"}
            assert item.metadata["level"] in {"easy", "medium", "hard"}

    def test_type_coverage(self, hq_adapter):
        items = hq_adapter.load_items()
        types = {item.metadata["type"] for item in items}
        assert types == {"comparison", "bridge"}

    def test_proportional_100(self, hq_adapter):
        engine = SamplingEngine(
            adapter=hq_adapter,
            method="proportional",
            budget=100,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert len(result.items) == 100
        assert result.estimate is not None
        assert 0.0 <= result.estimate.mean <= 1.0
        assert result.estimate.ci_lower < result.estimate.ci_upper

    def test_neyman_200(self, hq_adapter):
        engine = SamplingEngine(
            adapter=hq_adapter,
            method="neyman",
            budget=200,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert result.state.sampler_type == "neyman"
        assert len(result.state.realizations) == 2
        assert result.estimate is not None
        ci_width = result.estimate.ci_upper - result.estimate.ci_lower
        assert ci_width < 0.5

    def test_state_roundtrip(self, hq_adapter):
        engine = SamplingEngine(adapter=hq_adapter, method="proportional", budget=50, seed=42, eval_fn=_eval_fn)
        result = engine.run()
        json_str = result.state.to_json()
        restored = SamplingState.from_json(json_str)
        assert restored.benchmark == "hotpotqa"
        assert restored.budget_total == 50
        assert len(restored.history) == len(result.state.history)


# ===================================================================
# ALCE fixtures and real-data tests
# ===================================================================

@pytest.fixture(scope="module")
def alce_adapter():
    if not _HAS_ALCE:
        pytest.skip("ALCE dataset not found locally")
    from bsamp.sampling.adapters.alce import ALCEAdapter
    return ALCEAdapter(_ALCE_ROOT, subsets=["asqa"])


@pytest.mark.skipif(not _HAS_ALCE, reason="ALCE dataset not found locally")
class TestALCEReal:

    def test_load_population(self, alce_adapter):
        items = alce_adapter.load_items()
        assert len(items) > 50, f"ALCE/asqa should have items, got {len(items)}"

        for item in items[:20]:
            assert item.benchmark == "alce"
            assert item.payload["question"], "question should not be empty"
            assert item.target["answer"] is not None
            assert item.metadata["subset"] == "asqa"

    def test_proportional_30(self, alce_adapter):
        engine = SamplingEngine(
            adapter=alce_adapter,
            method="proportional",
            budget=30,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert len(result.items) <= 30
        assert result.estimate is not None

    def test_neyman_50(self, alce_adapter):
        engine = SamplingEngine(
            adapter=alce_adapter,
            method="neyman",
            budget=50,
            seed=42,
            eval_fn=_eval_fn,
        )
        result = engine.run()

        assert result.state.sampler_type == "neyman"
        assert result.estimate is not None

    def test_state_roundtrip(self, alce_adapter):
        engine = SamplingEngine(adapter=alce_adapter, method="proportional", budget=20, seed=42, eval_fn=_eval_fn)
        result = engine.run()
        json_str = result.state.to_json()
        restored = SamplingState.from_json(json_str)
        assert restored.benchmark == "alce"
        assert restored.budget_total == 20
        assert len(restored.history) == len(result.state.history)
