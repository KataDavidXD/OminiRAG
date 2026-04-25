"""
Benchmark Sampling -- 主入口示例。

演示如何用 SamplingEngine 对 UltraDomain 和 FreshWiki 做分层抽样并输出结果。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_HF_HUB = Path(os.environ.get("HF_HUB_DIR", str(Path.home() / ".cache" / "huggingface" / "hub")))
_UD_ROOT = str(os.environ.get("BENCHMARK_UD_ROOT",
    _HF_HUB / "datasets--TommyChien--UltraDomain" / "snapshots" / "aa8a51d523f8fc3c5a0ab90dd16b7f6b9dbb5d0d"))
_FW_ROOT = str(os.environ.get("BENCHMARK_FW_ROOT",
    _HF_HUB / "datasets--EchoShao8899--FreshWiki" / "snapshots" / "03f2f8abbe54c78e834f70783de105129c07e18e"))


def dummy_eval(config: dict[str, Any], item_id: str) -> float:
    """确定性伪评分: hash(item_id) 映射到 [0, 1]。"""
    return (hash(item_id) % 10000) / 10000.0


def sample_ultradomain() -> None:
    """对 UltraDomain 做 Neyman 抽样并打印摘要。"""
    from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
    from bsamp.sampling.engine import SamplingEngine

    adapter = UltraDomainAdapter(_UD_ROOT, target_domains=["physics", "cs", "mathematics"])
    engine = SamplingEngine(
        adapter=adapter,
        method="neyman",
        budget=60,
        seed=42,
        eval_fn=dummy_eval,
    )
    result = engine.run()

    print("=" * 60)
    print("  UltraDomain Neyman Sampling")
    print("=" * 60)
    print(f"  Method:       {result.state.sampler_type}")
    print(f"  Budget:       {result.state.budget_total}")
    print(f"  Sampled:      {len(result.items)}")
    print(f"  Realizations: {len(result.state.realizations)}")
    print()
    print("  Strata population:")
    for stratum, count in sorted(result.strata_summary.items()):
        print(f"    {stratum:30s}  N={count}")
    print()
    print("  Allocation vector:", result.realization.allocation)
    print()
    if result.estimate:
        est = result.estimate
        print(f"  Estimate:     {est.mean:.4f}")
        print(f"  Std Error:    {est.std_error:.4f}")
        print(f"  95% CI:       [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        print(f"  CI Width:     {est.ci_upper - est.ci_lower:.4f}")
    print()
    print("  Sample items (first 5):")
    for item in result.items[:5]:
        print(f"    {item.item_id:25s}  stratum={item.stratum:25s}  domain={item.metadata['domain']}")
    print()
    _interpret_neyman(result)


def _interpret_neyman(result) -> None:
    """打印 Neyman 结果的解读说明。"""
    est = result.estimate
    n_strata = len(result.strata_summary)
    n_items = len(result.items)
    n_realizations = len(result.state.realizations)

    print("  --- How to read this ---")
    print(f"  This Neyman run used a 2-phase strategy ({n_realizations} realizations):")
    print(f"    Phase 1 (pilot):  proportional draw to estimate per-stratum variance S_h")
    print(f"    Phase 2 (optimal): allocate remaining budget via n_h ~ N_h * S_h")
    print(f"  {n_items} items were drawn across {n_strata} strata.")
    print()
    print("  Allocation vector = [n_1, n_2, ..., n_H]  -- how many items per stratum")
    print("    Larger n_h means that stratum had higher N_h * S_h (bigger or noisier).")
    print("    Strata with n_h=1 are small or had low variance in the pilot.")
    print()
    if est:
        ci_w = est.ci_upper - est.ci_lower
        print(f"  Estimate = {est.mean:.4f}:  weighted average reward across all strata.")
        print(f"  95% CI   = [{est.ci_lower:.4f}, {est.ci_upper:.4f}]:  true population mean")
        print(f"             lies in this interval with 95% confidence.")
        print(f"  CI Width = {ci_w:.4f}:  narrower is better. Target < 0.05 for precision.")
        if ci_w > 0.15:
            print("    -> CI is still wide; increase budget or add more strata to tighten.")
        elif ci_w > 0.05:
            print("    -> CI is moderate; reasonable for exploratory comparisons.")
        else:
            print("    -> CI is tight; reliable for definitive conclusions.")
        print(f"  Std Error = {est.std_error:.4f}:  estimation uncertainty (SE = CI_width / 3.92).")
    print()


def sample_freshwiki() -> None:
    """对 FreshWiki 做 Proportional 抽样并打印摘要。"""
    from bsamp.sampling.adapters.freshwiki import FreshWikiAdapter
    from bsamp.sampling.engine import SamplingEngine

    adapter = FreshWikiAdapter(_FW_ROOT)
    engine = SamplingEngine(
        adapter=adapter,
        method="proportional",
        budget=20,
        seed=42,
        eval_fn=dummy_eval,
    )
    result = engine.run()

    print("=" * 60)
    print("  FreshWiki Proportional Sampling")
    print("=" * 60)
    print(f"  Method:       {result.state.sampler_type}")
    print(f"  Budget:       {result.state.budget_total}")
    print(f"  Sampled:      {len(result.items)}")
    print()
    print("  Strata population:")
    for stratum, count in sorted(result.strata_summary.items()):
        print(f"    {stratum:30s}  N={count}")
    print()
    print("  Allocation vector:", result.realization.allocation)
    print()
    if result.estimate:
        est = result.estimate
        print(f"  Estimate:     {est.mean:.4f}")
        print(f"  Std Error:    {est.std_error:.4f}")
        print(f"  95% CI:       [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        print(f"  CI Width:     {est.ci_upper - est.ci_lower:.4f}")
    print()
    print("  Sample items (first 5):")
    for item in result.items[:5]:
        qb = item.metadata.get("quality_bucket", "?")
        print(f"    {item.item_id:40s}  stratum={item.stratum:20s}  quality={qb}")
    print()
    _interpret_proportional(result)


def _interpret_proportional(result) -> None:
    """打印 Proportional 结果的解读说明。"""
    est = result.estimate
    total_pop = sum(result.strata_summary.values())
    n_items = len(result.items)

    print("  --- How to read this ---")
    print(f"  Proportional sampling: each stratum gets n_h ~ (N_h / N) * budget.")
    print(f"  Total population N = {total_pop}, sampled {n_items} ({n_items/total_pop:.0%} of population).")
    print()
    print("  Allocation vector = [n_1, n_2, ...] mirrors population proportions.")
    print("    Strata with more items get more samples -- no variance optimisation.")
    print("    Simple & unbiased, but not as efficient as Neyman if variances differ.")
    print()
    if est:
        ci_w = est.ci_upper - est.ci_lower
        print(f"  Estimate = {est.mean:.4f}:  weighted mean reward across strata.")
        print(f"  95% CI   = [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        print(f"  CI Width = {ci_w:.4f}")
        if n_items / total_pop > 0.15:
            print(f"    -> Sampling fraction is high ({n_items/total_pop:.0%}).")
            print(f"       Finite-population correction reduces SE significantly.")
        if ci_w > 0.15:
            print("    -> CI is wide; consider increasing budget or switching to Neyman.")
        elif ci_w > 0.05:
            print("    -> CI is moderate; useful for initial exploration.")
        else:
            print("    -> CI is tight; results are precise.")
    print()


def sample_mh_ultradomain() -> None:
    """对 UltraDomain 做 Metropolis-Hastings 自适应抽样，输出诊断信息。"""
    from bsamp.sampling.adapters.ultradomain import UltraDomainAdapter
    from bsamp.sampling.engine import SamplingEngine

    adapter = UltraDomainAdapter(_UD_ROOT, target_domains=["physics", "cs", "mathematics"])
    engine = SamplingEngine(
        adapter=adapter,
        method="mh",
        budget=80,
        seed=42,
        eval_fn=dummy_eval,
        mh_iterations=30,
        mh_temperature=2.0,
        mh_anneal_rate=0.95,
    )
    result = engine.run()

    print("=" * 60)
    print("  UltraDomain  Metropolis-Hastings Sampling")
    print("=" * 60)
    print(f"  Method:         {result.state.sampler_type}")
    print(f"  Budget:         {result.state.budget_total}")
    print(f"  Sampled:        {len(result.items)}")
    print(f"  MH iterations:  30")
    print(f"  Temperature:    2.0  (anneal 0.95)")
    print(f"  Realizations:   {len(result.state.realizations)}")
    print()

    print("  Strata population:")
    for stratum, count in sorted(result.strata_summary.items()):
        print(f"    {stratum:30s}  N={count}")
    print()

    alloc = result.realization.allocation
    print("  Final allocation vector:")
    if isinstance(alloc, dict):
        for stratum, n_h in sorted(alloc.items()):
            print(f"    {stratum:30s}  n_h={n_h}")
    else:
        labels = (result.state.sampler_state or {}).get("sorted_labels", [])
        for lbl, n_h in zip(labels, alloc):
            print(f"    {lbl:30s}  n_h={n_h}")
    print()

    if result.estimate:
        est = result.estimate
        print(f"  Estimate:       {est.mean:.4f}")
        print(f"  Std Error:      {est.std_error:.4f}")
        print(f"  95% CI:         [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        print(f"  CI Width:       {est.ci_upper - est.ci_lower:.4f}")
        print()

    sampler_st = result.state.sampler_state
    if sampler_st:
        n_accepted = sampler_st.get("n_accepted", 0)
        step_count = sampler_st.get("step_count", 0)
        temperature = sampler_st.get("temperature", "?")
        print("  MH diagnostics:")
        print(f"    Accepted / Steps:    {n_accepted} / {step_count}")
        if step_count > 0:
            print(f"    Acceptance rate:     {n_accepted / step_count:.2%}")
        print(f"    Final temperature:   {temperature:.6f}")
        energy_trace = sampler_st.get("energy_trace", [])
        if energy_trace:
            print(f"    Energy (initial):    {energy_trace[0]:.6f}")
            print(f"    Energy (final):      {energy_trace[-1]:.6f}")
            print(f"    Energy (min):        {min(energy_trace):.6f}")
        cur_alloc = sampler_st.get("current_allocation")
        labels = sampler_st.get("sorted_labels", [])
        if cur_alloc and labels:
            print("    Best allocation (MH state):")
            if isinstance(cur_alloc, dict):
                for lbl in labels:
                    print(f"      {lbl:30s}  n_h={cur_alloc.get(lbl, 0)}")
            else:
                for lbl, val in zip(labels, cur_alloc):
                    print(f"      {lbl:30s}  n_h={val}")
    print()

    print("  Sample items (first 5):")
    for item in result.items[:5]:
        print(f"    {item.item_id:25s}  stratum={item.stratum:25s}  domain={item.metadata['domain']}")
    print()
    _interpret_mh(result)


def _interpret_mh(result) -> None:
    """打印 MH 结果的解读说明。"""
    est = result.estimate
    sampler_st = result.state.sampler_state or {}
    n_accepted = sampler_st.get("n_accepted", 0)
    step_count = sampler_st.get("step_count", 0)
    energy_trace = sampler_st.get("energy_trace", [])

    print("  --- How to read this ---")
    print("  Metropolis-Hastings is a 2-phase adaptive method:")
    print("    Phase 1 (pilot):       proportional draw -> estimate per-stratum variance S_h")
    print("    Phase 2 (MH search):   iteratively propose new allocation vectors a,")
    print("                           accept/reject by Metropolis criterion, anneal temperature.")
    print("    Final draw:            sample items using the converged allocation.")
    print()
    print("  Key diagnostics:")
    if step_count > 0:
        rate = n_accepted / step_count
        print(f"    Acceptance rate = {rate:.0%}  ({n_accepted}/{step_count})")
        if rate > 0.8:
            print("      -> Very high: temperature may be too warm or budget is small.")
            print("         Try lower initial temperature or more iterations.")
        elif rate > 0.5:
            print("      -> High but acceptable for exploration phase.")
        elif rate > 0.2:
            print("      -> Good range (20-50%): MH is mixing well.")
        else:
            print("      -> Low: chain may be stuck. Try higher temperature or slower annealing.")
    if energy_trace:
        e0, ef, emin = energy_trace[0], energy_trace[-1], min(energy_trace)
        reduction = (1 - ef / e0) * 100 if e0 > 0 else 0
        print(f"    Energy:  {e0:.6f} -> {ef:.6f}  (reduced {reduction:.1f}%,  min={emin:.6f})")
        print("      Energy = stratified variance Var(F_hat). Lower = better allocation.")
        if reduction > 30:
            print("      -> Substantial improvement over proportional starting point.")
        elif reduction > 10:
            print("      -> Moderate improvement; proportional was already decent.")
        else:
            print("      -> Minimal change; proportional may already be near-optimal here.")
    print()
    if est:
        ci_w = est.ci_upper - est.ci_lower
        print(f"  Estimate = {est.mean:.4f}:  weighted mean reward (same formula as Neyman).")
        print(f"  95% CI   = [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        print(f"  CI Width = {ci_w:.4f}")
        print("    Compare this CI width against Neyman/Proportional with same budget.")
        print("    MH should produce equal or narrower CI if it found a better allocation.")
    print()


def main() -> None:
    sample_ultradomain()
    sample_freshwiki()
    sample_mh_ultradomain()


if __name__ == "__main__":
    main()
