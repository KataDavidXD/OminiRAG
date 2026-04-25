# Benchmark Sampling

Research-grade sampling component for the WTB (Workflow Test Bench) RAG optimization framework. Estimates benchmark performance using minimal evaluation budget through stratified, Neyman-optimal, and Metropolis-Hastings adaptive sampling.

## Architecture

![System Architecture](docs/assets/image.png)

The system consists of three layers:

- **Benchmark Adapters** -- load and normalize FreshWiki / UltraDomain datasets into a unified `BenchmarkItem` format (payload / target / metadata split)
- **Sampling Component** -- stratification, sampler selection, sequential estimation with CI, budget control, and configuration comparison
- **WTB Infrastructure** -- checkpoint / rollback / fork for reproducibility, Ray-based parallel evaluation, and caching

## Quick Start

```bash
# Install
pip install -e .

# Run the demo (requires local HuggingFace dataset cache)
python main.py

# Run tests
pytest tests/ -v
```

### Minimal Usage

```python
from bsamp.sampling import SamplingEngine, UltraDomainAdapter

adapter = UltraDomainAdapter(
    root_dir="path/to/ultradomain",
    target_domains=["physics", "cs", "mathematics"],
)

engine = SamplingEngine(
    adapter=adapter,
    method="neyman",       # "proportional" | "neyman" | "mh"
    budget=200,
    seed=42,
    eval_fn=lambda cfg, iid: my_rag_eval(cfg, iid),  # (config, item_id) -> float
)

result = engine.run()

print(f"Sampled:  {len(result.items)} items")
print(f"Estimate: {result.estimate.mean:.4f}")
print(f"95% CI:   [{result.estimate.ci_lower:.4f}, {result.estimate.ci_upper:.4f}]")

# Persist full state (JSON-serializable, supports checkpoint/rollback)
result.save("sampling_result.json")
```

## Sampling Methods

| Method | Key Idea | When to Use |
|--------|----------|-------------|
| **proportional** | `n_h = B * N_h / N` | Default baseline, no variance estimates needed |
| **neyman** | `n_h = B * N_h * S_h / sum(N_k * S_k)` | When strata have different variances; runs a pilot phase first |
| **mh** | MCMC search over allocation space | Large number of strata (>15); adaptive variance minimization |

### Method Aliases

`"prop"`, `"optimal"`, `"metropolis"`, `"metropolis-hastings"` are accepted as aliases.

## Supported Benchmarks

### UltraDomain (TommyChien/UltraDomain)

- ~3833 QA items across 20 domains (agriculture, art, biology, cs, ...)
- Stratified by `domain` x `length` (3 bins) = up to 60 strata
- Per-item cost: low-medium (single QA)

```bash
hf download TommyChien/UltraDomain --repo-type=dataset
```

### FreshWiki (EchoShao8899/FreshWiki)

- ~100 Wikipedia topics with article-generation tasks
- Stratified by `quality_bucket` (low/mid/high) x `text_length` (2 bins) = ~5 strata
- Per-item cost: high (full article generation)

```bash
hf download EchoShao8899/FreshWiki --repo-type=dataset
```

## Project Structure

```
bsamp/
  loader/                      # Raw data loaders (FreshWikiAPI, UltraDomainAPI)
  sampling/
    engine.py                  # SamplingEngine facade + SamplingResult
    types.py                   # Core types (BenchmarkItem, SamplingState, Estimate, ...)
    stratification.py          # Stratum construction + collapse logic
    estimator.py               # Sequential mean/var/CI estimator + early stopping
    budget.py                  # Budget controller (standalone or state-bound)
    comparison.py              # Paired t-test + CI elimination
    diagnostics.py             # ESS, variance reduction, Gelman-Rubin R-hat
    adapters/                  # Benchmark-specific loaders
      base.py                  # Abstract BenchmarkAdapter
      freshwiki.py             # FreshWiki adapter
      ultradomain.py           # UltraDomain adapter
    samplers/                  # Sampling algorithms
      base.py                  # Abstract BaseSampler
      stratified.py            # Proportional + Neyman allocation
      mh.py                    # Metropolis-Hastings adaptive sampler
    integration/               # Infrastructure bridges
      wtb.py                   # WTB checkpoint/rollback/fork bridge
      ray_parallel.py          # Ray-based parallel evaluation
tests/
  test_integration_sampling.py # Synthetic end-to-end tests (12 tests)
  test_real_data_sampling.py   # Real dataset tests (14 tests)
  test_*.py                    # Unit tests for each module
docs/
  SAMPLING_ARCHITECTURE.md     # Detailed architecture documentation (Chinese)
  SAMPLING_PLAN.md             # Original design plan
```

## Key Design Principles

**Two layers of randomness:**

1. **Layer 1 (Allocation)** -- how many items per stratum? This is what MH optimizes.
2. **Layer 2 (Realization)** -- which specific items within each stratum? Conditional random draw, not optimized.

**Eval function contract:**

```python
eval_fn: Callable[[dict[str, Any], str], float]
#                   ^rag_config     ^item_id  ^reward
```

Any function matching this signature plugs in. Without `eval_fn`, the engine returns sampled items without computing estimates.

**State serialization:**

`SamplingState` is fully JSON-serializable via `to_json()` / `from_json()`. This enables:
- Local file checkpoint: `state.to_json() -> file`
- WTB checkpoint: auto-checkpointed via LangGraph node execution
- Exact replay: `ItemRealization.rng_state_before` allows reproducing the exact same item draw

## Tests

```bash
# All unit + integration tests (synthetic data)
pytest tests/ -v --ignore=tests/test_integration_wtb.py --ignore=tests/test_integration_ray_parallel.py

# Real dataset tests (requires local HF cache)
pytest tests/test_real_data_sampling.py -v

# WTB integration tests (requires wtb package)
pytest tests/test_integration_wtb.py -v

# Ray parallel tests (requires ray)
pytest tests/test_integration_ray_parallel.py -v
```

## Configuration Reference

```python
SamplingEngine(
    adapter=...,                    # BenchmarkAdapter instance (required)
    method="proportional",          # "proportional" | "neyman" | "mh"
    budget=100,                     # Max items to evaluate
    seed=42,                        # RNG seed for reproducibility
    stratification_config=None,     # Auto-detected from adapter.name if None
    eval_fn=None,                   # (config, item_id) -> float
    rag_config=None,                # Arbitrary dict for config identification
    mh_iterations=50,               # MH steps (only for method="mh")
    mh_temperature=1.0,             # Initial temperature (MH only)
    mh_anneal_rate=0.99,            # Annealing rate (MH only)
    stopping=StoppingConfig(        # Early stopping thresholds
        ci_threshold=0.05,
        relative_precision=0.10,
        confidence=0.95,
    ),
)
```

## License

Internal research project.
