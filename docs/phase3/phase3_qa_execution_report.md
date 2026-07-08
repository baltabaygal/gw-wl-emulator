# Phase 3 QA Execution Report

Execution date: 2026-06-15

## Cache Safety

Stale simulator-grid caches from commit `f8c837cd4177133d8c5c390f90be2ffb929e2bbe` were quarantined under `data/cache/stale_phase2c_f8c837/`.

Current simulator checkout used by Phase 3 outputs:

`59c107efc320264b2a6d91bf31a89e62760b5ca7`

The smoke posterior grid metadata includes current simulator commit, NSF checkpoint hash, and preprocessing stats hash.

## Smoke Benchmark Integrity

The central mixed smoke grid completed without:

- NaN or Inf likelihood values;
- cache metadata mismatches;
- posterior normalization failures;
- posterior serialization failures.

## Tests

Phase 3-specific tests:

`9 passed`

Full suite:

`52 passed, 1 skipped`

The skipped test is the Phase 2C cached-grid verification. It skips because active root-level stale simulator-grid caches were quarantined before Phase 3 execution.

## Output Checks

Created or updated:

- `data/results/phase3_posterior_grids/`
- `data/results/phase3_posterior_metrics.json`
- `data/results/phase3_runtime_results.json`
- `data/results/phase3_coverage_results.json`
- `plots/figures/phase3_posterior_benchmark/`
- `docs/phase3_cache_refresh.md`
- `docs/phase3_posterior_smoke_report.md`
- `docs/phase3_posterior_benchmark_main.md`
- `docs/phase3_coverage_report.md`
- `docs/phase3_runtime_benchmark.md`
- `docs/phase3_figures_summary.md`
- `docs/phase3_posterior_benchmark_report.md`

