# Phase 3 QA Report

Phase 3 regression coverage was added in:

- `tests/test_phase3_catalogs.py`
- `tests/test_phase3_posterior_grid.py`
- `tests/test_posterior_metrics.py`
- `tests/test_phase3_coverage.py`
- `tests/test_phase3_outputs.py`

The tests verify reproducible redshift generation, required metadata hashing, shared prior bounds, posterior normalization, finite summaries, ordered credible intervals, valid coverage fractions, stale-cache refusal, required output metadata, and NaN/Inf rejection.

Recommended QA command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/
```

