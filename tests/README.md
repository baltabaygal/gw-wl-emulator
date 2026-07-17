# tests/ — Pytest suite

Regression and validation tests for the C++ bindings, dataset generation, the NSF
emulator, and the physics. Run with the `test` conda env (Python 3.12, compiled
`gwlensing` module):

```bash
make pytest          # python3 -m pytest tests
```

Tests that depend on a dataset check for it first and **skip** if it is absent, so a
partial checkout won't hard-fail. Fixture datasets live under `datasets/` (e.g.
`datasets/test_suite`, `datasets/tiny`, `datasets/backend_current_1k`); shared test data
is in `tests/data/`.

## Grouping

- **Cosmology / physics:** `test_cosmology_params.py`, `test_physics_regression.py`,
  `test_reproducibility.py`, `test_derivative_audit.py`, `test_representation.py`.
- **Datasets / generation:** `test_metadata.py`, `test_split_integrity.py`,
  `test_ood_generation.py`, `test_parallel_generation.py`, `test_nsf_dataset.py`.
- **NSF model / likelihood:** `test_nsf_model.py`, `test_nsf_likelihood*.py`,
  `test_nsf_tail_validation.py`, `test_train_nsf_smoke.py`, `test_emulator_baseline.py`.
- **Phase 2c / 3 / 3b / 3c / 3d:** `test_phase2c_*`, `test_phase3*` (coverage, catalogs,
  posteriors, scaling, likelihood definition) — mirrors the reports in `docs/`.
- **Subhalo:** `test_subhalo_demo.py`.
- **Infra:** `test_cache_utils.py`, `test_posterior_metrics.py`, `capture_reference_lnmu.py`.
