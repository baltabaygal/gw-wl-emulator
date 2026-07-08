# ml/ — Emulator training, evaluation & inference

The machine-learning emulator of the magnification PDF: model definitions, data loaders,
training drivers, evaluation, and posterior-recovery benchmarks.

> The **production** smooth-PDF model does not live here — it is in the
> `gw-wl-emulator-ar` worktree (`ml/autoresearch/smooth_model.py`, `load_smooth_fn()`).
> See the "NSF emulator" section of the root `CLAUDE.md`. This folder holds the core
> library and the earlier NSF/baseline experiments.

## Core library

| File | Role |
|------|------|
| `params.py` | **Single source of truth** for the parameter space (FIDUCIAL, PRIOR_6D, WIDE_6D, `CONTEXT_KEYS`) |
| `data.py` | Dataset loader; auto-detects HDF5 schema (7-d 1+6d vs legacy 4-d) |
| `nsf_model.py`, `nsf_dataset.py`, `nsf_likelihood.py` | Neural Spline Flow model, dataset wrapper, likelihood |
| `cache_utils.py`, `phase3_common.py` | Shared caching and phase-3 helpers |
| `baselines.py` | Baseline density estimators for comparison |

## Training & evaluation

`train_nsf.py`, `train_baseline.py` — training drivers (default `--dataset_dir datasets/…`).
`evaluate_nsf_pdf.py`, `evaluate_nsf_tail_robustness.py`, `diagnose_tail.py`,
`map_nsf_failures.py` — accuracy and tail diagnostics.
`run_capacity_study.py`, `run_scaling_study.py`, `run_pca_analysis.py`,
`run_interpolation_ceiling.py`, `run_derivative_audit.py` — analysis studies.

## Inference / posteriors

`generate_mock_catalogs.py`, `posterior_metrics.py`, `posterior_scaling_metrics.py`,
`plot_phase3_posteriors.py`, `benchmark_phase3_runtime.py`, `phase3b_diagnostics.py`.

Datasets referenced by default paths now live under `datasets/` (was `datasets_*` at repo
root). Reports for these experiments are in `docs/` (phase2*/phase3* files).
