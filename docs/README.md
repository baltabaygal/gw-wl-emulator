# docs/ — Reports, validations & workflow notes

The written record behind the code: reports produced across the project's phases plus
workflow guides. Many `phase*` and `backend_current` reports are **written by scripts** in
`ml/` and `python/` and **asserted by tests** in `tests/` — the file paths here match the
paths hardcoded there, so keep the two in sync if you move a report.

```
docs/
├── reference/         # workflow & how-to guides + dataset/perf/QA reports
├── backend_current/   # production 1k-backend dataset, training, posterior, recovery
├── phase2/            # phase 2 / 2b / 2c — representation & conditional NSF
├── phase3/            # phase 3 / 3b / 3c / 3d — density accuracy, coverage, posteriors
└── subhalo/           # substructure model derivations & pipeline comparison
```

## reference/

Guides: `dev_workflow.md`, `docker.md`, `walkthrough.md`, `troubleshooting.md`.
Reports: `dataset_report.md`, `dataset_validation_report.md`, `performance_report.md`,
`qa_report.md`, `invalid_sample_analysis.md`, `optionA_check.md`, `optionB_check.md`.

## backend_current/

`backend_current_dataset_metadata.json`, `backend_current_dataset_validation.md`,
`backend_current_training_report.md`, `backend_current_posterior_validation.md`,
`backend_current_recovery_benchmark.md`.

## phase2/ (representation & conditional NSF)

`phase2_*` (target design, baseline, representation, likelihood replacement, derivative
audit, interpolation ceiling), `phase2b_*` (conditional NSF training/validation/QA/
recovery), `phase2c_*` (hardening, tail validation, runtime optimization, failure map,
cache reproducibility).

## phase3/ (density accuracy, coverage & posteriors)

`phase3_*` (coverage, density accuracy, figures, maxN, N-scaling, posterior benchmarks,
QA), `phase3b_*` (grid refinement, local density, redshift decomposition),
`phase3c_*` (likelihood compatibility & definition), `phase3d_*` (posterior/redshift
scaling, metrics).

## subhalo/

`subhalo_combining.md` (design doc), `subhalo_factor_closure.md`,
`subhalo_unresolved_handoff.md`, `variance_derivation.md`, `pyhalo_pipeline_comparison.md`.
