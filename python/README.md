# python/ — Phase-1 bindings, dataset generation & validation

Utilities built directly on the `gwlensing` C++ module: generating training datasets from
the sampler, validating them, and the early plotting/diagnostics work. Run everything with
the `test` conda env (Python 3.12) — see root `CLAUDE.md`.

## Key files

| File | Role |
|------|------|
| `generate_dataset.py` | Sample the parameter box (LHS) and write HDF5 datasets (schema 2.0). Default `--output_dir datasets` |
| `validate_dataset.py` | Sanity/integrity checks on a generated dataset |
| `generate_dataset_report.py`, `report_generator.py` | Human-readable dataset reports |
| `validation_plots.py`, `plot_parameter_coverage.py`, `plot_lnmu.py` | Coverage and distribution plots |
| `testing_api.py` | Smoke test of the C++ API (`make test`) |
| `config.py` | Shared configuration |
| `benchmark.py` | Sampler timing benchmarks |
| `diagnostics.py`, `smoothness_diagnostics.py`, `debug_invalid_samples.py` | Data QA |
| `ood_utilities.py` | Out-of-distribution region helpers |
| `run_phase1_validation.py` | End-to-end phase-1 validation driver |

## Common commands

```bash
make generate-small-dataset          # tiny dataset for smoke tests
python python/validate_dataset.py
python python/generate_dataset.py --num_points 100 --nsamples 5000 --output_dir datasets/mydata
```

Generated datasets land in `datasets/` (gitignored). See `datasets/README.md`.
