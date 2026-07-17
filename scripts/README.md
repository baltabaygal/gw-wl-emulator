# scripts/ — Reproducible build & analysis workflows

Standalone drivers, grouped by purpose. **Run them from the repo root** — scripts write
outputs with root-relative paths (e.g. `plots/…`, `data/…`) regardless of where the file
lives. Most Python scripts need the `test` conda env; the subhalo-gate scripts noted in
`CLAUDE.md` can run with system Python.

```
scripts/
├── build/         # build & dev tooling
├── subhalo_gate/  # substructure gate: sampling, screen, factor calibration, validation
├── figures/       # all plotting / replot scripts
└── comparisons/   # cross-backend / cross-model comparisons
```

## build/

`build.sh` (CMake + pybind, via `make build`), `watch_build.sh` (auto-rebuild on `cpp/`
changes, `make watch`), `dev_shell.sh` / `dev_shell.ps1`, `run_make.ps1`.
The `Makefile` calls these as `./scripts/build/build.sh` and `./scripts/build/watch_build.sh`.

## subhalo_gate/ (see CLAUDE.md §4)

Demos: `subhalo_demo.py`, `subhalo_resolved_demo.py`.
Screen: `subhalo_screen.py` (slow, ~3 min), `subhalo_gate.py` (vectorized, <2 s),
`screen_geom_check.py`.
Factor calibration: `subhalo_factor_convergence.py`, `subhalo_factor_brute_multiseed.py`,
`subhalo_factor_redshift_check.py`, `subhalo_factor_proxy_check.py`,
`subhalo_mfloor_sensitivity.py`.
Validation: `validate_split_vs_brute.py`, `verify_subhalo_models*.py`, `ace_gap.py`,
`benchmark_subhalo_pdf.py`, `test_subhalo_sampling.py`.
Experiments: `experiment_local_subtraction.py`, `fit_bias_profile.py`.

## figures/

All `plot_*.py` and `replot_*.py` — gate figures, Vaskonen Fig.3 reproductions,
log-lin/pdf-ratio plots, realization diagrams, NFW/subhalo-factor diagnostics.

## comparisons/

`compare_all_models.py`, `compare_cpp_analytical.py`, `compare_mmin.py`,
`compare_with_90g.py`.
