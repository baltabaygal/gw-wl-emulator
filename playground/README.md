# playground/ — Subhalo-factor experiments (scratch)

Exploratory experiments behind the subhalo "factor" calibration. This is scratch/research
space — the canonical, published versions live in `scripts/subhalo_gate/` and
`paper_prod/`. Each subfolder pairs its script(s) with the `.json`/`.png` outputs they
produced.

| Folder | What it explores |
|--------|------------------|
| `analytic/` | Analytic deficit / cumulant-components check |
| `dgate/` | d-gate area/threshold scans (incl. r300 variants) |
| `paired/` | Paired-estimator area scans (dense / smooth / more-realizations) |
| `population/` | Population-aggregate runs per source redshift (`zs*`), `step3_data/`, `z=*.csv` |
| `stratified/` | Stratified Monte-Carlo estimator |
| `proof/` | `subhalo_factor_proof.py` proof-of-concept + `test_Lambda.ipynb` |

## Notes

- Scripts assume they are run from the **repo root** (outputs use root-relative paths).
- `proof/subhalo_factor_proof.py` does `import subhalo_factor_proxy_check`, which lives in
  `scripts/subhalo_gate/`. This bare import already depended on that module being on
  `sys.path`; run it with `PYTHONPATH=scripts/subhalo_gate` (or from that dir) if needed.
- Nothing in the tracked production code imports from here — safe to prune when an
  experiment is no longer needed.
