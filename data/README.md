# data/ — Derived numeric data & model artifacts

Numeric outputs from the C++ sampler and the ML pipeline. Raw sampler output is
**immutable** — write new files, never edit in place.

## Subfolders

| Folder | Contents |
|--------|----------|
| `raw/` | Immutable C++ sampler output (`lnmu_*.txt`, `wrapper_test.txt`, raw PDFs) |
| `models/` | Saved model checkpoints and training histories |
| `cache/` | Cached intermediates (gitignored; safe to delete/regenerate) |
| `results/` | Benchmark and analysis result files |
| `mock_catalogs/` | Generated mock ET/LISA event catalogs |
| `processed/` | Training/test sets (currently empty; datasets now live in top-level `datasets/`) |

## Loose result files (top level of `data/`)

Numerous `.npz` / `.csv` / `.json` files from subhalo-factor and variance-sweep runs live
directly in `data/` (e.g. `variance_sweep_data_z*.npz`, `subhalo_factor_*_check.npz`,
`vaskonen_fig3_linlin_subhalos*.json`). These are referenced by scripts in `scripts/` and
`paper_prod/` by name, so they are kept here rather than moved. If you tidy them into a
subfolder later, update the referencing scripts to match.

Note: `data/cache/` is gitignored; most other files here are tracked.
