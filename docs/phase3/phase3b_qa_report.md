# Phase 3B QA Report

## Added Regression Coverage

Phase 3B adds focused tests for the diagnostic harness:

| Test file | Coverage |
|---|---|
| `tests/test_phase3b_likelihood_audit.py` | Likelihood audit detects tail/clamping mismatches and grid validation rejects non-finite values. |
| `tests/test_phase3b_single_z.py` | Single-z isolation pipeline returns the expected result structure without running heavy simulator work in unit tests. |
| `tests/test_phase3b_grid_refinement.py` | Grid-refinement output structure and finite metrics are validated. |
| `tests/test_phase3b_redshift_decomposition.py` | Redshift-slice decomposition returns low/mid/high slice records. |
| `tests/test_phase3b_local_density.py` | Local density residual writer creates JSON, report, and plot outputs. |

## Targeted Phase 3B Test Run

Command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test python -m pytest tests/test_phase3b_likelihood_audit.py tests/test_phase3b_single_z.py tests/test_phase3b_grid_refinement.py tests/test_phase3b_redshift_decomposition.py tests/test_phase3b_local_density.py -q
```

Result: `6 passed`.

After removing a harmless empty-legend warning from the local-density writer, the affected test was rerun:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/test_phase3b_local_density.py -q
```

Result: `1 passed`.

## Full Suite

Command requested in the brief:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/
```

Result:

```text
58 passed, 1 skipped, 1 warning in 85.08s
```

The warning was the local-density empty-legend warning noted above and was fixed after the full-suite run.

## Cache Safety

The Phase 3B diagnostics preserve strict cache metadata:

- catalog hashes are retained in grid metadata,
- simulator commit is included in cache keys,
- NSF checkpoint hash and preprocessing-stats hash are verified,
- NaN/Inf grids are rejected by normalization and tested directly,
- no prior bounds or simulator physics were changed.
