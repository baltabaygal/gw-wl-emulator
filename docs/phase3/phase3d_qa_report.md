# Phase 3D — QA and Regression Report

This report summarizes the verification of the scaling runner, metrics analysis, and output generation.

## Unit and Regression Tests

We created 4 new test files to cover the Phase 3D deliverables:

- **Command**: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/test_phase3d_scaling.py tests/test_phase3d_redshift_scaling.py tests/test_phase3d_metrics.py tests/test_phase3d_outputs.py`
- **Result**: **6 passed**
- **Runtime**: `0.41s`

## New Unit Tests

1. **`tests/test_phase3d_scaling.py`**:
   - Verifies the structure and validity of `data/results/phase3d_scaling_results.json`.
   - Asserts that all computed metrics like JSD and TV lie within physical ranges ($0 \le \text{Metric} \le 1.0$).
2. **`tests/test_phase3d_redshift_scaling.py`**:
   - Asserts that `data/results/phase3d_redshift_scaling_results.json` exists and is correctly split by catalog type (`mixed_uniform`, `mixed_low_z_dominated`, `mixed_high_z_dominated`) and catalog size.
3. **`tests/test_phase3d_metrics.py`**:
   - Verifies that the power-law fitting routine `fit_power_law` correctly fits convergent datasets and is robust against pathological flat/saturated data.
4. **`tests/test_phase3d_outputs.py`**:
   - Verifies that all report markdown files and generated plot files exist and are populated.
