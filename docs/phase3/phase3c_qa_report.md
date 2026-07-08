# Phase 3C — QA and Regression Report

This report summarizes the verification of the simulator-compatible likelihood evaluation mode and grid re-run pipelines.

## Regression Suite Execution

We ran the entire codebase test suite including the 4 new regression tests designed specifically for Phase 3C:

- **Command**: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/`
- **Results**: **64 passed, 1 skipped**
- **Runtime**: `78.64s`

## New Phase 3C Unit Tests

The following tests were introduced and successfully executed:

1. **`tests/test_phase3c_likelihood_definition.py`**:
   - Asserts the existence and structure of the side-by-side likelihood definition audit JSON.
2. **`tests/test_phase3c_compatibility_mode.py`**:
   - Verifies that both `continuous` and `simulator_compatible` modes can be invoked on a `ConditionalNSF` model.
   - Verifies that passing invalid modes or invalid inputs (e.g. `NaN`) properly raises a `ValueError` rather than silently continuing.
3. **`tests/test_phase3c_smoke_retest.py`**:
   - Verifies that the smoke re-run results exist and are populated with valid posterior metrics ($0 \le \text{JSD} \le 1$).
   - Verifies that grid caches generated for the NSF grid store the correct `likelihood_mode` attribute within their metadata payload.
4. **`tests/test_phase3c_single_z_retest.py`**:
   - Verifies that the single-redshift retest outputs exist and are populated for all target redshifts ($z = 0.5, 1.5, 2.5$).
