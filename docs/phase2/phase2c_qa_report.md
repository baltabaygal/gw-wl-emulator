# Phase 2C — QA and Regression Test Report

Generated on: 2026-06-15 19:06:00 UTC
Test Command: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=build:. conda run -n test pytest tests/`

---

## 1. Test Session Summary

All tests in the repository test suite have passed successfully:

- **Total Tests**: 44
- **Passed**: 44
- **Failed**: 0
- **Duration**: 90.63 seconds
- **Platform**: darwin (Python 3.12.13, pytest-9.0.3, pluggy-1.6.0)

---

## 2. Scientific & Software Verification Details

The test suite includes dedicated regression and verification checks for the Phase 2C hardening:

### 2.1 Likelihood Batching & Equivalence
- **File**: `tests/test_nsf_likelihood_batching.py`
- **Verification**: Checks that the flat-batched implementation `catalog_log_likelihood` returns numerically identical log-likelihood values to the unbatched loop across a batch of cosmologies.
- **Tolerance**: Verified within a strict tolerance of $10^{-5}$.
- **Status**: **PASSED**

### 2.2 Tail Metric Validation
- **File**: `tests/test_nsf_tail_validation.py`
- **Verification**: Verifies that the tail probability mass function and binned quantile interpolation return finite, valid values on mock datasets.
- **Status**: **PASSED**

### 2.3 Cache Mismatch Safeguard
- **File**: `tests/test_cache_utils.py`
- **Verification**: Tests that the `CacheManager` successfully writes and reads cached npz files, and raises a loud `ValueError` if the requested parameters (e.g., seed, git commit, cosmological parameters) differ from those saved inside the cache metadata.
- **Status**: **PASSED**

### 2.4 Output Integrity & Non-finite Value Checks
- **File**: `tests/test_phase2c_outputs.py`
- **Verification**:
  1. Confirms the existence and non-zero size of all required validation reports and JSON files.
  2. Recursively parses all JSON files to verify that **no NaN/Inf values** are present or silently accepted.
  3. Verifies that loading cached grids via the cache manager matches the grid definition in the metadata, and that cached grids match the saved npz files exactly.
- **Status**: **PASSED**

---

## 3. Detailed Test File Breakdown

| Test File | Passed Tests | Description |
|---|---|---|
| `tests/test_backend_consistent_training.py` | 2 | Training consistency verification |
| `tests/test_backend_current_likelihood_benchmark.py` | 3 | Likelihood benchmarks against current backend |
| `tests/test_cache_utils.py` | 2 | Cache save/load and loud error checks |
| `tests/test_derivative_audit.py` | 1 | Derivative audit checks |
| `tests/test_emulator_baseline.py` | 4 | Baseline MLP checks |
| `tests/test_interpolation_ceiling.py` | 1 | Interpolation limits checks |
| `tests/test_likelihood_benchmark.py` | 3 | General likelihood benchmark checks |
| `tests/test_metadata.py` | 1 | Metadata definition checks |
| `tests/test_nsf_dataset.py` | 1 | NSF dataset class checks |
| `tests/test_nsf_likelihood.py` | 1 | NSF log_prob evaluation checks |
| `tests/test_nsf_likelihood_batching.py` | 2 | Pairwise batching and flat-batching checks |
| `tests/test_nsf_model.py` | 3 | NSF model structure and check-pointing |
| `tests/test_nsf_tail_validation.py` | 2 | Quantile and tail metrics checks |
| `tests/test_ood_generation.py` | 1 | Out-of-distribution generation checks |
| `tests/test_parallel_generation.py` | 1 | Multi-process simulator checks |
| `tests/test_phase2c_outputs.py` | 3 | File existence, NaN/Inf checks, grid matching |
| `tests/test_physics_regression.py` | 8 | C++ simulator physics regression checks |
| `tests/test_representation.py` | 2 | Representation capacity checks |
| `tests/test_reproducibility.py` | 1 | Random seeding reproducibility checks |
| `tests/test_split_integrity.py` | 1 | Training/Validation/Test split checks |
| `tests/test_train_nsf_smoke.py` | 1 | NSF training smoke checks |
