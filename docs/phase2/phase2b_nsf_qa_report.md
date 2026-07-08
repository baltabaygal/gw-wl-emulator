# Phase 2B — NSF QA and Regression Report

Generated on: 2026-06-15 17:01:00 UTC

This report summarizes the QA and regression validation for the Conditional Neural Spline Flow (NSF) emulator pipeline. The test suite has been fully implemented and executed successfully inside the `test` conda environment.

---

## 1. Unit and Regression Test Suite

We have implemented four custom test files inside the `tests/` folder to cover every phase of the NSF pipeline:

### 1.1 Dataset and Standardization: [test_nsf_dataset.py](file:///Users/baltabay/Desktop/gw-wl-emulator/tests/test_nsf_dataset.py)
- **Verifies**:
  - `prepare_nsf_data` correctly computes dataset statistics and writes them to a JSON file.
  - `get_nsf_dataloaders` correctly initializes training, validation, and test PyTorch dataloaders.
  - Context and target standardization logic maps the training dataset to exactly mean 0 and standard deviation 1.
  - Prevents precision loss on the 4.5 million sample summation by computing statistics in double precision (`float64`).
- **Status**: **PASSING**

### 1.2 Model Mechanics: [test_nsf_model.py](file:///Users/baltabay/Desktop/gw-wl-emulator/tests/test_nsf_model.py)
- **Verifies**:
  - Flow feed-forward evaluation returns finite log probabilities with the correct shape for both PyTorch tensors and NumPy arrays.
  - Flow sampling maps context to target samples in the physical space with correct shapes.
  - Continuous density evaluated on a grid integrates approximately to 1.0 (using NumPy 2.0 compatible trapezoidal integration).
  - Saving and loading checkpoints preserves parameter weights, standardization statistics, and predictions exactly.
- **Status**: **PASSING**

### 1.3 End-to-End Training Smoke Test: [test_train_nsf_smoke.py](file:///Users/baltabay/Desktop/gw-wl-emulator/tests/test_train_nsf_smoke.py)
- **Verifies**:
  - The entire training pipeline in `train_nsf.py` runs end-to-end without crashing.
  - Executes a fast 2-epoch training run on the `datasets_tiny` dataset (3,938 samples) with early stopping and loss curve plotting.
  - Verifies that weights checkpoint is successfully written.
- **Status**: **PASSING**

### 1.4 Prior Bounds & Likelihood Evaluation: [test_nsf_likelihood.py](file:///Users/baltabay/Desktop/gw-wl-emulator/tests/test_nsf_likelihood.py)
- **Verifies**:
  - Prior boundary constraints behave correctly, returning `-inf` for parameters outside the valid range ($h \notin [0.59, 0.76]$, $\Omega_M \notin [0.20, 0.40]$, or $\sigma_8 \notin [0.65, 1.05]$).
  - Evaluates to a finite value inside boundaries.
- **Status**: **PASSING**

---

## 2. Test Execution Output

All 4 test files run and pass successfully in the conda environment:

```
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0
rootdir: /Users/baltabay/Desktop/gw-wl-emulator
collected 6 items

tests/test_nsf_dataset.py .                                              [ 16%]
tests/test_nsf_model.py ...                                              [ 66%]
tests/test_train_nsf_smoke.py .                                          [ 83%]
tests/test_nsf_likelihood.py .                                           [100%]

============================== 6 passed in 10.8s ===============================
```
