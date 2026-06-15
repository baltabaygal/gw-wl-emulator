# Phase 2A.6 – Likelihood Benchmark QA Report

This QA report documents the verification and testing of the simulator-vs-emulator inference benchmark suite. We evaluate reproducibility, prior bounds consistency, data consistency, and the status of regression tests.

---

## 1. QA Verification Items

### C1. Reproducibility
* **Method**: Verify that generating mock catalogs twice with the same cosmology parameters and seed yields identical, bitwise-reproducible magnification samples.
* **Status**: **PASS**. 
* **Validation**: The test `test_mock_catalog_reproducibility` in `tests/test_likelihood_benchmark.py` generates the catalog twice with `seed=42` and confirms that the resulting binned histogram counts match exactly.

### C2. Prior Consistency
* **Method**: Verify that the simulator likelihood $L_{\text{sim}}(\theta)$ and emulator likelihood $L_{\text{emu}}(\theta)$ enforce identical parameter bounds.
* **Bounds**:
  * $h \in [0.59, 0.76]$
  * $\Omega_M \in [0.20, 0.40]$
  * $\sigma_8 \in [0.65, 1.05]$
* **Status**: **PASS**.
* **Validation**: The test `test_prior_bounds_consistency` confirms that both likelihood functions return `-np.inf` or very large penalty values for parameters outside these bounds.

### C3. Data Consistency
* **Method**: Verify that the same mock catalog (binned on the same hybrid 100-bin log-linear grid) and the same objective function (Poisson likelihood) are used.
* **Status**: **PASS**.
* **Validation**: The binned mock catalog generated for $z = 2.40$ was shared between the simulator and emulator MCMC and parameter recovery runs. Both evaluate the same log-likelihood expression $-\sum_i n_i \log p_i(\theta)$.

### C4. Regression Tests
* **Method**: Verify that `tests/test_likelihood_benchmark.py` executes successfully, all test cases pass, and all metrics remain finite.
* **Status**: **PASS**.
* **Validation**: We ran the automated tests using `pytest` inside the `test` conda environment:
  * All 24 test cases (including physics regressions, reproducibility, and benchmark validations) passed.
  * Runtime metrics and likelihood evaluations are finite and successfully saved.

---

## 2. QA Final Verdict

**OVERALL STATUS: PASS**

The likelihood benchmark suite has been verified for correctness, prior consistency, reproducibility, and compliance with the Phase 2A.6 specifications.
