# Walkthrough - Phase 2A.7 Retraining and Posterior Validation Completed

This walkthrough summarizes the results and accomplishments for Phase 2A.7 – Backend-Consistent Retraining and Posterior Validation, which tested whether the baseline MLP histogram emulator could replace the C++ physics backend for likelihood inference once retrained on the current version.

---

## 1. Accomplishments

1. **Dataset Regeneration**:
   * Validated the 1k configurations dataset `datasets_backend_current_1k` generated on the updated C++ physics backend.
   * Confirmed dataset stability and verified parameter coverage ranges.

2. **MLP Retraining**:
   * Retrained the baseline MLP (`[128, 128, 128]`, GELU, Softmax output) on the updated C++ backend dataset.
   * Achieved low validation loss (KL loss ~0.027 in-distribution) with early stopping triggered at epoch 60.
   * Verified training and PDF agreement plots.

3. **Nelder-Mead Point-Estimate Rerun**:
   * Rerun parameter recovery on 20 random cosmologies.
   * Checked recovery biases: the emulator exhibits significant systematic biases (e.g. mean bias of $-0.087$ in $\sigma_8$ vs $-0.012$ in the simulator).

4. **2D Likelihood-Surface Grid Validation**:
   * Created and executed `ml/run_likelihood_surface_validation.py` to compare likelihood surfaces on a 20x20 grid of $(\Omega_M \times \sigma_8)$ with fixed $h=0.67$ at $z \in \{0.5, 1.5, 2.5\}$ using 10 CPU cores.
   * Discovered near-zero contour overlaps (Jaccard index of $0\%$ for $z=0.5$ and $1.5$, and only $6.7\%$ for $z=2.5$ at $2\sigma$ level) and high MLE location offsets, proving a systematic mismatch.

5. **QA & Regression Testing**:
   * Created `tests/test_backend_consistent_training.py` and `tests/test_backend_current_likelihood_benchmark.py`.
   * Verified that all 29 tests pass successfully.

---

## 2. Verification Outcomes

* **All 29 Automated Tests Passed**:
  ```text
  tests/test_backend_consistent_training.py ..                             [  6%]
  tests/test_backend_current_likelihood_benchmark.py ...                   [ 17%]
  tests/test_derivative_audit.py .                                         [ 20%]
  ...
  ======================== 29 passed in 71.67s =========================
  ```

* **Visual Exhibits Generated**:
  * **Redshift Likelihood Contours**: Overlays simulator confidence contours (dashed orange) against emulator contours (solid blue).
    * [z=0.5 Likelihood Grid](file:///Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12/figures/likelihood_grid_z5.png)
    * [z=1.5 Likelihood Grid](file:///Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12/figures/likelihood_grid_z15.png)
    * [z=2.5 Likelihood Grid](file:///Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12/figures/likelihood_grid_z25.png)
  * **Parameter Recovery Bias Scatter**: [parameter_bias.png](file:///Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12/figures/parameter_bias.png)
  * **Runtime Speedup Comparison**: [runtime_speedup.png](file:///Users/baltabay/.gemini/antigravity/brain/f3f9b801-1f10-4591-9e78-0368fe2e4a12/figures/runtime_speedup.png)

---

## 3. Verdict & Recommendation

* **Recommendation**: **Option B: "Histogram emulator is insufficient for production likelihood replacement. Normalizing Flows (NSF) are immediately justified."**
* **Justification**: Despite retraining on the consistent physics backend, the discrete histogram MLP fails to reproduce the correct likelihood surface, yielding systematic offsets in MLE parameters and zero posterior contour overlap. This is due to the MLP's failure to model the low-probability tails accurately and its lack of continuity constraints. Normalizing Flows are required to map continuous PDFs and capture tail behaviors for high-precision inference.
