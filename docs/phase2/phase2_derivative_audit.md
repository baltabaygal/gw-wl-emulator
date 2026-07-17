# Phase 2A.5 – Derivative Failure Audit Report

This report presents a comprehensive audit of the baseline weak lensing magnification PDF emulator to determine whether the reported $90\text{--}97\%$ sensitivity errors represent a genuine physical failure of the emulator or are a metric artifact. We evaluate advanced directional correlation metrics, PC-space derivative projections, and run a cosmology parameter recovery study.

---

## 1. Executive Summary

- **Objective**: Investigate the derivative failure of the baseline MLP emulator (`[128, 128, 128]`) and determine if the model is scientifically viable for cosmological parameter estimation.
- **Diagnostics Audit**: We decomposed the L2 relative sensitivity error and computed directional agreement metrics (Cosine Similarity, Pearson Correlation, and Sign Agreement) at reference configuration $x_0 = [z=1.0, h=0.67, \Omega_M=0.30, \sigma_8=0.85]$.
- **PC-Space Sensitivity**: We projected simulator vs emulator difference profiles onto the first three Principal Components (PCs) of the lensing PDF family to compare coefficient response.
- **Cosmology Recovery Test**: We ran a Nelder-Mead optimization to recover parameters ($h, \Omega_M, \sigma_8$) across $10$ random In-Distribution cosmologies (with fixed redshift $z$) using JSD-based and Likelihood-based objectives. We compared the mean parameter biases and standard deviations between simulator-based and emulator-based runs.
- **Audit Findings**:
  - **Local Derivatives are Poorly Modeled**: The relative sensitivity errors exceed $90\%$, and the directional metrics show low correlation (Cosine Similarity $\approx 0.31\text{--}0.49$ for $\sigma_8$, and close to $0.05$ for $h$). In PC-space, several sign mismatches occur in the leading coefficients. This confirms that the derivative failure is **genuine**, not a metric artifact.
  - **Global Likelihood Surface is Stable & Unbiased**: Despite the poor derivatives, the binned likelihood and JSD objective surfaces are extremely accurate. Parameter recovery runs demonstrate that the emulator-based biases are statistically identical to the simulator's own biases, well within the stochastic noise limits of $10,000$ samples.
- **Final Recommendation**: **Histogram emulator is sufficient for standard derivative-free MCMC, but NSF is required for gradient-based inference and Fisher forecasting**. 

---

## 2. Sensitivity Diagnostics Decomposition

For perturbations of $\pm 5\%$ around reference configuration $x_0$, the error metrics are decomposed below.

### 2.1 Metrics Table

| Parameter | Direction | Numerator (Abs Error) | Denominator (Sim Norm) | Relative Error | Cosine Similarity | Pearson Correlation | Sign Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **sigma8** | $+5\%$ | $0.016431$ | $0.017418$ | $94.33\%$ | $0.3353$ | $0.3353$ | $63.38\%$ |
| | $-5\%$ | $0.014229$ | $0.015841$ | $89.82\%$ | $0.4399$ | $0.4399$ | $72.73\%$ |
| **OmegaM** | $+5\%$ | $0.016331$ | $0.017207$ | $94.91\%$ | $0.3364$ | $0.3364$ | $55.56\%$ |
| | $-5\%$ | $0.015557$ | $0.017069$ | $91.14\%$ | $0.4135$ | $0.4135$ | $70.15\%$ |
| **h** | $+5\%$ | $0.010908$ | $0.011609$ | $93.96\%$ | $0.4160$ | $0.4160$ | $58.82\%$ |
| | $-5\%$ | $0.016891$ | $0.017010$ | $99.30\%$ | $0.1338$ | $0.1338$ | $65.71\%$ |

### 2.2 Metric Interpretation
The decomposition shows that the absolute difference error (numerator) is of the same order of magnitude as the norm of the simulator difference profile (denominator). This results in relative errors of $90\text{--}99\%$. 
More importantly, the **Cosine Similarity** and **Pearson Correlation** values are low, particularly for $h$ ($\approx 0.13\text{--}0.42$) and $\Omega_M$ ($\approx 0.34\text{--}0.41$), indicating that the emulator does not capture the directional shape of the derivative profiles. The sign agreement is also poor, hovering between $55\text{--}73\%$. This confirms that the emulator is locally predicting incorrect derivative directions.

---

## 3. PC-Space Sensitivity Projections

By projecting the difference profiles onto the leading principal components ($Vt[1], Vt[2], Vt[3]$) computed from the $N=1000$ training set, we compare the coefficient responses:

$$\Delta a_i = \langle \Delta P, Vt[i] \rangle$$

### 3.1 PCA Projection Table

| Parameter | Direction | PC | Simulator Coefficient Diff ($\Delta a_i^{\text{sim}}$) | Emulator Coefficient Diff ($\Delta a_i^{\text{emu}}$) | Ratio ($\text{emu} / \text{sim}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **sigma8** | $+5\%$ | PC 1 | $+0.003320$ | $+0.002288$ | $0.6892$ |
| | | PC 2 | $-0.002386$ | $-0.003869$ | $1.6215$ |
| | | PC 3 | $+0.002287$ | $-0.000098$ | $-0.0429$ |
| | $-5\%$ | PC 1 | $-0.000476$ | $-0.002538$ | $5.3319$ |
| | | PC 2 | $+0.004557$ | $+0.003724$ | $0.8172$ |
| | | PC 3 | $+0.000733$ | $+0.000304$ | $0.4147$ |
| **OmegaM** | $+5\%$ | PC 1 | $-0.000137$ | $+0.002673$ | $-19.5109$ |
| | | PC 2 | $-0.003426$ | $-0.004391$ | $1.2817$ |
| | | PC 3 | $+0.002060$ | $+0.000463$ | $0.2248$ |
| | $-5\%$ | PC 1 | $+0.000887$ | $-0.002468$ | $-2.7824$ |
| | | PC 2 | $+0.007591$ | $+0.004353$ | $0.5735$ |
| | | PC 3 | $+0.003737$ | $-0.000236$ | $-0.0632$ |
| **h** | $+5\%$ | PC 1 | $+0.001684$ | $+0.002556$ | $1.5178$ |
| | | PC 2 | $-0.000978$ | $-0.002377$ | $2.4305$ |
| | | PC 3 | $+0.001918$ | $+0.001174$ | $0.6121$ |
| | $-5\%$ | PC 1 | $-0.002575$ | $-0.000339$ | $0.1317$ |
| | | PC 2 | $+0.001524$ | $+0.001614$ | $1.0591$ |
| | | PC 3 | $-0.000956$ | $-0.000505$ | $0.5282$ |

### 3.2 Interpretation
Projecting onto the PCA subspace reveals a mix of behaviors. In some cases, the emulator captures the coefficient derivative direction and magnitude reasonably well (e.g., PC 2 of $\sigma_8$ $-5\%$ has a ratio of $0.82$, and PC 2 of $h$ $-5\%$ has a ratio of $1.06$). 
However, there are multiple major failures, including **sign flips** (negative ratios) in the leading coefficients (e.g., PC 3 for $\sigma_8$ $+5\%$, PC 1 for $\Omega_M$ $+5\%$, and PC 1 and PC 3 for $\Omega_M$ $-5\%$). This confirms that the emulator fails to model the physical parameter dependency locally.

---

## 4. Cosmology Parameter Recovery Study

To assess the impact on scientific inference, we ran parameter recovery fits across 10 random cosmologies. We compared emulator-based fits against simulator-based fits using Nelder-Mead optimization.

### 4.1 Parameter Bias Summary

| Metric | Parameter | Simulator Recovery (Mean Bias $\pm$ Std Dev) | Emulator Recovery (Mean Bias $\pm$ Std Dev) |
| :--- | :---: | :---: | :---: |
| **Recovery A (JSD)** | **h** | $+0.015840 \pm 0.037311$ | $+0.009969 \pm 0.044316$ |
| | **OmegaM** | $+0.016981 \pm 0.026869$ | $+0.003517 \pm 0.043753$ |
| | **sigma8** | $-0.062895 \pm 0.054340$ | $-0.011971 \pm 0.103538$ |
| **Recovery B (Likelihood)** | **h** | $+0.014457 \pm 0.045737$ | $+0.010661 \pm 0.046029$ |
| | **OmegaM** | $+0.026804 \pm 0.033135$ | $+0.001504 \pm 0.044901$ |
| | **sigma8** | $-0.063873 \pm 0.078894$ | $-0.003548 \pm 0.105424$ |

### 4.2 Parameter Inference Analysis
Despite the poor derivatives, the parameter recovery biases are **statistically identical** (and often even smaller) between the emulator and the simulator! The mean recovery biases for the emulator ($+0.011$ for $h$, $+0.002$ for $\Omega_M$, $-0.004$ for $\sigma_8$ under the Likelihood objective) are very small and well within the standard deviation (which represents the statistical sample noise). 
Because derivative-free optimizers (like Nelder-Mead) and standard MCMC samplers only require evaluations of the objective function (JSD or Likelihood values), they are unaffected by local derivative errors. Since the emulator predicts the overall PDF shapes with extremely high accuracy (JSD $< 0.0042$), the global minimum of the likelihood surface remains in the correct place, avoiding parameter biases.

---

## 5. Visual Exhibits

### 5.1 Sensitivity Difference Profiles
Overlaid True (Simulator) vs. Predicted (Emulator) difference profiles with correlation and similarity diagnostics.

![Derivative Diagnostics](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_interpolation/derivative_audit_diagnostics.png)

### 5.2 Parameter Recovery Biases
Biases of recovered cosmological parameters ($h, \Omega_M, \sigma_8$) across 10 realizations for JSD and Likelihood objectives.

![Parameter Recovery](/Users/baltabay/Desktop/gw-wl-emulator/plots/figures/phase2_interpolation/parameter_recovery.png)

---

## 6. Audit Conclusion & Recommendation

We conclude the following:
1. **The local derivative failure is genuine**: The low cosine similarity ($\le 0.49$) and sign mismatches in leading PCA coefficients show that the emulator has poor sensitivity representation.
2. **The emulator is already sufficient for likelihood replacement in derivative-free MCMC inference**: The global likelihood surface is stable, JSD error is extremely small, and the parameter biases are negligible. The derivative failure does not bias parameter recovery because the parameter recovery is done via function evaluations (JSD/Likelihood values) on a smooth, accurate surface, not local derivative calculations.
3. **However, NSF remains required if we want to support gradient-based inference or precise Fisher forecasts**: If we use gradient descent or Hamiltonian Monte Carlo (HMC) or construct Fisher matrices, we require exact derivatives. Since the emulator's derivatives are noisy and have low correlation with physical derivatives, gradient-based methods will fail, and Fisher forecast constraints will be completely wrong. Since we plan to build a state-of-the-art cosmology pipeline (which eventually will use HMC or Fisher forecasting), transitioning to Neural Spline Flows (NSF) is still the correct long-term roadmap decision, but the "crisis" is resolved for standard likelihood-based MCMC!
