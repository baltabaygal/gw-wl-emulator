# Phase 3C — Likelihood Compatibility Report

This report evaluates whether aligning the likelihood definition of the simulator reference and the Neural Spline Flow (NSF) model resolves the posterior disagreement observed in Phase 3B.

## 1. Phase 3B Diagnosis Recap
Phase 3B identified a massive mismatch between the simulator reference posterior and the NSF emulator posterior (e.g. JSD $\approx 0.61$ and TV $\approx 0.95$ for the smoke catalog). A key hypothesis was that the two paths were evaluating different likelihood objects: the simulator uses a histogram-like clamped and floored ($10^{-12}$) probability mass, whereas the NSF evaluates an unclamped continuous probability density.

## 2. Exact Likelihood-Definition Mismatch
The audit confirmed three main differences:
1. **Density vs. Bin Mass**: Simulator uses probability mass in bins, whereas NSF evaluates continuous density. This introduces a scale factor of $\approx \log(\text{bin\_width}) \approx -3.91$.
2. **Zero-Probability Floor**: Simulator floors probabilities to $10^{-12}$ while NSF does not.
3. **Clamping**: Simulator clips out-of-range events, whereas NSF evaluates the tail density.

## 3. Compatibility Patch Description
We implemented a `simulator_compatible` likelihood mode in `ml/nsf_likelihood.py`. This mode:
1. Rounds redshifts ($z$) to 2 decimal places to match simulator slice grouping.
2. Clamps catalog events to the simulator bin edges.
3. Evaluates the NSF continuous density at bin centers, converts it to bin probability mass (multiplying by bin widths), normalizes it to sum to 1.0, and applies the $10^{-12}$ floor.

## 4. Local Residual Recheck
At the truth cosmology, the continuous log-likelihood residual was $+5102.58$ log units (mostly the expected $\log(\text{bin\_width})$ offset). Under compatibility mode, the residual at truth drops to **$+12.16$**, representing an alignment of the absolute log-likelihood scale.
However, at the edge/OOB point (`nsf_smoke_map`), the residual is **$+406.67$** log units. The NSF over-predicts the likelihood at this corner point relative to truth, whereas the simulator strongly disfavors it.

## 5. Smoke Retest
Rerunning the 12x12 grid posterior comparison for the mixed-redshift smoke catalog shows:

| Metric | Continuous Mode (Before) | Compatible Mode (After) |
|---|---|---|
| **Posterior JSD** | `0.671407` | `0.686908` |
| **Total Variation (TV)** | `0.992474` | `0.997889` |
| **68% Credible Overlap** | `0.000` | `0.000` |
| **95% Credible Overlap** | `0.000` | `0.000` |
| **MAP offset ($\Omega_M$, $\sigma_8$)** | `(-0.1273, +0.2909)` | `(-0.1455, +0.2909)` |

Allying the likelihood structure did not improve the posterior agreement; it slightly worsened the JSD and TV.

## 6. Single-z Retest
Rerunning the fixed-redshift grids shows a similar lack of improvement:

- **z=0.5**: Continuous JSD: `0.693147` | Compatible JSD: `0.693147`
- **z=1.5**: Continuous JSD: `0.663012` | Compatible JSD: `0.642212`
- **z=2.5**: Continuous JSD: `0.688028` | Compatible JSD: `0.688061`

Fixed-redshift posterior disagreement is not resolved by likelihood compatibility.

## 7. QA and Reproducibility
All unit and regression tests pass (64/64 tests passed). The cache logic correctly captures `likelihood_mode` in the NPZ grid metadata, ensuring cache reproducibility.

## 8. Final Recommendation and Verdict

### **Verdict: Option C**
> [!CAUTION]
> The compatibility patch does **not** resolve the posterior mismatch. We must **revisit the emulator target or redefine the posterior reference object**.

The posterior mismatch is not a simple likelihood-definition artifact (clamping, flooring, density units). It is a physical density modeling error: the NSF emulator's shape predictions across parameter space ($\theta$) carry systematic errors that accumulate over $N=1000$ catalog events, shifting the posterior modes to the edge of the prior bounds. We recommend investigating the NSF model capacity or changing the emulator target representation.
