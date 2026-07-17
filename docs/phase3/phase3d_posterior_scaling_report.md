# Phase 3D — Posterior Statistics Scaling Report

This report evaluates whether the Neural Spline Flow (NSF) posterior mismatch is a finite-statistics noise artifact or a real model limitation by measuring the scaling law of the discrepancy.

## 1. Motivation
In Phase 3C, we aligned the likelihood definitions of the C++ simulator reference and the NSF emulator. However, the posterior mode disagreement persisted for the $N=1000$ smoke catalog. This phase investigates if this mismatch decays asymptotically as statistical support (information) increases.

## 2. Benchmark Design
We designed a controlled scaling matrix:
- **Catalog sizes**: $N \in \{250, 1000, 5000\}$.
- **Catalog types**:
  - `mixed_uniform` (Central Mixed: $[1/3, 1/3, 1/3]$ weights on discrete redshifts $\{0.5, 1.5, 2.5\}$)
  - `mixed_low_z_dominated` (Low-z Dominated: $[0.6, 0.3, 0.1]$ weights on support)
  - `mixed_high_z_dominated` (High-z Dominated: $[0.1, 0.3, 0.6]$ weights on support)
- **Grid settings**: 12x12 grid resolution, using `simulator_compatible` likelihood mode with `nsim_per_z = 2000` (to maintain speed and precision).
- **Repetitions**: 3 runs per configuration with seeds `710001, 710002, 710003` to average over catalog realizations.

## 3. Scaling Results
The mean JSD and TV metrics averaged over seeds for each catalog size and type are:

| Catalog Type | N | Mean JSD | Mean TV | Mean 68% Overlap | Mean 95% Overlap |
|---|---:|---:|---:|---:|---:|
| `mixed_uniform` | 250 | `0.638541` | `0.962944` | `0.000` | `0.044` |
| `mixed_uniform` | 1000 | `0.691229` | `0.999409` | `0.000` | `0.000` |
| `mixed_uniform` | 5000 | `0.693147` | `1.000000` | `0.000` | `0.000` |
| `mixed_low_z_dominated` | 250 | `0.613629` | `0.938065` | `0.000` | `0.089` |
| `mixed_low_z_dominated` | 1000 | `0.693091` | `0.999973` | `0.000` | `0.000` |
| `mixed_low_z_dominated` | 5000 | `0.693147` | `1.000000` | `0.000` | `0.000` |
| `mixed_high_z_dominated` | 250 | `0.580529` | `0.893014` | `0.056` | `0.068` |
| `mixed_high_z_dominated` | 1000 | `0.680017` | `0.994408` | `0.000` | `0.000` |
| `mixed_high_z_dominated` | 5000 | `0.693147` | `1.000000` | `0.000` | `0.000` |

## 4. Low-z vs Mixed-catalog Results
Both the low-z dominated and uniform mixed catalogs behave similarly: they reach complete saturation (JSD $\approx 0.693$, TV $\approx 1.0$) at $N=1000$ and $N=5000$. High-z dominated catalogs have slightly better overlap at $N=250$, but also diverge completely to maximum mismatch by $N=5000$. 

## 5. Trend Fits
We fit the scaling curves with the power-law relation:
$$\text{Metric}(N) = A \cdot N^{-\alpha} + B$$
where $B$ represents the asymptotic floor. The estimated parameters are:

- **`mixed_uniform`**: $B_{\text{JSD}} = 0.6931$ | $B_{\text{TV}} = 1.0000$
- **`mixed_low_z_dominated`**: $B_{\text{JSD}} = 0.6931$ | $B_{\text{TV}} = 1.0000$
- **`mixed_high_z_dominated`**: $B_{\text{JSD}} = 0.6931$ | $B_{\text{TV}} = 1.0000$

All fit statuses are classified as **SATURATED**.

## 6. Cache and QA Status
All cache keys are strict and verified. Preprocessing stats hash, NSF checkpoint hash, likelihood mode, catalog hash, and git commit verify correctly. All 64 tests in the test suite pass.

## 7. Final Interpretation and Verdict

### **Verdict: Option C**
> [!CAUTION]
> The posterior mismatch **does not** shrink with increasing statistics. It persists and saturates at maximum disagreement (JSD $\to \log(2) \approx 0.6931$, TV $\to 1.0$) as information increases.

This is the classic signature of **systematic bias**. As the catalog size $N$ increases, the posterior peaks become narrower. Because the NSF model has small systematic density estimation/gradient errors, the NSF posterior mode is offset from the true simulator posterior mode. As statistics increase, the narrow posteriors pull apart entirely, leading to zero overlap. The NSF emulator misses the true posterior geometry and **needs a model repair** before proceeding to production.
