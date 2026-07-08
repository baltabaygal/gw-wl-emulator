# Phase 2B — NSF PDF and Tail Validation Report

Generated on: 2026-06-15 15:57:02 UTC
Model: `data/models/conditional_nsf_backend_current.pt`

## 1. Summary Metrics

| Split Type | Count | Mean KL | Mean JSD | Mean Wasserstein |
|---|---|---|---|---|
| interpolation | 527 | 0.095844 | 0.008802 | 0.010383 |
| ood | 388 | 0.331471 | 0.033066 | 0.036980 |

## 2. Representative PDF Overlays

### Interpolation Cosmology Configuration
Parameters: $z = 4.24, h = 0.629, \Omega_M = 0.237, \sigma_8 = 0.731$
![PDF Overlay interpolation](figures/phase2b_nsf_pdf_validation/pdf_overlay_interpolation.png)

### Ood Cosmology Configuration
Parameters: $z = 9.59, h = 0.693, \Omega_M = 0.417, \sigma_8 = 1.206$
![PDF Overlay ood](figures/phase2b_nsf_pdf_validation/pdf_overlay_ood.png)

## 3. Moment and Tail Quantiles Analysis

We compare the recovered moments and high-magnification tails for the representative configurations.

### Interpolation Moments Table
| Metric | Simulator | NSF | Difference |
|---|---|---|---|
| Mean | 0.005960 | 0.006369 | +0.000409 |
| Variance | 0.023309 | 0.022150 | -0.001159 |
| Skewness | 1.880782 | 1.559288 | -0.321495 |
| Kurtosis | 8.457068 | 5.070194 | -3.386874 |

### Interpolation High-Magnification Tail Quantiles
| Quantile | Simulator | NSF | Difference |
|---|---|---|---|
| 99% | 0.510091 | 0.530870 | +0.020779 |
| 99.5% | 0.657875 | 0.666164 | +0.008290 |
| 99.9% | 1.038356 | 0.963336 | -0.075020 |

### Interpolation Tail Probability Masses
| Tail Bound | Simulator | NSF | Difference |
|---|---|---|---|
| P(lnmu > 0.2) | 0.092178 | 0.088300 | -0.003878 |
| P(lnmu > 0.5) | 0.010580 | 0.012300 | +0.001720 |
| P(lnmu > 1.0) | 0.001351 | 0.000500 | -0.000851 |

### Ood Moments Table
| Metric | Simulator | NSF | Difference |
|---|---|---|---|
| Mean | 0.063221 | 0.033315 | -0.029907 |
| Variance | 0.287847 | 0.134375 | -0.153472 |
| Skewness | 1.896823 | 0.974809 | -0.922014 |
| Kurtosis | 4.555952 | 0.463671 | -4.092281 |

### Ood High-Magnification Tail Quantiles
| Quantile | Simulator | NSF | Difference |
|---|---|---|---|
| 99% | 2.376971 | 1.115191 | -1.261780 |
| 99.5% | 2.500000 | 1.159983 | -1.340017 |
| 99.9% | 2.500000 | 1.208719 | -1.291281 |

### Ood Tail Probability Masses
| Tail Bound | Simulator | NSF | Difference |
|---|---|---|---|
| P(lnmu > 0.2) | 0.281635 | 0.256600 | -0.025035 |
| P(lnmu > 0.5) | 0.147688 | 0.121300 | -0.026388 |
| P(lnmu > 1.0) | 0.062254 | 0.025600 | -0.036654 |
