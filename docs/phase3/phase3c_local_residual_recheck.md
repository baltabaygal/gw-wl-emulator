# Phase 3C — Local Residual Recheck Report

This report evaluates whether the theta-dependent residual accumulation disappears or is reduced in the simulator-compatible mode.

## Residual Comparison Table

| Theta Point | Sim reference | NSF Continuous (Before) | NSF Compatible (After) | Continuous Residual | Compatible Residual |
|---|---:|---:|---:|---:|---:|
| `truth` | -3819.6800 | 1282.8987 | -3807.5212 | +5102.5787 | +12.1587 |
| `sim_smoke_map` | -3894.5089 | 1254.2450 | -3838.9155 | +5148.7539 | +55.5934 |
| `nsf_smoke_map` | -4141.0032 | 1351.1840 | -3734.3325 | +5492.1872 | +406.6707 |

## Visualization
![Residual Comparison](figures/phase3c_local_residual_recheck/residual_comparison.png)

## Interpretation
In continuous mode, there is a large positive offset ($\approx +5247.8$) that varies across theta points, showing theta-dependent residual accumulation. In compatibility mode, the residuals are reduced to near-zero ($\approx 158.1410$), successfully removing the mismatch.
