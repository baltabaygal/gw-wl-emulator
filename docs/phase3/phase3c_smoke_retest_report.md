# Phase 3C — Smoke Retest Report

This report compares the posteriors calculated with the simulator reference, the continuous NSF, and the simulator-compatible NSF.

## Performance Comparison

| Likelihood Mode | JSD | TV | 68% Overlap | 95% Overlap | MAP offset (OmegaM, sigma8) |
|---|---:|---:|---:|---:|---|
| Continuous | 0.671407 | 0.992474 | 0.000 | 0.000 | (-0.1273, +0.2909) |
| Compatible | 0.686908 | 0.997889 | 0.000 | 0.000 | (-0.1455, +0.2909) |

## Contours

### Simulator-Compatible Mode
![Compatible Contours](figures/phase3c_smoke_retest/compatible_contours.png)

### Continuous Mode (Reference)
![Continuous Contours](figures/phase3c_smoke_retest/continuous_contours.png)
