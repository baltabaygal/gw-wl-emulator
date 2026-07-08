# Phase 3D — Posterior Statistics Scaling Report

Summary of posterior metrics across increasing event counts.

## Performance Table

| Catalog Type | N | Mean JSD | Mean TV | Mean 68% Overlap | Mean 95% Overlap |
|---|---:|---:|---:|---:|---:|
| `mixed_uniform` | 250 | 0.638541 | 0.962944 | 0.000 | 0.044 |
| `mixed_uniform` | 1000 | 0.691229 | 0.999409 | 0.000 | 0.000 |
| `mixed_uniform` | 5000 | 0.693147 | 1.000000 | 0.000 | 0.000 |
| `mixed_low_z_dominated` | 250 | 0.613629 | 0.938065 | 0.000 | 0.089 |
| `mixed_low_z_dominated` | 1000 | 0.693091 | 0.999973 | 0.000 | 0.000 |
| `mixed_low_z_dominated` | 5000 | 0.693147 | 1.000000 | 0.000 | 0.000 |
| `mixed_high_z_dominated` | 250 | 0.580529 | 0.893014 | 0.056 | 0.068 |
| `mixed_high_z_dominated` | 1000 | 0.680017 | 0.994408 | 0.000 | 0.000 |
| `mixed_high_z_dominated` | 5000 | 0.693147 | 1.000000 | 0.000 | 0.000 |

## Scaling Curves
![JSD Scaling](figures/phase3d_scaling/jsd_scaling.png)
![TV Scaling](figures/phase3d_scaling/tv_scaling.png)
