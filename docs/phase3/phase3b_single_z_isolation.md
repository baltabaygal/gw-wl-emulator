# Phase 3B Single-z Isolation Benchmark

Fixed-redshift catalogs were evaluated with the same central cosmology as the smoke catalog and the same 2D posterior axes.

| z | JSD | TV | 68% overlap | 95% overlap | dOmegaM MAP | dsigma8 MAP | Max normalized shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.693147 | 1.000000 | 0.000 | 0.000 | -0.1091 | +0.2182 | 21.952 |
| 1.5 | 0.663012 | 0.987604 | 0.000 | 0.059 | +0.0545 | -0.1818 | 1.077 |
| 2.5 | 0.688028 | 0.998655 | 0.000 | 0.000 | +0.0545 | -0.1091 | 69037.875 |

## Interpretation

If the fixed-redshift rows are healthy while the mixed catalog fails, the next target is mixed-catalog aggregation or grid resolution. If one or more fixed-redshift rows also fail, the mismatch is already present before redshift mixing and should be treated as density-model or low-z behavior.

## Figures

- z=0.5: `plots/figures/phase3b_single_z/single_z_z0p5_contours.png`
- z=1.5: `plots/figures/phase3b_single_z/single_z_z1p5_contours.png`
- z=2.5: `plots/figures/phase3b_single_z/single_z_z2p5_contours.png`
