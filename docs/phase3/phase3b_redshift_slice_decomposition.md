# Phase 3B Redshift-Slice Decomposition

The central mixed catalog was split into low-, mid-, and high-redshift subsets, then each subset was evaluated on the same 2D posterior grid.

| Slice | z | JSD | TV | 68% overlap | 95% overlap | dOmegaM MAP | dsigma8 MAP |
|---|---:|---:|---:|---:|---:|---:|---:|
| low_z | 0.5 | 0.645227 | 0.980663 | 0.000 | 0.059 | -0.1273 | +0.2182 |
| mid_z | 1.5 | 0.687678 | 0.998553 | 0.000 | 0.000 | +0.0182 | -0.1091 |
| high_z | 2.5 | 0.583010 | 0.941186 | 0.056 | 0.054 | +0.0545 | -0.1455 |

## Figures

- low_z: `plots/figures/phase3b_redshift_slice/low_z_contours.png`
- mid_z: `plots/figures/phase3b_redshift_slice/mid_z_contours.png`
- high_z: `plots/figures/phase3b_redshift_slice/high_z_contours.png`
