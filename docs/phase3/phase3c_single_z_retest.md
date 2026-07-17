# Phase 3C — Single-z Retest Report

This report evaluates whether the compatibility patch resolves the fixed-redshift failure modes.

## Single-z Metrics Table

| Redshift (z) | Mode | JSD | TV | 68% Overlap | 95% Overlap | MAP offset (OmegaM, sigma8) |
|---:|---|---:|---:|---:|---:|---|
| 0.5 | Continuous | 0.693147 | 1.000000 | 0.000 | 0.000 | (-0.1091, +0.2182) |
| 0.5 | Compatible | 0.693147 | 1.000000 | 0.000 | 0.000 | (-0.1091, +0.2182) |
| | | | | | | |
| 1.5 | Continuous | 0.663012 | 0.987604 | 0.000 | 0.059 | (+0.0545, -0.1818) |
| 1.5 | Compatible | 0.642212 | 0.975962 | 0.000 | 0.059 | (+0.0545, -0.1818) |
| | | | | | | |
| 2.5 | Continuous | 0.688028 | 0.998655 | 0.000 | 0.000 | (+0.0545, -0.1091) |
| 2.5 | Compatible | 0.688061 | 0.998665 | 0.000 | 0.000 | (-0.0909, +0.2182) |
| | | | | | | |

## Visualizations

- z=0.5: `plots/figures/phase3c_single_z_retest/compatible_z0p5.png`
- z=1.5: `plots/figures/phase3c_single_z_retest/compatible_z1p5.png`
- z=2.5: `plots/figures/phase3c_single_z_retest/compatible_z2p5.png`
