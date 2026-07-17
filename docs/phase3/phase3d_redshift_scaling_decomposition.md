# Phase 3D — Low-z vs Mixed-catalog Scaling Decomposition

Comparison of redshift-dominated catalog scaling behaviors.

## Redshift scaling Comparison

| Catalog Type | N | Mean JSD | Mean TV | Width OmegaM | MAP offset OmegaM |
|---|---:|---:|---:|---:|---:|
| `mixed_uniform` | 250 | 0.638541 | 0.962944 | 0.0292 | +0.0364 |
| `mixed_uniform` | 1000 | 0.691229 | 0.999409 | 0.0463 | +0.0364 |
| `mixed_uniform` | 5000 | 0.693147 | 1.000000 | 0.0366 | -0.0364 |
| `mixed_low_z_dominated` | 250 | 0.613629 | 0.938065 | 0.0323 | -0.0364 |
| `mixed_low_z_dominated` | 1000 | 0.693091 | 0.999973 | 0.0129 | -0.0121 |
| `mixed_low_z_dominated` | 5000 | 0.693147 | 1.000000 | 0.0174 | -0.0121 |
| `mixed_high_z_dominated` | 250 | 0.580529 | 0.893014 | 0.0509 | -0.0606 |
| `mixed_high_z_dominated` | 1000 | 0.680017 | 0.994408 | 0.0581 | +0.0485 |
| `mixed_high_z_dominated` | 5000 | 0.693147 | 1.000000 | 0.0153 | +0.0061 |

## Figures
![JSD Redshift Scaling](figures/phase3d_redshift_scaling/jsd_scaling.png)
![TV Redshift Scaling](figures/phase3d_redshift_scaling/tv_scaling.png)
