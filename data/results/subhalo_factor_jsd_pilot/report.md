# subhalo_factor PDF-level JSD acceptance (model 3 vs brute truth)

Truth = brute subhalo sampling (model 1 arm, every clump above m_floor=1e7 explicit), two independent seed halves, combined. Default fixed-<N>=100 threshold rule, full model, subhalo on. JSD in nats, 120 shared bins; floor_pred = half-split floor rescaled by (1/N1+1/N2); excess = max(JSD - floor, 0).

## z1   (floor_half=8.20e-03, n_truth=4,800)

| config | kwargs | kthr_clump | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| truth | subhalo_brute=1,subhalo_model=1 | - | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0753 | 1.0007 | 1.288 | 1.820 | 2.66 |
| f1em5 | subhalo_model=3,subhalo_factor=1e-05 | 1.28e-09 | 3.24e-03 | 4.10e-03 | 0.00e+00 | 0.0765 | 1.0003 | 1.317 | 1.821 | 2.48 |
| f1em3 | subhalo_model=3,subhalo_factor=0.001 | 1.28e-07 | 3.21e-03 | 4.10e-03 | 0.00e+00 | 0.0802 | 1.0009 | 1.273 | 1.865 | 6.39 |
| f1em2 | subhalo_model=3,subhalo_factor=0.01 | 1.28e-06 | 4.34e-03 | 4.10e-03 | 2.45e-04 | 0.0724 | 1.0003 | 1.275 | 1.640 | 2.93 |
| f1em1 | subhalo_model=3,subhalo_factor=0.1 | 1.28e-05 | 3.67e-03 | 4.10e-03 | 0.00e+00 | 0.0719 | 1.0004 | 1.296 | 1.595 | 2.38 |
| m1_f1em2 | subhalo_model=1,subhalo_factor=0.01 | 1.28e-06 | 3.47e-03 | 4.10e-03 | 0.00e+00 | 0.0751 | 1.0003 | 1.310 | 1.701 | 2.88 |

## z5   (floor_half=5.85e-03, n_truth=4,797)

| config | kwargs | kthr_clump | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| truth | subhalo_brute=1,subhalo_model=1 | - | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.2522 | 1.0030 | 2.382 | 5.814 | 26.98 |
| f1em5 | subhalo_model=3,subhalo_factor=1e-05 | 9.04e-09 | 3.71e-03 | 2.92e-03 | 7.83e-04 | 0.2640 | 1.0038 | 2.407 | 5.387 | 215.53 |
| f1em3 | subhalo_model=3,subhalo_factor=0.001 | 9.04e-07 | 4.03e-03 | 2.92e-03 | 1.11e-03 | 0.2651 | 1.0066 | 2.544 | 8.490 | 22.52 |
| f1em2 | subhalo_model=3,subhalo_factor=0.01 | 9.04e-06 | 3.69e-03 | 2.92e-03 | 7.68e-04 | 0.2735 | 1.0059 | 2.588 | 9.582 | 40.16 |
| f1em1 | subhalo_model=3,subhalo_factor=0.1 | 9.04e-05 | 2.77e-03 | 2.92e-03 | 0.00e+00 | 0.2406 | 1.0047 | 2.392 | 3.852 | 12.63 |
| m1_f1em2 | subhalo_model=1,subhalo_factor=0.01 | 9.04e-06 | 4.00e-03 | 2.92e-03 | 1.08e-03 | 0.2651 | 1.0073 | 2.134 | 12.646 | 38.62 |
