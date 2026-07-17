# subhalo_factor PDF-level JSD acceptance (model 3 vs brute truth)

Truth = brute subhalo sampling (model 1 arm, every clump above m_floor=1e7 explicit), two independent seed halves, combined. Default fixed-<N>=100 threshold rule, full model, subhalo on. JSD in nats, 120 shared bins; floor_pred = half-split floor rescaled by (1/N1+1/N2); excess = max(JSD - floor, 0).

## z1   (floor_half=2.30e-04, n_truth=239,997)

| config | kwargs | kthr_clump | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| truth | subhalo_brute=1,subhalo_model=1 | - | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.0757 | 1.0003 | 1.292 | 1.809 | 3.79 |
| f1em5 | subhalo_model=3,subhalo_factor=1e-05 | 1.28e-09 | 1.44e-04 | 1.15e-04 | 2.84e-05 | 0.0751 | 1.0002 | 1.291 | 1.765 | 4.01 |
| f1em3 | subhalo_model=3,subhalo_factor=0.001 | 1.28e-07 | 1.14e-04 | 1.15e-04 | 0.00e+00 | 0.0773 | 1.0004 | 1.292 | 1.797 | 4.06 |
| f1em2 | subhalo_model=3,subhalo_factor=0.01 | 1.28e-06 | 1.45e-04 | 1.15e-04 | 3.03e-05 | 0.0777 | 1.0004 | 1.294 | 1.788 | 4.08 |
| f1em1 | subhalo_model=3,subhalo_factor=0.1 | 1.28e-05 | 1.44e-04 | 1.15e-04 | 2.85e-05 | 0.0777 | 1.0006 | 1.291 | 1.792 | 4.04 |
| m1_f1em2 | subhalo_model=1,subhalo_factor=0.01 | 1.28e-06 | 1.61e-04 | 1.15e-04 | 4.63e-05 | 0.0772 | 1.0005 | 1.298 | 1.814 | 3.56 |

## z5   (floor_half=2.14e-04, n_truth=239,887)

| config | kwargs | kthr_clump | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| truth | subhalo_brute=1,subhalo_model=1 | - | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.2616 | 1.0041 | 2.478 | 7.109 | 44.73 |
| f1em5 | subhalo_model=3,subhalo_factor=1e-05 | 9.04e-09 | 1.38e-04 | 1.07e-04 | 3.06e-05 | 0.2611 | 1.0037 | 2.415 | 7.360 | 69.91 |
| f1em3 | subhalo_model=3,subhalo_factor=0.001 | 9.04e-07 | 1.55e-04 | 1.07e-04 | 4.74e-05 | 0.2616 | 1.0054 | 2.439 | 7.650 | 61.32 |
| f1em2 | subhalo_model=3,subhalo_factor=0.01 | 9.04e-06 | 1.32e-04 | 1.07e-04 | 2.47e-05 | 0.2605 | 1.0060 | 2.423 | 7.433 | 60.52 |
| f1em1 | subhalo_model=3,subhalo_factor=0.1 | 9.04e-05 | 1.33e-04 | 1.07e-04 | 2.54e-05 | 0.2595 | 1.0118 | 2.424 | 7.137 | 42.21 |
| m1_f1em2 | subhalo_model=1,subhalo_factor=0.01 | 9.04e-06 | 2.77e-04 | 1.07e-04 | 1.70e-04 | 0.2559 | 1.0043 | 2.423 | 7.362 | 58.10 |
