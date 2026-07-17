# Cross-study shard screen + block-aware floors (2026-07-13)

Mechanism under test: batch-mean kappa anchor (cpp/lensing.cpp::sample_lnmu) -> shared shard shift Delta ~= -2*kappa_max/n. Delta_pred from each shard's own most extreme ray; Delta_obs = median(shard) - median(siblings). KS_fix = KS vs siblings after dropping the ray and un-shifting.

## Shards where the anchor shift is detectable

| study | array | shard | n | lnmu_min | kappa_max | Delta_pred | Delta_obs | Dobs/sig | n(k>3) | KS_raw | KS_fix |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mmin_pd_convergence | z5_main_truthB | 4 | 14991 | -13.84 | 1015 | -0.1354 | -0.1290 | -48.2 | 1 | 0.2882 | 0.0316 |
| mmin_convergence | z10_main_nm200ctl | 3 | 29963 | -12.50 | 518 | -0.0346 | -0.0353 | -15.4 | 3 | 0.0655 | 0.0055 |
| mmin_pd_convergence | z1_main_truthB | 7 | 14999 | -10.18 | 163 | -0.0217 | -0.0203 | -21.9 | 1 | 0.1715 | 0.0136 |
| mmin_pd_convergence | z5_main_m1e5 | 7 | 29983 | -9.28 | 104 | -0.0070 | -0.0062 | -3.3 | 3 | 0.0162 | 0.0034 |
| mmin_convergence | z5_main_m1e5 | 5 | 29982 | -9.23 | 102 | -0.0068 | -0.0055 | -2.9 | 1 | 0.0144 | 0.0063 |
| mmin_pd_convergence | z1_main_truthA | 0 | 14999 | -7.48 | 43 | -0.0057 | -0.0063 | -7.1 | 1 | 0.0514 | 0.0090 |
| mmin_convergence | z5_main_m1e7 | 1 | 29984 | -8.29 | 64 | -0.0043 | -0.0042 | -2.3 | 1 | 0.0129 | 0.0034 |
| mmin_pd_convergence | z1_main_m1e5 | 1 | 29999 | -8.23 | 62 | -0.0041 | -0.0054 | -8.7 | 4 | 0.0447 | 0.0117 |
| mmin_pd_convergence | z1_main_truthB | 3 | 14999 | -6.22 | 23 | -0.0031 | +0.0007 | +0.8 | 1 | 0.0093 | 0.0336 |
| mmin_convergence | z1_sub_truthB | 6 | 15000 | -5.91 | 20 | -0.0027 | -0.0021 | -2.7 | 1 | 0.0191 | 0.0111 |
| nz_convergence | z1_main_nz50 | 2 | 29999 | -6.78 | 31 | -0.0020 | -0.0019 | -3.6 | 1 | 0.0181 | 0.0053 |
| mmin_pd_convergence | z5_main_truthB | 0 | 14991 | -4.34 | 10 | -0.0013 | +0.0165 | +6.2 | 2 | 0.0406 | 0.0434 |
| mmin_pd_convergence | z5_main_truthB | 7 | 14991 | -3.31 | 6 | -0.0008 | +0.0177 | +6.6 | 1 | 0.0370 | 0.0387 |
| nz_convergence | z0.2_main_truthB | 5 | 15000 | -2.49 | 4 | -0.0006 | -0.0006 | -5.5 | 1 | 0.0622 | 0.0048 |
| mmin_convergence | z0.2_main_truthA | 0 | 15000 | -1.40 | 3 | -0.0004 | -0.0004 | -4.0 | 1 | 0.0457 | 0.0067 |
| mmin_pd_convergence | z1_main_truthB | 0 | 14999 | -1.16 | 3 | -0.0004 | +0.0038 | +4.1 | 0 | 0.0338 | 0.0367 |
| mmin_pd_convergence | z1_main_truthB | 5 | 14999 | -1.01 | 3 | -0.0004 | +0.0039 | +4.2 | 0 | 0.0356 | 0.0386 |
| mmin_convergence | z0.2_main_m1e5 | 5 | 30000 | -1.95 | 4 | -0.0002 | -0.0002 | -2.2 | 1 | 0.0267 | 0.0036 |
| mmin_pd_convergence | z1_main_truthB | 2 | 14999 | -0.13 | 0 | -0.0000 | +0.0038 | +4.1 | 0 | 0.0378 | 0.0378 |
| mmin_pd_convergence | z5_main_truthB | 1 | 14991 | -0.44 | 0 | -0.0000 | +0.0193 | +7.2 | 0 | 0.0468 | 0.0468 |
| mmin_pd_convergence | z5_main_truthB | 2 | 14991 | -0.42 | 0 | -0.0000 | +0.0167 | +6.2 | 0 | 0.0465 | 0.0465 |
| mmin_pd_convergence | z5_main_truthB | 3 | 14991 | -0.46 | 0 | -0.0000 | +0.0157 | +5.8 | 0 | 0.0407 | 0.0407 |
| mmin_pd_convergence | z5_main_truthB | 5 | 14991 | -0.45 | 0 | -0.0000 | +0.0180 | +6.7 | 0 | 0.0448 | 0.0448 |
| mmin_pd_convergence | z5_main_truthB | 6 | 14991 | -0.45 | 0 | -0.0000 | +0.0177 | +6.6 | 0 | 0.0475 | 0.0475 |

## Block-aware truth-half floors (exact enumeration of all C(16,8)=12870 shard splits)

sample-level permutation destroys the shared shard shift and is too narrow; the shard is the exchangeable unit. pctl = where the ORIGINAL A|B split sits in the block null.

| study | group | floor_half (sample) | block-null median | block-null 95% | observed A|B | pctl |
|:--|:--|--:|--:|--:|--:|--:|
| mmin_convergence | z0.2_main | 3.026e-04 | 3.000e-04 | 6.067e-04 | 3.026e-04 | 51 |
| mmin_convergence | z10_main | 2.290e-04 | 2.566e-04 | 2.966e-04 | 2.288e-04 | 10 |
| mmin_convergence | z1_main | 2.351e-04 | 2.507e-04 | 3.075e-04 | 2.356e-04 | 31 |
| mmin_convergence | z1_sub | 2.181e-04 | 2.219e-04 | 2.763e-04 | 2.182e-04 | 45 |
| mmin_convergence | z5_main | 2.957e-04 | 2.603e-04 | 3.252e-04 | 2.957e-04 | 83 |
| mmin_pd_convergence | z0.2_main | 3.216e-04 | 2.639e-04 | 3.642e-04 | 3.216e-04 | 87 |
| mmin_pd_convergence | z10_main | 2.166e-04 | 2.468e-04 | 3.035e-04 | 2.168e-04 | 16 |
| mmin_pd_convergence | z1_main | 6.753e-04 | 8.349e-04 | 1.438e-03 | 6.756e-04 | 30 |
| mmin_pd_convergence | z5_main | 2.196e-03 | 2.435e-03 | 2.692e-03 | 2.197e-03 | 4 |
| nz_convergence | z0.2_main | 3.277e-04 | 3.134e-04 | 4.150e-04 | 3.277e-04 | 60 |
| nz_convergence | z10_main | 2.142e-04 | 2.637e-04 | 3.161e-04 | 2.140e-04 | 4 |
| nz_convergence | z1_main | 2.352e-04 | 2.491e-04 | 3.004e-04 | 2.352e-04 | 31 |
| nz_convergence | z1_sub | 2.273e-04 | 2.195e-04 | 2.664e-04 | 2.274e-04 | 61 |
| nz_convergence | z5_main | 2.620e-04 | 2.496e-04 | 3.050e-04 | 2.609e-04 | 63 |
| kappathr_subhalo_jsd | z10_off | 4.347e-04 | 4.333e-04 | 5.295e-04 | 4.348e-04 | 51 |
| kappathr_subhalo_jsd | z10_on | 5.338e-04 | 4.607e-04 | 5.667e-04 | 5.335e-04 | 87 |
| kappathr_subhalo_jsd | z1_off | 2.228e-04 | 2.124e-04 | 2.656e-04 | 2.227e-04 | 64 |
| kappathr_subhalo_jsd | z1_on | 2.651e-04 | 2.265e-04 | 2.759e-04 | 2.650e-04 | 90 |
| subhalo_factor_jsd | z1 | 2.302e-04 | 2.296e-04 | 2.726e-04 | 2.302e-04 | 51 |
| subhalo_factor_jsd | z5 | 2.143e-04 | 2.452e-04 | 3.033e-04 | 2.144e-04 | 15 |