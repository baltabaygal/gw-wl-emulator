# Bias-layer prototype: grid-decoupled LOS-correlated field

Old arm = replica of the current per-cell iid lognormal layer (lensing.cpp deltaNhfNFW + sampling loop, kappa only, pure Poisson counts). New arm = same halo layer, environment drawn from the pencil-projected linear P(k) with per-cell cylinder-window amplitude (no free scale).

## (a) MC: tail + body vs Nz (fixed-<N>=100 rule per Nz)

| zs | Nz | arm | f(k>1) | f(k>0.5) | q99.9 | Var(|k|<0.5) |
|--:|--:|:--|--:|--:|--:|--:|
| 1 | 25 | old | 0.00e+00 | 5.00e-05 | 0.2870 | 7.3100e-04 |
| 1 | 25 | new | 0.00e+00 | 5.00e-05 | 0.2696 | 7.2423e-04 |
| 1 | 100 | old | 5.00e-05 | 1.50e-04 | 0.3209 | 7.7175e-04 |
| 1 | 100 | new | 0.00e+00 | 1.50e-04 | 0.3022 | 8.1221e-04 |
| 1 | 200 | old | 0.00e+00 | 1.50e-04 | 0.3342 | 8.6083e-04 |
| 1 | 200 | new | 0.00e+00 | 5.00e-05 | 0.2961 | 8.4493e-04 |
| 1 | 400 | old | 0.00e+00 | 1.50e-04 | 0.3041 | 8.2711e-04 |
| 1 | 400 | new | 0.00e+00 | 5.00e-05 | 0.2943 | 8.4505e-04 |
| 1 | 800 | old | 5.00e-05 | 1.00e-04 | 0.3180 | 8.2946e-04 |
| 1 | 800 | new | 0.00e+00 | 1.50e-04 | 0.2969 | 9.1307e-04 |
| 10 | 25 | old | 2.45e-03 | 2.21e-01 | 1.1066 | 4.3007e-03 |
| 10 | 25 | new | 1.50e-03 | 1.97e-01 | 1.0606 | 3.8165e-03 |
| 10 | 100 | old | 2.60e-03 | 3.11e-01 | 1.1280 | 3.6902e-03 |
| 10 | 100 | new | 2.10e-03 | 3.02e-01 | 1.0880 | 3.3445e-03 |
| 10 | 200 | old | 3.45e-03 | 3.24e-01 | 1.2661 | 3.4155e-03 |
| 10 | 200 | new | 2.00e-03 | 3.22e-01 | 1.0665 | 3.3218e-03 |
| 10 | 400 | old | 4.40e-03 | 3.31e-01 | 1.3345 | 3.4226e-03 |
| 10 | 400 | new | 2.45e-03 | 3.28e-01 | 1.0917 | 3.2805e-03 |
| 10 | 800 | old | 4.25e-03 | 3.36e-01 | 1.3098 | 3.3773e-03 |
| 10 | 800 | new | 2.85e-03 | 3.35e-01 | 1.1933 | 3.2515e-03 |

## (b) Default grid (Nz=100): per-cell sigma comparison

| zs | w-mean sig_old | w-mean sig_new | clust_std old | clust_std new |
|--:|--:|--:|--:|--:|
| 1 | 1.508 | 0.677 | 0.0024 | 0.0084 |
| 10 | 1.779 | 0.408 | 0.0181 | 0.0297 |

## Analytic continuum-limit check: linearized clustering std vs Nz

(std of the bias-modulated mean explicit kappa, lambda ~ 1+g; old = iid cells, new = correlated field. A grid-decoupled model must be flat in Nz.)

| zs | Nz | clust_std old | clust_std new | p99 sig_b old | p99 sig_b new |
|--:|--:|--:|--:|--:|--:|
| 1 | 25 | 0.0033 | 0.0082 | 3.00 | 1.01 |
| 1 | 50 | 0.0029 | 0.0084 | 3.18 | 1.32 |
| 1 | 100 | 0.0024 | 0.0084 | 3.36 | 1.60 |
| 1 | 200 | 0.0019 | 0.0082 | 3.57 | 1.84 |
| 1 | 400 | 0.0015 | 0.0079 | 3.77 | 1.97 |
| 10 | 25 | 0.0276 | 0.0280 | 2.59 | 0.65 |
| 10 | 50 | 0.0227 | 0.0292 | 2.78 | 0.87 |
| 10 | 100 | 0.0181 | 0.0297 | 2.99 | 1.18 |
| 10 | 200 | 0.0142 | 0.0297 | 3.22 | 1.53 |
| 10 | 400 | 0.0110 | 0.0293 | 3.47 | 1.86 |

## (c) kappa_thr sweep (Nz=100): w-p99 per-cell sigma_b

| zs | kappa_thr | p99 sig_b old | p99 sig_b new |
|--:|--:|--:|--:|
| 1 | 1e-04 | 3.30 | 1.56 |
| 1 | 3e-04 | 3.59 | 1.72 |
| 1 | 1e-03 | 3.93 | 1.76 |
| 1 | 3e-03 | 4.22 | 1.80 |
| 1 | 1e-02 | 5.06 | 1.83 |
| 10 | 1e-04 | 2.30 | 0.92 |
| 10 | 3e-04 | 2.56 | 1.06 |
| 10 | 1e-03 | 2.89 | 1.17 |
| 10 | 3e-03 | 3.26 | 1.20 |
| 10 | 1e-02 | 3.82 | 1.25 |
