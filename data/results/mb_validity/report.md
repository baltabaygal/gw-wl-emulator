# M_b/M validity map (bias layer, halo only)

Claim under test: M_b (tube-segment mass entering sigma_b) is 'much larger than the typical lens masses'. Default grid Nz=NM=100, fixed-<N>=100 rule, subhalos off.

gwlensing spot-check zs=1: kthr C++ 1.277748e-04 vs port 1.277748e-04 (rel 0.0e+00)


## Nz = 25

| z_s | kappa_thr | typ. lens M (w_bias-median) | M_b/M w_bias q10/q50/q90 | M_b/M w_enc q50 | w_bias share ratio<1 | <10 | <100 | sigma-clamped cells |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 5.98e-06 | 1.1e+12 | 27 / 52 / 81 | 49 | 0.0% | 1.1% | 99.4% | 0.0% |
| 1 | 1.13e-04 | 1.1e+12 | 20 / 34 / 47 | 32 | 0.0% | 1.4% | 100.0% | 0.0% |
| 5 | 8.33e-04 | 2.6e+11 | 6.5 / 13 / 18 | 11 | 0.1% | 25.8% | 100.0% | 0.0% |
| 10 | 1.27e-03 | 1.1e+11 | 4.9 / 9.4 / 14 | 8.2 | 0.2% | 54.7% | 100.0% | 0.0% |

## Nz = 100

| z_s | kappa_thr | typ. lens M (w_bias-median) | M_b/M w_bias q10/q50/q90 | M_b/M w_enc q50 | w_bias share ratio<1 | <10 | <100 | sigma-clamped cells |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 6.57e-06 | 2.8e+12 | 7.2 / 14 / 20 | 13 | 0.2% | 22.8% | 100.0% | 0.0% |
| 1 | 1.28e-04 | 2.3e+12 | 4.6 / 8.3 / 11 | 7.1 | 0.2% | 73.8% | 100.0% | 0.0% |
| 5 | 9.04e-04 | 4.4e+11 | 1.6 / 3.1 / 4.3 | 2.6 | 2.7% | 100.0% | 100.0% | 0.0% |
| 10 | 1.37e-03 | 1.7e+11 | 1.1 / 2.2 / 3.2 | 1.8 | 7.7% | 100.0% | 100.0% | 0.2% |

Comoving-corrected M_b (x(1+z)^2), zs=1: w_bias share ratio<1 = 0.2%, <10 = 12.3%.


## Nz = 400

| z_s | kappa_thr | typ. lens M (w_bias-median) | M_b/M w_bias q10/q50/q90 | M_b/M w_enc q50 | w_bias share ratio<1 | <10 | <100 | sigma-clamped cells |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 6.75e-06 | 5.7e+12 | 1.9 / 3.6 / 5.2 | 3.1 | 2.5% | 100.0% | 100.0% | 0.0% |
| 1 | 1.31e-04 | 3.6e+12 | 1.2 / 2.1 / 2.9 | 1.7 | 6.1% | 100.0% | 100.0% | 0.0% |
| 5 | 9.28e-04 | 5.6e+11 | 0.41 / 0.77 / 1.1 | 0.62 | 83.2% | 100.0% | 100.0% | 0.5% |
| 10 | 1.41e-03 | 2.2e+11 | 0.28 / 0.56 / 0.8 | 0.45 | 99.9% | 100.0% | 100.0% | 1.0% |

Figure: plots/mb_validity_map.png (default grid). White contour = cells jointly carrying 90% of the bias-layer clustering-variance weight; black contour = M_b/M = 1.


## Verdict

The claim is quantitative and testable, and it does NOT hold as stated at the default grid for high z_s: weighted by the cells' actual contribution to the bias layer's clustering variance, the median M_b/M is ~14 (z_s=0.2) and ~8 (z_s=1) — 'larger', not 'much larger' — and drops to ~3 (z_s=5) and ~2 (z_s=10), where the separate-universe/PBS premise (environment mode >> halo mass) is simply not satisfied. Essentially zero weight sits at M_b/M > 100 at any z_s. Under grid refinement the claim inverts: at Nz=400, 83-100% of the weight has M_b < M at z_s >= 5 (sigma(M_b) is then the halo's own formation variance or larger, and the Mmin clamp starts engaging) — this is the '(Delta z chosen so that ...)' validity condition being exited by refinement, now quantified. The (1+z)^2 physical-vs-comoving rmax wart moves the z_s=1 numbers by less than one grade (share below 10 goes 74% -> 12%) and does not change the high-z_s conclusion.
