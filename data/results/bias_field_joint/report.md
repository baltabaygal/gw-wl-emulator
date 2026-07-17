# Joint-field sizing: clustered weak background

Weak arm a_w = Dg * (bias-weighted first moment of sub-threshold annuli + sub-Mmin continuum), computed DIRECTLY (the sum-rule subtraction is diagnostic-only: untruncated-NFW overshoot). Two correlation conventions bracket the transverse window: PENCIL (R=0.5 kpc, keeps all aliased 3D power - hard upper bound) and WINDOWED (explicit: counting cylinder; weak: disc at the kappa-weighted RMS beam radius, R_L(M) floor - realistic). Denominator Var_tot = Sigma barN k2bar + sigma_W^2 (analytic, unclipped); meas.body is the full-C++ CLIPPED Var(|kappa|<0.5) at Nz=100 (definitions differ at high zs where the tail dominates).

## 1. Variance budget

| zs | k_thr | sigma_W | Var_e_pois | Var_w_pois | S_ee pen/win | S_ww pen/win | 2S_ew pen/win | rho_ew | Var_tot | meas.body |
|--:|--:|--:|--:|--:|:--|:--|:--|--:|--:|--:|
| 0.2 | 6.567e-06 | 0.0001 | 1.271e-05 | 4.418e-09 | 4.05e-06 / 9.90e-07 | 4.27e-06 / 3.72e-07 | 8.32e-06 / 1.21e-06 | 0.999 | 1.272e-05 | 2.323e-05 |
| 1 | 1.278e-04 | 0.0013 | 7.425e-04 | 1.762e-06 | 1.66e-04 / 7.09e-05 | 3.26e-04 / 5.90e-05 | 4.64e-04 / 1.29e-04 | 0.995 | 7.442e-04 | 1.111e-03 |
| 5 | 9.038e-04 | 0.0100 | 7.432e-03 | 9.906e-05 | 1.10e-03 / 6.10e-04 | 4.69e-03 / 1.74e-03 | 4.49e-03 / 2.01e-03 | 0.973 | 7.531e-03 | 4.078e-03 |
| 10 | 1.373e-03 | 0.0157 | 1.158e-02 | 2.461e-04 | 1.50e-03 / 8.83e-04 | 8.43e-03 / 3.86e-03 | 6.94e-03 / 3.50e-03 | 0.949 | 1.182e-02 | 2.624e-03 |

## 2. Headline ratios (windowed = realistic; pencil in parens = upper bracket)

| zs | S_ww/Var_w_pois | S_ww/Var_tot | 2S_ew/Var_tot | (S_ww+2S_ew)/Var_tot | S_ee/Var_tot | new/meas.body |
|--:|:--|:--|:--|:--|:--|:--|
| 0.2 | 84.211 (967.6) | 2.92e-02 (3.36e-01) | 9.54e-02 (6.54e-01) | 1.25e-01 (9.90e-01) | 0.0778 (0.3184) | 6.82e-02 |
| 1 | 33.513 (185.1) | 7.93e-02 (4.38e-01) | 1.73e-01 (6.24e-01) | 2.52e-01 (1.06e+00) | 0.0953 (0.2229) | 1.69e-01 |
| 5 | 17.584 (47.4) | 2.31e-01 (6.23e-01) | 2.66e-01 (5.96e-01) | 4.98e-01 (1.22e+00) | 0.0811 (0.1460) | 9.19e-01 |
| 10 | 15.684 (34.3) | 3.26e-01 (7.13e-01) | 2.96e-01 (5.87e-01) | 6.23e-01 (1.30e+00) | 0.0747 (0.1268) | 2.81e+00 |

## 3. Mass bookkeeping

sum-rule closure: I_b(Mlo) -> 1 as Mlo -> 0 for a PBS-consistent (HMF, b) pair. f_exp = kappa-weighted explicit capture fraction; > 1 exposes the untruncated-NFW overshoot (projected mass within rmax exceeds M200 when rmax >~ r200), which is why the subtraction route for a_w is diagnostic-only. overshoot = summed negative part of the subtraction a_w.

| z | I_m(1e-3) | I_b075: 1e3 / 1e0 / 1e-3 | I_b080(1e-3) | I_b075_low(<Mmin) | f_exp | overshoot |
|--:|--:|:--|--:|--:|--:|--:|
| 0.2 | 0.7457 | 0.7395 / 0.7693 / 0.7922 | 0.8130 | 0.1165 | 1.495 | 3.61e-03 |
| 1 | 0.6962 | 0.6936 / 0.7300 / 0.7577 | 0.7786 | 0.1432 | 1.034 | 3.65e-02 |
| 5 | 0.5002 | 0.5051 / 0.5735 / 0.6237 | 0.6446 | 0.2779 | 0.570 | 8.12e-02 |
| 10 | 0.3277 | 0.3150 / 0.4205 / 0.4978 | 0.5177 | 0.3894 | 0.415 | 7.64e-02 |
