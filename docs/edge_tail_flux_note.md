# Edge, tail exponent, and flux calibration of the production emulator (2026-07-11)

One-day study of three proposed improvements to the production smooth model
(`gw-wl-emulator-ar/ml/autoresearch/smooth_model.py`). Full experiment log with all
tables: `-ar` REPORT.md (two 2026-07-11 sections) + RESEARCH_tails.md (tail section).
Reference data: 15 legacy panels x 400k fresh raw samples (seeds 7000+17j, -ar build).

## 1. Low-mu edge: fitted quantile is oracle-level; Dyer-Roeder is NOT the sim's edge

- **Oracle test:** replacing the ridge-fitted edge (0.1% lnmu quantile regression,
  resid 0.008) with each panel's TRUE empirical quantile moves KL by <1%. Edge
  localization is solved; the z=1 KL outliers live in the flow body, not the edge.
- **Dyer-Roeder (empty-beam ODE, alpha=0):** lnmu_min = -0.127 / -1.65 at zs=1/8 vs the
  simulator's observed q0.1% edge -0.111 / -0.478 and its own hard floor
  -2 ln(1+meankappa) = -0.180 / -0.833. **Never substitute DR for the fitted edge.**
  Why the sim edge differs (cpp/lensing.cpp:489-506):
  1. the subtracted mean `meankappa` is the mean of EXPLICIT above-threshold
     halos+filaments only -> sigma8-dependent via the HMF (observed edge moves ~2x
     across the sigma8 panels at fixed z; DR is sigma8-free by construction);
  2. with <N>=100 explicit halos/LOS, P(empty beam) = e^-100 ~ 0 -> the observed edge
     is a soft convolution edge well above the hard floor;
  3. the sub-threshold Gaussian background is zero-mean and unbounded below;
  4. kappa>1 core-crossing rays (detA >> 1, demagnified) are KEPT as valid samples down
     to lnmu ~ -10 (detA <= 0 dropped) -> a real sparse population below any "edge".
  The DR-vs-sim offset is a quantified statement of how far the halo-model construction
  sits from continuum theory — paper material, not a model component.
- **Window sharpness:** EDGE_W 0.02 -> 0.01 shipped (z=1 KL -4%, PIT~0 spike trimmed,
  no high-z cost). 0.005 OVER-cuts at z>=5 (P(PIT<.005) -> 0.65x nominal); a hard
  Heaviside would zero the genuine kappa>1 population. Do not sharpen further.
- `sample_lensing_raw_ml` returns PRE-mean-subtraction kappa (mean of returned kappa
  = meankappa) — useful for future edge/floor diagnostics.

## 2. Tail exponent: mu^-2 (image plane) == mu^-3 (source plane); ours is mu^-2

- Source plane (fold-caustic universality): p_S(mu) ~ mu^-3. Image plane picks up one
  power of mu (dOmega_I = mu dOmega_S): p_I(mu) ~ mu^-2.
- The simulator samples random observer-sky directions => IMAGE plane. Confirmations:
  Hill alpha = 1.95-2.03 (10M samples), <1/mu> = 1 sum rule (image-plane law; source
  plane would obey <mu> = 1), kappa>1 rays kept.
- "Try both" race: Poisson LL on the 10M tail counts (mu >= 8): mu^-2 beats mu^-3 by
  1400-6800 nats at every z; mu^-3 under-predicts survival 2x at mu=15, 10x at mu=120.
- **What ML fits: the image-plane dP/dmu with the mu^-2 tail** (POT survival slope -> 1
  in lnmu). Source-plane comparisons (e.g. ACE) get mu^-3 automatically via P_S ~ P_I/mu.
- Gotcha: survival-slope fits on a mu-truncated reference (histogram stops at 300) are
  biased steep — the missing >300 mass deflates the top of the fit window. Use
  density-based fits.

## 3. Flux calibration (<1/mu> sum rule) — SHIPPED, largest accuracy win since exp24

- The z=1 "shoulder" outlier was largely a +0.006 lnmu MEAN bias (13% of the narrow
  PDF's width). Enforcing the flux sum rule corrects the mean for free.
- Shipped: p(x) -> p(x - delta), delta(ctx) = ln<1/mu>_model - ln F_trim(ctx), where
  F_trim = the sim's SUPPORT-RESTRICTED flux (above the edge window), fit by
  `fit_flux_target.py` -> `cache/flux_target_fit.json` (ridge on tail_features,
  330 configs, resid 0.0007 in lnF; data = lowz_aug + cache/flux_grid.npz).
- **Do NOT target raw <1/mu> = 1**: the sim's raw sample mean is corrupted by rare
  kappa>1 rays (stress z=1: raw <1/mu> = 1.08 from a handful of 1/mu ~ 1e4 events;
  matching it gives KL 0.50), and exact-1 over-shifts at z >= 3.5 where the genuine
  within-support flux is 1.005-1.02.
- Results (end-to-end through load_smooth_fn, fine 300-bin equal-mass KL):
  mean panel KL 0.0091 -> 0.0035, median 0.0051 -> 0.0034; z=1 outliers
  0.0367 -> 0.0064, 0.0255 -> 0.0065; KS <= 0.021 everywhere; PIT edge spike <= 1.4x.
  Posterior recovery (16x1000-event mocks, same seeds as the 2026-07-03 baseline):
  h bias -0.46+-0.20 -> **-0.20+-0.22 sigma_post** (zero-consistent);
  Om +0.29+-0.17 -> +0.34+-0.21 (unchanged: not mean-driven). Control clean.
- Remaining known residual: z=1 WIDTH mismatch (model sigma 4-8% off) -> body-retrain
  territory. Do NOT attempt an affine variance calibration in lnmu: rescaling corrupts
  the mu^-2 tail exponent.
- 6d hook: `flux_target_fit_6d.json` lazy-loaded like the edge/tail caches. When
  retraining (6d / subhalo-corrected data), rerun fit_flux_target.py on the new
  datasets — add it to the retraining recipe alongside prepare_fix / fit_tail_amplitude.
