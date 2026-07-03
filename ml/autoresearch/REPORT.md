# Autoresearch run: high-z far-tail fix for the NSF emulator

Run on branch `autoresearch/jun16` (worktree `gw-wl-emulator-ar`), env: conda `test`
(py3.12, torch 2.12, zuko 1.6), device MPS. Method adapted from karpathy/autoresearch:
edit one hackable trainer, train, score against a fixed simulator metric, keep/revert.

## Problem

Production flow `conditional_nsf_backend_current.pt` under-covers the strong-lensing
far tail at high z: survival `P(mu>t)` collapses to 0 at `mu ~ 3.4` for every z, while
the simulator keeps mass to `mu ~ 5-8`. The far tail **is in the training data**, so
this is a model/representation problem.

## Root cause (verified)

zuko RQS spline `bound=5.0` + `lnmu_std~0.16-0.24` ⇒ the spline only covers `mu <~ 3.4`.
Beyond it the transform is identity into a **standard-normal base**, whose Gaussian tail
decays far faster than the true `dP/dmu ~ mu^-3`. Direct check: at standardized lnmu=6,
Normal base log-density = -16.3 (collapsed) vs StudentT(df=4) = -6.8 (heavy tail kept).

## Metric (fixed, `prepare_ar.py`)

- **TLSE** (primary): mean `|log10 S_nsf(mu>t) - log10 S_sim(mu>t)|` over high-z eval
  points (central + high-structure corner, z=2,3.5,5,8) and t=2,3,4,5.
- **BodyNLL** (guard): flow NLL on simulator samples in the science window.
- Ground truth = simulator survivals cached once (`cache/groundtruth.npz`).

## Experiment log (`results.tsv`)

| exp | change | TLSE | BodyNLL | status |
|-----|--------|------|---------|--------|
| baseline | bins8 bound5 normal | 2.805 | -0.209 | keep |
| 1 | bound 5→8 | 1.894 | -0.338 | keep |
| 2 | **StudentT base df=4** | 0.539 | -0.529 | keep |
| 3 | bound 8→12 | 0.293 | -0.526 | keep |
| 4 | bins 8→16 | 0.200 | -0.534 | keep |
| 5 | bins 16→24 | 0.259 | -0.535 | discard (tail undertrained) |
| 6 | tail_weight=10 | 0.368 | -0.509 | discard (overshoot) |
| 7 | tail_weight=3 | 0.143 | -0.531 | keep |
| 8 | **tail_weight=2** | **0.046** | **-0.536** | **WINNER** |

TLSE 2.805 → 0.046 (~60×). Body fidelity and val NLL also improved (no tradeoff).

## Winner config

`bins=16, bound=12, transforms=6, hidden=128, base=StudentT(df=4), tail_weight=2`
(upweight lnmu>1.5 ×3, with weighted validation selection). Saved:
`ml/autoresearch/models/flow_ar.pt`.

## Held-out validation (`validate_ar.py`, points NOT in the metric panel)

Cosmologies `(0.60,0.22,0.70)` and `(0.74,0.39,1.03)`, z = 1.5, 2.5, 4.0, 6.5:

- **Production mean TLSE = 1.643** (survival = 0.00000 for all mu>4).
- **Winner mean TLSE = 0.225** (7.3× better; tracks the simulator tail).

See `validation_dpdmu.png`: production cuts off at mu≈3; winner follows the simulator
to mu≈6-8 at z = 2, 3.5, 5, 8 (high-structure corner).

## Caveats / next levers

- Held-out TLSE (0.225) > panel TLSE (0.046): some panel-tuning; still a large win.
- `tail_weight=2` slightly *overshoots* the thinnest tails (low-structure, low z).
  A z/structure-dependent weight, or `tail_weight≈1.5`, could trade that off.
- Not promoted to production (isolation from the other agent). To promote: retrain on
  the production dataset/stats and copy into `data/models/`.

---

## FINAL STATUS (supersedes the above; see RESEARCH_tails.md)

The TLSE-only "winner" above (exp8) was **smooth-metric-blind**: its dP/dmu had a
reweighting bump at mu~4.5 and spline wiggles. After adding a tail-SHAPE metric and
researching the literature, the final model changed.

**Best model = body-flow + conditional power-law tail splice** (R1).
- Body: plain Normal-base flow (bound=16, bins=20, pure NLL) — faithful body, no heavy base.
- Tail (mu>2): analytic mu^-3.4 power law, amplitude = flow's survival at mu=2 (conditional,
  data-rich), redistributed. Smooth by construction.
- Impl: `splice_tail.py::make_spliced_log_prob_fn(body_fn, mu_u=2.0, alpha=3.4, anchor="survival")`,
  body checkpoint `models/flow_ar.pt`. Tagged `best-splice-model`.
- In-panel TailShape 0.081, Rough 0.002; **held-out mean TLSE 0.181 vs production 1.64 (9x)**.

**L1 downstream check (`l1_check.py`):** sigma_dm tail-insensitive (~10% even for the cutoff);
sigma_mu fixed to ~4% and logL bias ~0.01 by the splice; rare-event rate P(mu>5) median -5%
on held-out (production misses 100%).

**Data-first experiment — NEGATIVE result:** a 10x-denser tail dataset (`datasets_tailrich`,
100k/config) did NOT improve held-out (original 0.181, tail-rich 0.231, merged 0.217). For a
conditional density, config coverage > samples/config; and the splice's survival-at-mu=2 anchor
is already well-estimated with 10k/config. Held-out alpha sweep: alpha=3.4 best for P(mu>5).
=> data lever exhausted; original-data splice remains best.

**Remaining lever (optional, marginal):** conditional GPD — per-cosmology slope/scale fit to
the (now-available) dense exceedances. Expected gain small (alpha=3.4 already -5% median P(mu>5)).

**Recommendation:** strong stopping point. Productionize the splice (retrain body on full data,
bundle the wrapper, run full validation), or stop. Not promoted to shared production.

## 2026-07-02 — smoothness fix (exp18): production = flow_smooth + tanh blend

**Problem:** flow_ar.pt PDF not smooth — local log-log slope oscillated between -7 and +2
over mu ~ 1-13 (knot noise from 6 transforms x 20 RQS bins), kink at the spline domain
edge mu = e^(16*lnmu_std) = 13.35, spurious bump at mu ~ 0.5-0.8, and z=2 tail overshot
5x (slope penalty pinned all z to mu^-2). Diagnostic: `smoothness_refresh.png`.

**Fix (two orthogonal changes):**
1. `train_smooth.py` -> `models/flow_smooth.pt`: 3 transforms x 10 bins (4x fewer knots),
   StudentT df=2 base, dense curvature penalty (2nd-diff^2 of log p, 48-pt grid over
   mu 1.4-13.5, weight 30), NO slope pinning. 60 epochs, ~3 min on MPS, best ValNLL 0.510.
2. `smooth_model.py`: C-inf tanh blend (width 0.35 in lnmu) from the flow body to the
   analytic image-plane mu^-2 power law, density-anchored at mu_c=8 — after the data-rich
   range, before the bound kink and the StudentT lnmu-power asymptotics. mu_c scan:
   3.0 is WRONG (anchors on the shoulder, SCORE 0.238); 8/10/12 all ~ flow alone.

**Harness (prepare_ar.evaluate) + DenseRough (mean |d2 log p| / dlnmu^2, mu in [1,100]):**

| candidate          | SCORE  | Rough  | TLSE   | BodyNLL | DenseRough |
|--------------------|--------|--------|--------|---------|------------|
| flow_ar (old prod) | 0.1386 | 0.0787 | 0.2377 | -0.5238 | 28.6       |
| flow_smooth        | 0.1051 | 0.0110 | 0.2243 | -0.5215 | 4.1        |
| **flow_smooth+blend (SHIPPED)** | **0.1065** | **0.0114** | **0.1922** | -0.5231 | **3.8** |

Every metric improves or holds; smoothness ~7x better on the harness Rough and ~7.5x on
DenseRough. Verification figure: `smoothness_compare.png` (PDF + local-slope panels,
old vs new, all clean). Production entry point:
`from ml.autoresearch.smooth_model import load_smooth_fn`.

**Residual known imperfections:** small slope wiggle remains at z=8 mu ~ 5-11 (low
amplitude); z=2 far tail (mu > 30) still slightly above sim (true z=2 slope is steeper
than the universal mu^-2; a z-dependent alpha would fix it — deferred, ACE L1 argument
says downstream impact is tiny).

## 2026-07-02 (later) — exp19-22: low-mu physical cutoff + low-z accuracy (user review)

User review of exp18 across parameter space: (1) PDF does not go to 0 at low mu —
physically it must (empty-beam demagnification bound); (2) tail improved but shoulder
(mu ~ 1.5-6) overshoots at low z / low structure.

**Low-mu cutoff (SHIPPED, smooth_model.py):** the lnmu lower edge is a near-deterministic
function of (z,theta): ridge fit of the per-config 0.1% quantile of datasets_logz_1k,
resid std 0.008 (prepare_fix.py -> cache/edge_alpha_fit.json). Production density is
multiplied by a C-inf sigmoid window at (fit - 0.05), width 0.02 in lnmu, renormalized.
Harness cost: none (SCORE 0.1069 vs 0.1065). Also measured the deep-tail slope from the
clean 10M ref: alpha ~ 2.0 at ALL z (2.0/3.5/5/8 -> 1.95/2.00/2.03/2.03), so the mu^-2
blend is correct; the apparent z=2 slope problem was shoulder amplitude, not slope.

**Shoulder investigation (exp19-22, extended 15-panel shapeMAE in param_space_check.py):**
- exp19 d3-penalty: falsified "penalty flattens the knee" — worse everywhere.
- Dataset vs fresh sim at identical configs: consistent — it IS data sparsity
  (single-digit shoulder counts per config at z~1).
- exp20 low-z augmentation (150 configs x 40k, z in [0.3,2.5], gen_lowz_data.py):
  harness SCORE 0.107->0.100, TLSE 0.193->0.172 (both best-ever); z=1 shoulder barely
  moved -> at z~1 the penalty acts where data is absent even after 6M extra samples.
- exp21 hard z>1.6 penalty mask: z=1 MAE 0.31->0.23 but high-z 0.10->0.13 (trade-off).
- exp22 soft z-ramp: dominated. DECISION: ship exp20 (primary metrics), keep exp21
  checkpoint (flow_smooth_zmask.pt) for low-z-focused work.

**Production = flow_smooth_lowz.pt + tanh blend (mu_c=8, alpha=2) + edge cutoff**, via
`smooth_model.load_smooth_fn()`. Verification: param_space_check.png (3 cosmologies x
5 z, fresh 400k sim each; extended shapeMAE mean 0.159).

**Known residual:** z~1 shoulder (mu 1.5-6, dP/dlnmu <~ 1e-2) still ~2-3x high
(panel MAE ~0.3). Next levers if needed: stratified shoulder oversampling at low z with
importance weights, joint with a per-z penalty weight tuned on the extended panel.

## 2026-07-02 (round 3) — exp23: conditional POT tail = the tail-amplitude fix

**Root cause quantified:** the flow's density at the old blend anchor (mu=8) is nearly
context-INDEPENDENT (~1.2e-3 for all 15 panels) while the truth spans 5e-5..7e-3
(x140): anchor inflation x23 (center z=1), x13 (low-struct z=2), deficit x0.6
(high-struct z>=5). The flow cannot learn tail amplitude from ~10 tail samples/config;
the density-anchored blend propagated that error to the whole tail.

**Fix (EVT peaks-over-threshold with covariates — see RESEARCH_tails.md round 2):**
`fit_tail_amplitude.py` fits S3(ctx)=P(mu>3), S8(ctx)=P(mu>8) by ridge Poisson
regression on per-config exceedance COUNTS pooled over 637 configs (14966 / 2941
exceedances) — validated out-of-sample against fresh 400k sims: within +-20-40%
everywhere (vs x23/x0.6 before). `smooth_model.py` builds the tail from the fitted
survivals: transition exponent k(ctx)=ln(S3/S8)/ln(8/3) over mu in [3,8], softplus-
smoothed to the asymptotic mu^-2 beyond 8, C-inf everywhere; tanh blend with the flow
body at mu_c=2.4 (scanned 2.2-3.5); edge cutoff unchanged.

**Result (body flow_smooth_lowz.pt unchanged — no retraining needed):**
| | exp20 (flow-anchored) | exp23 POT (SHIPPED) |
|---|---|---|
| harness SCORE | 0.1002 | **0.0611** (all-time best; old record 0.0806) |
| TLSE | 0.1720 | **0.0449** |
| extMAE (15 panels) | 0.1589 | **0.0990** |
| z=1 MAE | 0.310 | **0.200** |
| Rough / DenseRough | 0.0122 / 3.4 | 0.0104 / 3.1 |

Remaining known residuals: z=1 shoulder mu~1.5-2.5 (MAE ~0.2-0.28 at center/low-struct;
flow body territory below the blend) and the S8 fit is ~1.3-1.9x high in the extreme
low-structure corner (S~1e-5 regime). Next lever if ever needed: more counting sims for
the amplitude fit at low structure + low z (pure counts, cheap).

## 2026-07-02 (round 4) — exp24: counting runs + u=2 extension

Two targeted upgrades on exp23 (user pointed at low-z and low-structure/low-z panels):
1. `gen_tail_counts.py`: 30 counting-only runs (1.5M samples each; 6 z in [0.4,2.5] x
   5 low/central-structure thetas) added to the Poisson amplitude fit — low-structure
   S3 validation ratios 1.2-1.4 -> 0.97-1.07.
2. Third threshold u=2 (fit_tail_amplitude.py THRESHOLDS=(2,3,8)); tail is now a
   3-segment smoothed power law (slopes k1 [2,3], k2 [3,8], 1 beyond), anchored
   exactly at S3 (anchor scan: "u3" dominates "u2" — the S2 fit is ~1.2x high on the
   stress row). Blend at mu_c=1.7 (scan in smooth_model.py comments) — the calibrated
   tail now covers the whole former z~1 shoulder problem region.

| | exp23 | exp24 (SHIPPED) |
|---|---|---|
| harness SCORE | 0.0611 | 0.0507 |
| TLSE | 0.0449 | **0.0374** |
| extMAE (15 panels) | 0.0990 | **0.0613** |
| z=1 MAE | 0.200 | **0.095** |

Day summary (flow_ar -> exp24): SCORE 0.139->0.051, extended-panel MAE ~0.3+ -> 0.061,
z=1 0.31+ -> 0.095, smooth everywhere, physical low-mu cutoff. Worst remaining panel:
low-structure z=1 (MAE 0.164; S~1e-5 regime, sim reference itself is count-noise there).

## 2026-07-03 — validation battery (ACE-parity + calibration + end-to-end)

1. **KL divergence** (`validate_kl.py`, vs fresh 400k sims, bias-corrected):
   median source-plane KL on the ACE support (mu <= 6) = **0.0073** — matching
   ACE-Lensing's published 0.007 (arXiv:2512.01607) — and unchanged (0.0061) when the
   window extends to mu=100, where ACE truncates. Caveat: each measures fidelity to its
   own pipeline's ground truth. Outliers: z=1 panels (low-structure 0.44, high 0.054) —
   KL is harsh on very narrow PDFs (sigma_lnmu ~ 0.05).
2. **Moments + flux conservation**: sigma/skew of lnmu track the sim; mean lnmu
   +0.005-0.015 high; <1/mu> = 0.992-1.013 (physical value 1) with NO enforcement.
3. **PIT calibration** (`validate_pit.py`, fresh seeds): KS 3-5% typical, 8-10% at z=1.
   Deviations: small PIT~0 spike (left edge slightly aggressive) + mild rightward mean
   shift at z=1; tails clean.
4. **Posterior recovery** (`validate_posterior.py`, the decisive test): 16 independent
   1000-event mock catalogs (z in [0.3,4], 7% distance noise), lensed by the SIMULATOR,
   inferred on an (h, Om) grid with the model likelihood; control = catalogs lensed by
   the model itself. Control mean bias: h +0.24+-0.28 sigma, Om -0.14+-0.28 (zero ✓).
   **Real test: h -0.46+-0.20 sigma_post, Om +0.29+-0.17 sigma_post per 1000-event
   catalog.** Model systematics stay below half the statistical error at the
   1000-event scale. (First run had a harness bug — bin-center distances — caught by
   the control at -6 sigma; fixed to exact per-event distances.)

**Conclusion:** fit for purpose for catalogs up to O(1000) events. The z~1 PDF outliers
do not translate into meaningful cosmology bias at this scale. For larger catalogs
(bias grows ~sqrt(N) vs stats): first lever = enforce <1/mu> = 1 exactly via an
ACE-style a-posteriori lnmu shift (delta = -ln<1/mu>), then revisit PIT left edge.
