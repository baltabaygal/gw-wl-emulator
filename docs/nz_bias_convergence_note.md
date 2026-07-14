# The bias layer couples a physical scale to the numerical grid — findings note (2026-07-14)

**One-sentence finding:** the MC engine's halo-clustering (BIAS) layer defines
its physical environment scale *through* the numerical integration grid — the
z-bin width Δz and the tube radius r_max(κ_thr) together set the region whose
density fluctuation σ(M_b) modulates halo counts — so two knobs that are
supposed to be pure numerics (Nz, κ_thr) silently change the physical
clustering model, and "convergence" in them is ill-posed for any observable
the bias tail feeds. This requires a fix (options below), or an explicit
declaration that the grid scale is part of the model.

Investigation date 2026-07-13; note written 2026-07-14. Companion to
`docs/convergence_mmin_nz_note.md` (the Nz JSD study whose *body*-level
verdict — Nz=100 converged in P(lnμ) — is unaffected).

## 1. The phenomenon

Raw-κ MC sweep (`scripts/convergence/vark_vs_nz.py`, 4 seeds × 50k, Nz =
25…1600, seed namespace 8e8, full model, fixed-⟨N⟩ rule):

- Variance convergence in Nz is **support-dependent**. On a fixed support
  |κ|<1 it converges gently from below (default Nz=100 at 0.91–0.96 of the
  Nz=1600 value); the |κ|<0.5 body converges from *above* (coarse grids
  overweight the shoulder). Any window that admits the strong tail does NOT
  converge.
- The strong-lensing population grows unsaturated: f(κ>1) at z_s=10 rises
  0.45% → 1.30% over Nz = 25 → 1600 (z_s=5: 0.09% → 0.32%); q99.99(κ)
  runs 2.05 → 8.8. The tail *extends* (growth is scale-dependent: q99 +10%,
  q99.9 +52%, q99.99 ×4.3), it does not renormalize.
- P(lnμ) (JSD, body-weighted) stays at the measurement floor at Nz=100
  throughout — no contradiction: variance is tail-weighted, JSD is not.

## 2. The evidence chain (what was ruled out, what was convicted)

1. **κ_thr split position — ruled out.** Frozen arm
   (`vark_vs_nz_frozen.py`, kappathr_flat pinned to the default's Nz=100
   value, namespace 8.1e8): growth identical to the rule arm
   (z=10 f(κ>1) = 7.1/9.7/13.6e-3 at Nz=100/400/1600 vs 7.0/9.5/13.0e-3).
2. **Strong-encounter count quadrature — ruled out.** Analytic
   `get_expected_halo_count(zs, κ_c, Nz)` for κ_c ∈ {0.5, 1} converges
   first-order in 1/Nz (0.967 at Nz=100 → 0.999 at 1600, all z_s). Also:
   ⟨N⟩(κ_c=1) ≈ 9e-5 at z=10 is ~100× below the measured f(κ_total>1) ⇒
   the κ>1 rays are COMPOUND (stacked moderate halos), pointing at
   clustering/multiplicity, i.e. the bias layer.
3. **Bernoulli shortcut (`λ·barN < 0.2` branch, lensing.cpp:572) —
   exonerated.** Exactly mean-preserving; only removes same-cell
   multiplicity (rate Σ m²/2, orders below f(κ>1), shrinking ∝1/Nz); bias
   bursts route to the exact Poisson branch anyway; and the bias-off arm
   (same shortcut) is flat.
4. **BIAS layer — convicted.** Bias-off arm (`vark_vs_nz_nobias.py`,
   namespace 8.2e8): the growth vanishes (z=10 f(κ>1) =
   5.05/5.34/5.47e-3 at Nz=100/400/1600, flat within errors; z=5 flat at
   ~1.0e-3). Quantitative clincher: bias-off Var(|κ|<1) ratio at Nz=100 is
   0.966 vs the analytic quadrature prediction 0.967 — with bias off, ALL
   remaining Nz-dependence is ordinary O(1/Nz) shell quadrature, converged
   by Nz≈400. Deterministic cross-check: extended quadrature arm
   (`data/results/convergence_analytic/sigW_nz_extended.npz`, Nz→6400):
   σ_W and κ_thr(rule) converge cleanly to true limits (σ_W² at 0.963–0.971
   of the limit at Nz=100, 0.9997 by 3200).

Figures: `plots/nz_tail_mechanism.png` (the three-arm proof),
`plots/vark_total_vs_nz.png` (support decomposition),
`plots/vark_vs_knobs.png` panel 2 (σ_W's true limit).

## 3. The mechanism — a physical knob hiding inside a numerical one

`lensing.cpp::deltaNhfNFW` (≈lines 123–127): each (z-shell, mass-bin) cell's
Poisson mean is modulated per realization by an INDEPENDENT log-normal
λ = exp(δ_b − σ_b²/2), with amplitude

σ_b = D_g(z) · b_halo(M,z) · **σ(M_b)**,  M_b = 2π r_max² [d_c(z)−d_c(z−Δz)] ρ_M0

— the mean mass of the tube segment whose LENGTH IS THE SHELL WIDTH and
whose RADIUS IS r_max(κ_thr). Consequences:

- **Refining Nz halves the segment → halves M_b → raises σ(M_b)** (smaller
  regions fluctuate more), while the draws stay independent per cell. The
  same stretch of universe gets more, wilder, mutually independent
  environments as the grid refines. This is not a discretization of any
  continuum random field: there is no Nz→∞ limit. As Nz→∞, σ(M_b)→∞ and
  the fraction of halos delivered inside >Λ-overdense bursts,
  E[λ·1(λ>Λ)] = Φ_c(lnΛ/σ_b − σ_b/2), → 1 for every Λ. Burst-delivered
  halos share a z-cell, stack on the ray, and build the growing compound
  κ>1 tail. (E[λ]=1 always ⇒ mean counts and the PDF body are untouched —
  which is why every JSD-level verdict survives.)
- **The known "σ_b blows up for κ_thr ≳ 3e-3" gotcha is the SAME disease
  through the transverse direction**: larger κ_thr → smaller r_max →
  smaller M_b → σ(M_b) explodes. Two documented pathologies, one cause:
  the modulation amplitude is tied to an unphysical, knob-dependent region
  size.
- **The mass-bin direction is also grid-coupled**: cells are independent
  across jM at fixed z, so the bias variance depends on NM as well —
  physically all masses at one z ride the SAME density field scaled by
  b(M,z). Plausibly a contributor to the NM confound seen in the Mmin
  study.

**Validity condition (stated by the reference model itself):** the bin is
"chosen so that M_b is much larger than the typical lens masses" (quoted
from the model description). I.e. the prescription is a peak–background
split whose scale separation is imposed implicitly by the grid geometry;
the default grid was chosen inside the validity domain, and refinement
exits it. The Δz→0 divergence is the double counting of halo-scale
structure as "environment."

## 4. Fix options

(a) **Freeze-and-document (zero code):** declare the Nz=100 (and default-
    κ_thr) segmenting part of the model definition — defensible given the
    paper's own condition — and forbid quoting bias-tail quantities
    (f(κ>1), q≳99.9 at z≳5, wide-support ⟨κ²⟩) as converged physics.
(b) **Fixed comoving segments L_b (minimal patch):** partition the LOS into
    Nz-independent segments of fixed length; one δ per segment per
    realization, shared by all shells and mass bins inside it (scaled by
    b(M,z)D(z)); σ from σ(M_b(L_b)). Restores a continuum limit; L_b is an
    explicit model parameter (sensitivity-scan it).
(c) **RECOMMENDED — explicit peak–background split field:** define ONE
    environment field δ_env(χ) along the LOS = linear density field
    low-pass filtered at a fixed comoving R_bg (≈ Lagrangian radius of the
    heaviest modulated halos, ~10–15 Mpc — the paper's condition promoted
    from a grid constraint to a field definition). Realize per realization
    on a coarse fixed grid (~R_bg spacing, where independence is
    approximately true) + interpolate, or exactly via Cholesky of the
    filtered-P_lin covariance (~100 nodes, negligible cost). Modulate every
    cell by exp(b·D·δ_env(χ_j) − ½[…]²). Amplitude is finite and
    grid-independent; the tube radius drops out ⇒ PREDICTION: the
    κ_thr ≳ 3e-3 blow-up disappears too (use as a validation gate).
    Ship behind a `bias_model` flag (legacy default 0 for bitwise
    reproducibility, same pattern as `kappa_anchor`).

Non-option: raising Nz (makes the model heavier-tailed, not more accurate).

## 5. Pre-implementation diagnostics (open)

- **M_b/M validity map at the default grid**: back-of-envelope suggests
  M_b ~ 1e13–1e14 for the heaviest mass bins at Nz=100, i.e. the
  "M_b ≫ M_lens" condition may already be only marginally satisfied at the
  top-mass bins BEFORE any refinement. Needs r_max(M,z) exposed (small
  binding or debug print). Quantify before choosing R_bg.
- Post-fix acceptance (harness already exists): f(κ>1) flat in Nz;
  κ_thr-sweep σ_κ no longer blowing up at ≳3e-3; JSD(old default, new
  default) at Nz=100 quantified; R_bg (or L_b) sensitivity scan.

## 6. Scope — what is and is not affected

Unaffected: all five knob-audit verdicts (κ_thr rule, subhalo_factor,
eps_floor, Mmin, Nz at the P(lnμ)/JSD level), the emulator and its
training/calibration pipeline (trims these rays by design), posterior
recovery. Affected (currently grid-defined, do not quote as physics):
f(κ>1), far-tail quantiles q≳99.9 at z_s≳5, raw or wide-support ⟨κ²⟩ —
and any future strong-lensing-adjacent application. Cross-study
reinterpretation: the Mmin/Nz-truth far-tail excesses (q99.9 undershoot
~25–30% at z=10) are bias-layer grid effects, NOT solely the κ_thr_eff
split coupling as previously inferred (that inference stands for the
κ_thr study itself).

## 7. Data & script inventory

- Sweeps: `scripts/convergence/vark_vs_nz.py` (+`_frozen`, `_nobias`) →
  `data/results/vark_nz/{report,report_frozen,report_nobias}.md`, shards +
  npz per arm; mechanism chain `data/results/vark_nz/mechanism_note.md`.
- Analytic: count scan (get_expected_halo_count, in mechanism_note);
  `data/results/convergence_analytic/sigW_nz_extended.npz` (Nz→6400).
- Figures (`scripts/figures/plot_convergence_vs_knobs.py`):
  `plots/nz_tail_mechanism.png`, `plots/vark_total_vs_nz.png`,
  `plots/vark_total_vs_knobs.png`, `plots/vark_vs_knobs.png`,
  `plots/convergence_excess_vs_knobs.png`.
- Seed namespaces: 8.0e8 (rule), 8.1e8 (frozen), 8.2e8 (bias-off) —
  disjoint from all prior studies.
