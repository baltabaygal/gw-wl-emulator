# Cascade correction — mode (SVD) analysis: is each ingredient a distinct, low-D, orthogonal deformation?

**Date:** 2026-07-21 · analysis-only (nothing built/trained). Scripts:
`scripts/cascade_diag/{mode_analysis.py, run_ext.py}`. Auto-tables:
`report_modes_lowz.md`, `report_modes_highz.md`. Figures (paper candidates):
`plots/mode_{dlogp_curves,sv_spectra,shapes}_{lowz,highz}.png`,
`plots/mode_orthogonality_summary.png`.

**Method.** Per ingredient i∈{sub,bias} and z_s, correction matrix
`M_i[θ,lnμ] = logP_{base+i} − logP_base` over the Om/σ8/h grid (θ rows) on a shared lnμ
grid. **Traps handled:** (1) moving edge MASKED (body+shoulder only, lnμ∈[q1,q99] of the
base; edge kept as a separate scalar); (2) MC noise floor on the singular-value spectrum
from the 6–8-seed block, same light Hann smoothing on data and noise, floor = 95th pct of
noise singular values; a mode is real only if S_k > floor(S1) (the top noise mode);
(3) both UNCENTERED (correction shape) and θ-CENTERED (cosmology dependence) SVD.
z_s∈{6,8,10} extension (base+sub; base+bias at z=6) added for the emergent-mode check —
**body/shoulder shape only, NO tail-amplitude claims (uncertified there).**

---

## Verdict up front — the clean thesis is REFUTED; the real story is stronger for the parametric case

| hypothesis | verdict |
|---|---|
| **H1** subhalo ≈ 1 width mode (weak 2nd tail mode by z=5) | **CONFIRMED (1 mode), 2nd-mode part REFUTED** — subhalo is a single width-like mode from z=0.5 to **z=10**; no 2nd mode emerges in the certified body/shoulder |
| **H2** clustering ≈ 1D SKEW mode, ORTHOGONAL to subhalo width | **REFUTED** — clustering is 1–2D (not a single mode), and the dominant modes are **collinear, not orthogonal** (cos u_sub·u_bias = +0.90/+0.87 at z≤1, **−0.89** at z≥5) |
| **H3** PC1≈∂P/∂σ, PC2≈∂P/∂skew | **PARTIAL** — PC1≈∂P/∂σ holds (PC1·H2 ≈ 0.75–0.86); **PC2≈∂P/∂skew does NOT** — PC2 is at/below the pointwise noise floor |

**The reframed thesis (what the paper should actually claim — tempered to what the SVD
licenses):** *"Weak-lensing PDF deformations are low-dimensional but not orthogonal;
physical origin survives primarily through moment amplitudes and redshift evolution, and is
efficiently recoverable only in the operator representation at realistic simulation
budgets."*
1. Each ingredient's pointwise correction is **low-dimensional at resolvable SNR** — ≤1
   (subhalo) or 1–2 (clustering) modes clear the noise floor. This bounds the *resolvable*
   dimensionality, **not** the intrinsic one: a later ingredient (baryons, WDM) could add a
   mode without breaking the statement. Write the version that survives future data —
   "low-D at resolvable SNR", never "intrinsically low-D".
2. The ingredients are **collinear in pointwise shape space** — both produce the *same*
   dominant asymmetric **shoulder-broadening** shape (cos ≈ 0.9). They differ in **amplitude
   and z-trend**, not mode direction: the subhalo width grows with z_s and stays positive;
   the clustering core width is smaller and **flips sign by z_s≈5** (cos → −0.89). **This
   sign flip is now established as physics** — it survives a robust anchor (mean excluding
   κ>1) AND a second σ estimator (all three agree, SNR 4–18 at z_s≥5; see the "Sign-flip
   artifact check" section). It is NOT an anchoring/metric artifact. (Physics paragraph is
   hedged: z_s≥5 is outside the event population; mechanism deferred to N-body.)
3. The physically-distinguishing feature (subhalo→width vs clustering→**skew**) is **real in
   the parametric moment** (`report_compose.md`: clustering Δskew SNR 5.3, subhalo ≈0) but
   sits **below the pointwise-shape noise floor** — the clustering-unique component has only
   moderate H3 overlap (0.27–0.50) and PC2 is noise. This is an **estimator-variance**
   statement, held **operationally**: at achievable Monte-Carlo budgets the operator
   coefficients are the high-SNR estimators and the ingredient distinction is *practically
   inaccessible* in the pointwise representation. It is **NOT** a claim that moments are
   more fundamental than the PDF — with unlimited samples the pointwise PDF strictly
   dominates any finite moment set. Do not promote "operationally efficient coordinates" to
   "intrinsic coordinates"; the tempered version is unassailable, the strong version invites
   "no, the PDF is; you're noise-limited".
4. Additive composition worked (#3) **not because the modes are orthogonal** but because the
   corrections are small and dominated by a shared low-D direction that superposes linearly,
   with negligible interaction.

> **⚠ Sign-flip caveat (kill the artifact hypothesis before the physics).** The z_s≳5
> anti-collinearity (cos → −0.89) traces to clustering's κ-core Δσ turning negative there —
> but that sits exactly where `report_metric.md` documents (i) the batch-mean-κ anchoring
> bug, (ii) metric-fragile, clip-sign-dependent clustering σ, and (iii) *outside* the event
> population (z≤4). The first hypothesis to kill is **measurement artifact**, not a
> nearby→line-of-sight-averaging physical transition. Required test (run before any physics
> narrative): re-measure with a **robust anchor (mean excluding κ>1)** and a **second σ
> estimator**; only if the sign survives a clean estimator is it physics. Result:
> **see the "Sign-flip artifact check" section** appended below.

> **⚠ v2 reconstruction result (the check, partially cashed).** The premise — that
> {Δσ, Δedge, Δtail, Δskew} reconstruct the PDF to budget — was tested (`ml/cascade/
> reconstruction_v2_report.md`). **Naive v2** (measured moment shifts used directly as warp
> knobs, θ-independent) comes back **short**: population-weighted (z~U(0.3,4)) it closes only
> ~3% of the ingredient excess, dominated by the heavily-weighted low-z panels. **BUT an
> oracle calibration disambiguates the cause: the operator basis is SUFFICIENT** — freely
> fitting the body-warp knobs drives KL to ~0 at z_s=0.5 and below the floor at z_s=3. So the
> shortfall is **calibration** (measured moment shift ≠ correct warp-knob value — the warp
> response is attenuated/reshaped), **not** a representation gap. The thesis survives with a
> precise qualifier: *the operators can reconstruct the PDF, but the cascade needs a
> knob-calibration stage* (fit knobs to the realized moment, or regress knobs to minimize
> reconstruction KL) — not the naive measured-moment-as-knob mapping. Cashing the check fully
> is one calibration stage away, not a dead thesis.

---

## A. Per-ingredient SVD (uncentered; modes above the top-noise-mode floor)

| z_s | ing | S1 | S2 | S3 | floor(S1) | n>floor | PC1 sym | PC1·H2 | PC1·H3 |
|----|----|----|----|----|----|----|----|----|----|
| 0.5 | sub | 4.06 | 1.33 | 1.27 | 1.52 | **1** | +0.09 | 0.66 | 0.56 |
| 0.5 | bias | 4.97 | 1.80 | 1.26 | 1.52 | **2** | +0.03 | 0.46 | 0.35 |
| 1.0 | sub | 4.50 | 1.38 | 1.20 | 1.39 | **1** | +0.08 | 0.76 | 0.63 |
| 1.0 | bias | 2.88 | 1.46 | 1.10 | 1.40 | **2** (marginal, S2 +4%) | −0.19 | 0.45 | 0.30 |
| 5.0 | sub | 6.13 | 1.21 | 1.12 | 1.51 | **1** | +0.09 | 0.78 | 0.64 |
| 5.0 | bias | 4.36 | 1.30 | 1.13 | 1.39 | **1** | +0.35 | 0.84 | 0.75 |
| 6.0 | sub | 6.14 | 1.27 | 1.15 | 1.27 | **1** | +0.12 | 0.75 | 0.62 |
| 6.0 | bias | 5.33 | 1.42 | 1.23 | 1.29 | **2** | +0.32 | 0.86 | 0.77 |
| 8.0 | sub | 6.90 | 1.24 | 1.17 | 1.41 | **1** | +0.06 | 0.78 | 0.65 |
| 10.0 | sub | 7.15 | 1.19 | 1.18 | 1.35 | **1** | +0.17 | 0.80 | 0.68 |

- **Subhalo: exactly 1 mode at every z_s (0.5→10).** PC1·H2 ≈ 0.66–0.80 (width-like in the
  shoulder; see `mode_shapes_*.png` — PC1 tracks the H2 dashed curve, PC2 is noise). The
  hoped-for 2nd (tail-steepening) mode does **not** appear in the body/shoulder even at
  z_s=10. (Any tail-steepening lives in the uncertified far tail — not claimable here.)
- **Clustering: 2 modes at z=0.5/1/6 (robust at 0.5 & 6, marginal at 1), 1 at z=5.** Higher-D
  than subhalo at low z, but PC2 is weak and not a clean skew shape above noise.
- PC1 sym ≈ 0 for all: the dominant mode is **one-sided** (shoulder pile-up), i.e. a mix of
  width+skew, not a pure symmetric width nor pure antisymmetric skew.

## B. Shared vs unique deformation (the honest orthogonality test)

| z_s | cos(u_sub,u_bias) | ‖u_bias⊥‖ | cos(u_bias⊥,H3) | cos(u_sub⊥,H2) | noise‖ |
|----|----|----|----|----|----|
| 0.5 | **+0.903** | 0.43 | 0.27 | 0.46 | 0.14 |
| 1.0 | **+0.869** | 0.49 | 0.45 | 0.69 | 0.22 |
| 5.0 | **−0.889** | 0.46 | 0.37 | 0.10 | 0.15 |
| 6.0 | **−0.893** | 0.45 | 0.50 | 0.05 | 0.11 |

- Dominant corrections are **collinear at z≤1** (both broaden) and **anti-collinear at z≥5**
  (clustering's width shift flips sign). Either way, **not orthogonal** — H2 refuted.
- ~45% of each correction is unique (projection geometry). The clustering-unique part has
  only **moderate** skew (H3) overlap (0.27–0.50), above the noise floor (~0.11–0.22) but
  far from a clean skew mode. So clustering *does* carry an extra, partly-skew component —
  but it is subdominant, not an orthogonal principal axis.

## C. SVD-vs-parametric (H3)
PC1·H2 (width) ≈ 0.66–0.86 across all z_s ⇒ **the leading mode IS the ∂P/∂σ width
direction** — the {σ,…} operator is the right leading knob (cross-validates the parametric
choice). PC1·H3 is also substantial (0.30–0.77) because PC1 is one-sided (shoulder), not
because a clean skew mode is resolved. No robust PC2 ⇒ **∂P/∂skew is not recoverable as a
pointwise shape mode** (it lives below the noise floor — see the reframed thesis).

## D. z_s = 6,8,10 extension (body/shoulder only)
Subhalo stays **1 width-like mode** (n>floor=1; PC1·H2 0.75–0.80) through z_s=10 — the
correction does not gain pointwise dimensionality as the effect grows; it grows in
**amplitude** (S1 4.1→7.1) along a fixed direction. **No tail-amplitude claim is made at
z_s≳5** (uncertified). Clustering at z_s=6 again shows 2 modes and the −0.89 anti-collinear
sign.

---

## Consequences for the cascade design
- Confirms the parametric route: the leading correction is a single ∂σ-like direction
  (subhalo) — the width operator is correct and the additive cascade tracks it (matches the
  +22/44/47/46% reconstruction wins).
- Clustering needs its **skew/2nd-mode** component, which is *below the pointwise floor* —
  so it must be carried as a **parametric moment (Δskew)**, not learned from pointwise
  residuals. This is exactly the v2 spec (wire skew) and explains why the width-only v1
  degraded clustering.
- The paper's central figure is **not** "orthogonal modes." It is: (i) each correction is
  low-D and width-dominated in pointwise shape (`mode_shapes`, `mode_sv_spectra`), and
  (ii) the ingredient-distinguishing physics is resolvable only in the parametric moments,
  not the pointwise shape (`mode_orthogonality_summary` + the report_compose skew SNR).

## Sign-flip artifact check (result) — it SURVIVES: physics, not artifact

Re-measured Δσ (clustering − base) at z_s=1,5,6 (6 seeds × 200k, raw κ,γ) under three
estimators: κ≤1-core σ with the **full-mean anchor**, with the **robust anchor (mean
excluding κ>1)**, and **IQR** on the robust-anchored core.

| z_s | estimator | Δσ_sub | Δσ_bias | SNR_bias |
|----|----|----|----|----|
| 1.0 | core (full anchor) | +0.0006 | −0.0019 | 1.4 |
| 1.0 | core (robust anchor) | +0.0006 | −0.0019 | 1.4 |
| 1.0 | IQR (robust) | +0.0024 | **+0.0015** | 6.2 |
| 5.0 | core (full anchor) | +0.0063 | −0.0088 | 3.8 |
| 5.0 | core (robust anchor) | +0.0064 | −0.0091 | 4.4 |
| 5.0 | IQR (robust) | +0.0130 | −0.0067 | 9.5 |
| 6.0 | core (full anchor) | +0.0069 | −0.0101 | 7.2 |
| 6.0 | core (robust anchor) | +0.0071 | −0.0102 | 6.5 |
| 6.0 | IQR (robust) | +0.0142 | −0.0080 | 17.6 |

- **The anchoring bug is NOT the cause:** robust anchor ≡ full anchor (−0.0091 vs −0.0088 at
  z=5). Rules out the `report_metric.md` pathology as the driver.
- **At z_s≥5 the negative Δσ_bias survives a clean second estimator** — all three agree, SNR
  4–18. The sign flip is **real physics**, not a metric/anchor artifact.
- At z_s=1 it is estimator-fragile and ≈0 (IQR flips to +): clustering barely touches the
  core width at z≤1, then **robustly narrows it by z_s≥5** (opposite to subhalos, which widen
  it — Δσ_sub>0 everywhere). The complementary shoulder/tail *fattening* (the production
  +Var) is the other half of the same effect.
- **Physics paragraph earned (hedged):** clustering narrows the subcritical convergence core
  at high z_s while fattening the shoulder — physically consistent with line-of-sight
  averaging of correlated structure by the broad high-z_s lensing kernel. Mechanism deferred
  to N-body. **Caveats:** z_s≥5 is OUTSIDE the event population (gate-irrelevant); the
  shoulder/tail part is uncertified in this method class; do not quote absolute tail numbers.
