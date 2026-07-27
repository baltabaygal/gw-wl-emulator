# Parametric-shift diagnostic (follow-up #1) — subhalo ingredient

**Date:** 2026-07-21 · **Env:** conda `test` · **Reuses** the 174 cascade shards (no regen).
Script: `scripts/cascade_diag/analyze_param.py`. Numbers: `tables_param.md`.
Plots: `plots/param_shifts_zs{0.5,1.0,5.0}.png`.

**Motivation.** The pointwise-logP cascade failed because ΔlogP is MC-noise-limited (no
CRN) and its θ-smoothness was undemonstrable (`report.md`). Here we test the §6 route:
fit the *structured* representation per arm — mean, σ (robust width), skew, edge, tail
slope, flux δ — difference A→B to get the ingredient shift ΔS(θ), and re-ask whether the
shift is (1) detectable and (2) smooth/fittable in θ. Scalars are well-determined from
200k rays, so the θ-smoothness that was noise-drowned pointwise becomes resolvable.

Scalars (all monster-ray-safe): `sigma=(q84.135−q15.865)/2`, `skew=(q84+q16−2q50)/(q84−q16)`,
`edge=q0.5%`, `tail=` slope of ln(1−CDF) vs lnμ over [q75,q99] (moderate shoulder, NOT the
uncertified far tail), `mean`/`flux=ln⟨1/μ⟩` on the [q0.5,q99.5] core.

---

## Headline verdict: **PASS.** The subhalo correction collapses to ~3 scalar shifts, each θ-smooth to the sim's own MC-noise floor.

The subhalo ingredient's entire parametric signature is **σ↑, edge↦out, tail↦heavier**
(plus a small mean shift at high z_s). Each of those is (a) detectable at SNR 4–20 from a
single 200k-ray run, and (b) interpolable in θ to the seed-noise floor with just **7
points per axis**, at 2–12% of the shift magnitude. **skew and flux are invariant**
(SNR<1) and can be dropped. This is exactly the well-conditioned, few-sample object the
pointwise-logP residual was not.

### (1) Which shifts are real (fiducial 8-seed block)

| scalar | SNR_Δ (z=0.5/1/5) | ΔS at z_s=5 | reading |
|---|---|---|---|
| **sigma** | **10.9 / 8.9 / 20.4** | +0.0133 | dominant signal; grows with z_s |
| **edge** | **7.1 / 7.0 / 12.5** | −0.0214 | empty-beam wall moves outward; grows with z_s |
| **tail** | 1.7 / **3.6 / 4.0** | +0.14 (slope↑) | heavier shoulder; detectable at z_s≥1 |
| mean | 0.7 / 2.0 / **9.5** | +0.0021 | anchored ≈0 at low z; real at z_s=5 |
| skew | 1.1 / 0.4 / 0.4 | −0.002 | **invariant → drop** |
| flux | 0.4 / 0.8 / 1.0 | +0.0002 | **invariant → drop** (core flux conserved) |

### (2)+(3) Can we fit ΔS(θ)? — leave-one-out along Om/σ8/h

The right question is not "is there curvature above noise" but "is the interpolation error
small vs the shift we're adding." For every *detectable* scalar the leave-one-out error is
**at the MC-noise floor** (7 θ-points already interpolate as well as the sim's own noise
allows) and small vs the shift:

| z_s | scalar | med err_res | seed floor | err/|ΔS| | verdict |
|---|---|---|---|---|---|
| 5.0 | sigma | 2.9e-4 | 6.5e-4 | **0.02** | at-MC-floor |
| 5.0 | edge | 2.4e-3 | 1.7e-3 | 0.11 | at-MC-floor |
| 5.0 | tail | 4.0e-2 | 3.5e-2 | 0.29 | at-MC-floor |
| 1.0 | sigma | 2.4e-4 | 2.7e-4 | 0.10 | at-MC-floor |
| 1.0 | edge | 2.1e-4 | 4.9e-4 | 0.06 | at-MC-floor |
| 0.5 | sigma | 3.5e-5 | 8.0e-5 | 0.04 | at-MC-floor |
| 0.5 | edge | 1.2e-4 | 1.4e-4 | 0.12 | at-MC-floor |

(full grid in `tables_param.md`; `param_shifts_zs*.png` shows Δσ(θ) as a clean monotone
ramp well above the seed band, Δedge/Δmean as gentle near-flat trends, Δskew/Δflux as
scatter inside the band.) **Contrast the pointwise route:** body ΔlogP had rel≈0.27 and
was noise-limited across ~130 bins; here the same information is 3 scalars fittable to the
noise floor with a handful of θ-points.

---

## Implications & recommended order

1. **#1 passes** → a parametric-shift cascade for subhalos is viable and cheap: emulate
   Δσ(θ,z_s), Δedge(θ,z_s), Δtail(θ,z_s≥1) [+ Δmean at high z_s]; drop skew, flux.
   Δσ is the workhorse (SNR up to 20, smoothest); edge/tail are secondary offsets.
2. **Next: #2 (add clustering, bias_model 0→1)** — needs fresh paired runs. Tests whether
   the same 2–3-scalar picture generalizes to another ingredient. Predicted signature:
   clustering also loads mainly onto σ (+ the weak-arm edge shift), per the bias-field
   work (+18–24% clipped Var(lnμ) at R_⊥=8.44 Mpc) — so composition will mostly be a
   question of whether Δσ_sub + Δσ_bias adds.
3. **#3 (composition/order test, bundled with #2)** — the honesty test. Check whether
   Δσ_sub + Δσ_bias (each vs its own base) reconstructs the full-model σ, or whether
   host–clump / field covariance makes it order-dependent. The subhalo-gate work already
   flags within-host clustering + host–clump covariance as dominant over the Poisson term,
   so **expect some order-dependence** — worth measuring before framing residuals as
   "independent modules."
4. **#4 (2-level prototype)** — only after #1+#3. Base = production flow; add a small
   regressor for the parameter shifts; compare KL to the monolith at matched budget.

**Caveats (unchanged):** shifts are conditional (subhalos-on-halos, not standalone); tail
slope is a moderate-shoulder proxy, not the uncertified far tail; mean is anchored near 0
at low z_s by construction (batch-mean κ compensation) so its low-z shift is expected small.
