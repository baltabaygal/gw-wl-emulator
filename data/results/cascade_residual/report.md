# Cascade (hierarchical-residual) emulator — subhalo make-or-break diagnostic

**Date:** 2026-07-21 · **Env:** conda `test` (py3.12) · **Ingredient tested:** subhalos
(`subhalo_model=3`, `subhalo_factor=1e-2`) on top of the halo-only baseline.

**Question this answers (and ONLY this):** is the pointwise log-P subhalo *residual*
`ΔlogP = logP_B − logP_A` actually smaller in dynamic range and smoother in θ than the
base `logP_A` — separately in the low-μ **edge**, the **body**, and the high-μ **tail**?
This decides whether a pointwise-logP cascade is worth building. It is **not** the cascade.

Reproduce: `scripts/cascade_diag/{gen_worker,run_all,analyze}.py`. Data (174 shards,
N=200k rays each) in scratchpad `cascade_data/`. Design: A/B arms at 3×7-point Om/σ8/h
axis sweeps × z_s∈{0.5,1,5} (grid, 1 seed) + a fiducial 8-seed noise block. Regions
assigned per-bin by the base-arm CDF: EDGE (CDF<1%), BODY (1–99%), TAIL (>99%).
Numbers: `tables.md`. Plots: `plots/{residual_curves,edge_zoom,summary_bars}.png`.

---

## Headline verdict

**Mixed, and the split matters.** The subhalo residual *is* far smaller in dynamic
range than the base — decisively in the **body** — but a **pointwise-logP cascade is
the wrong representation**: the residual is **MC-noise-limited**, not curvature-limited,
because turning the ingredient on **breaks common random numbers**, and it is **not
demonstrably smoother in θ** once you account for that noise. Recommendation: pursue the
cascade only on the **structured/parametric** representation (§6 of the brief), not on
pointwise logP.

| region | dynamic range smaller? | usable SNR? | smoother in θ (per unit signal)? |
|---|---|---|---|
| **body** | **YES** — 4–7% of base | YES — SNR≈4.5–6 | **NO** — measurement noise-limited (rel_res/rel_base 1.1–3.2) |
| **edge** | YES — 14–16% of base | marginal — SNR≈3.3–3.9, diverges *at* the wall | NO — rel_res/rel_base 3–8; few defined bins |
| **tail** | partly — 22–29% of base | **NO** — SNR≈1.5–1.9 (≈ noise) | NO / indistinguishable from noise |

---

## 1. Dynamic range — the claim HOLDS (best in the body)

Robust (5–95 pctile) spread of the curve over defined bins, residual vs base:

| z_s | edge res/base | body res/base | tail res/base |
|---|---|---|---|
| 0.5 | 0.14 | **0.04** | 0.26 |
| 1.0 | 0.15 | **0.05** | 0.29 |
| 5.0 | 0.16 | **0.07** | 0.22 |

The residual carries **4–7% of the base's dynamic range in the body**, ~15% at the edge,
~22–29% in the tail. `residual_curves.png` shows why: ΔlogP is a clean, low-order shape —
a sharp **+0.7 spike at the empty-beam wall**, a shallow **negative dip** just above it,
a broad **+0.1–0.2 positive shoulder** through the mid-body — riding on a ~4-unit base.
So the *first* claimed advantage (smaller dynamic range → less nonlinear object) is real,
and it is genuinely smooth **in lnμ**. This is the encouraging half.

## 2. MC noise floor & CRN — the binding constraint

**Common random numbers do NOT survive the ingredient toggle.** At a fixed seed the two
arms have per-ray correlation ≈ 0 (measured −0.007): the subhalo branch consumes extra
RNG and reshuffles the whole downstream stream. So `ΔlogP` is a difference of two
*independent* estimates — its MC noise is the **sum**, not a suppressed difference. The
8-seed block gives the per-bin noise directly:

| z_s | region | signal (dr_res) | noise_res | **SNR_res** | noise_res vs noise_base |
|---|---|---|---|---|---|
| 1.0 | edge | 0.39 | 0.101 | 3.9 | 0.101 > 0.074 |
| 1.0 | body | 0.21 | 0.043 | 4.8 | 0.043 > 0.031 |
| 1.0 | tail | 0.48 | 0.253 | 1.9 | 0.253 > 0.168 |

(other z_s identical in character; `tables.md`). Two things to read off:

- **The residual's own noise is ~30–50% larger than the base's** (no CRN), while its
  signal is 4–20× smaller ⇒ the dynamic-range win in §1 is **partly eaten by SNR**.
- **Tail SNR ≈ 1.5–1.9 — the pointwise tail residual is at the noise floor** (visible as
  the exploding purple band in `residual_curves.png`). Body SNR ≈ 5 is workable; edge
  ≈ 3.5 but diverges right at the wall (§4).

## 3. Smoothness in θ — NOT demonstrated; measurement is noise-limited

Leave-one-out linear interpolation error along each 7-point axis, normalized by the
curve's own dynamic range along that axis (`rel`); `rel_res/rel_base < 1` would confirm
"residual needs fewer θ samples." **It does not:**

- **body:** rel_res/rel_base ≈ **1.1–3.2** (residual *worse* per unit signal)
- **edge:** ≈ **3–8** (much worse)
- **tail:** ≈ 0.8–1.6 (comparable, both noise-dominated)

But this is **not** evidence the residual is intrinsically wiggly in θ. The absolute
interp error `err_res` in the body (0.036–0.049) **equals the MC noise floor `noise_res`
(0.041–0.043)** — i.e. the leave-one-out test is measuring **noise, not θ-curvature**.
The base is nearer noise-limited too, but its far larger genuine signal keeps its
*relative* error low. **Conclusion:** at 200k rays/config with no CRN, the residual's
true θ-smoothness is **unresolvable** — MC noise swamps it. The "fewer θ samples" claim
can be neither confirmed nor realized in the pointwise-logP representation without either
CRN (impossible here) or a large increase in rays/config.

## 4. Edge — moves modestly, diverges at the wall

Subhalos **lift and slightly widen** the empty-beam edge (`edge_zoom.png`): logP_B sits
above logP_A at the wall and the robust edge (0.5% quantile) shifts outward with z_s:

| z_s | Δ edge (q0.5%) |
|---|---|
| 0.5 | −0.0010 |
| 1.0 | −0.0035 |
| 5.0 | −0.0214 |

Small but real and z_s-growing — the edge is a *moving* feature, as the brief warned.
The residual **diverges at the very wall** (the +0.7 spike) and only **7–12 bins** are
defined in the EDGE region in *both* arms, so a pointwise log-residual is ill-posed
exactly where the physics is sharpest. (`min` lnμ per arm is monster-demag-ray junk —
ignore; use the quantile.)

---

## Recommended next direction (do NOT build yet)

The diagnostic supports the brief's §6 hypothesis. The residual is a **clean, smooth,
low-order object in lnμ within the body**, but the **pointwise-logP** encoding is the
wrong target: it is MC-noise-limited (no CRN), ill-posed at the moving edge, and
uninformative in the tail. The right cascade — if pursued — should emulate how each
ingredient **shifts the structured/parametric representation the production model already
uses**: σ(lnμ), the tail index, the edge/empty-beam location, and the flux δ. Those are
(a) low-dimensional, (b) far more robust to MC noise than per-bin logP, and (c) exactly
where the subhalo residual concentrates its signal (a variance/shoulder bump + a small
edge shift). A parametric-shift cascade also sidesteps the moving-support problem that
sinks the pointwise version.

**Caveats (honest):** residuals are conditional/non-commuting (this only characterizes
subhalos-on-halos, not a standalone effect); tail statements are qualitative
(uncertified far tail); "no CRN" is specific to this forward model's RNG structure and is
itself a first-class finding for the cascade design.
