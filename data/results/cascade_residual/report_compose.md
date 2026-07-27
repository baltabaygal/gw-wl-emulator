# Clustering generalization (#2) + composition / order test (#3)

**Date:** 2026-07-21 · **Env:** conda `test`. Script: `scripts/cascade_diag/analyze_compose.py`.
Numbers: `tables_compose.md`. Plot: `plots/composition_sigma.png`.

Four arms per config, matched θ/seed: **base**=halo, **sub**=halo+sub (both reused from #1),
**bias**=halo+clustering, **full**=halo+sub+clustering. Clustering = **production config**
(bias_model=1, spherical top-hat, R_s=20 Mpc, weak arm on; CLAUDE.md 2026-07-20). New arms:
174 shards (bias+full × 87 configs), 200k rays, 0 failures.

---

## #2 — does the parametric-shift picture generalize to clustering? **YES.**

Clustering reduces to the *same* kind of object: a few θ-smooth scalar shifts, each
detectable from one 200k run and interpolable to the MC-noise floor with 7 θ-points.

| scalar | SNR_bias (z=0.5/1/5) | Δσ_bias / reading |
|---|---|---|
| **sigma** | **12.4 / 6.0 / 10.4** | +0.0011 / +0.0014 / −0.0068 — dominant, at-MC-floor all z_s |
| **edge** | **25.8 / 12.5 / 3.0** | wall moves (weak arm), strongest at low z_s |
| **skew** | **5.3 / 1.9 / 2.2** | −0.014 — clustering DOES move skew (subhalos did not) |
| tail | 0.1 / 1.6 / **5.7** | detectable only at z_s=5 |
| flux | 2.1 / 1.8 / **7.1** | weak-arm low-μ shift shows up at z_s=5 |
| mean | 2.8 / 2.2 / 1.2 | ≈ invariant |

So the cascade's per-ingredient object is robust across two very different ingredients.
**But the active scalar set is ingredient-specific:** subhalos load on {σ, edge, tail};
clustering loads on {σ, edge, skew} (+ tail/flux at high z_s). A parametric cascade should
therefore carry the *union* {σ, edge, skew, tail, flux, mean} and let each ingredient move
whichever it moves — still ≤6 numbers vs a full PDF.

> **⚠ σ-definition caveat.** Here σ = robust IQR (q84−q16)/2, a *body* width. Clustering's
> Δσ_bias is **positive at z_s≤1 but negative at z_s=5**, whereas the production model's
> *clipped Var(lnμ)* boost is positive at all z_s (1.15/1.11/1.08). Clustering sharpens the
> IQR core while fattening the shoulders at high z_s — so the two σ measures disagree in
> sign there. Whoever defines the parametric σ for the real cascade must pick the
> Var-based (production) σ, not the IQR, or carry both. Flagged, not resolved here.

## #3 — do the corrections compose? **Additive at leading order; ~2σ hints of order-dependence in σ/edge at z_s=5.**

Non-additivity `NA = ΔS_full − (ΔS_sub + ΔS_bias) = [bias-on-sub] − [bias-on-base]`.
SNR_NA = |NA| / seed-noise(NA); **>3 would mean a real interaction, not MC scatter.**

| z_s | scalar | additive | measured full | NA | \|NA\|/\|full\| | **SNR_NA** |
|---|---|---|---|---|---|---|
| 0.5 | sigma | +0.0020 | +0.0019 | −0.0001 | 4% | 0.5 |
| 1.0 | sigma | +0.0038 | +0.0036 | −0.0002 | 7% | 1.0 |
| 5.0 | sigma | +0.0064 | +0.0047 | −0.0017 | **37%** | **2.0** |
| 5.0 | edge | −0.0167 | −0.0134 | +0.0033 | **25%** | 1.6 |
| (all others) | | | | ≤17% | <1 |

**Reading.** With 8 seeds × 200k rays, **NA is below the noise (SNR_NA < 3) for every
scalar** ⇒ composition is additive to within this test's precision, and a leading-order
"sum of independent shifts" cascade is defensible. **However** the two most physical
scalars — σ and the edge — carry the *largest* fractional non-additivity, and it is
**sub-additive and grows with z_s** (σ: 4%→7%→37%; see `composition_sigma.png`, where the
additive sum overshoots the measured full at z_s=5). This is exactly the direction the
subhalo-gate work predicted: host–clump + host–field covariance makes the ingredients
partially saturate rather than add. At z_s=5 it reaches ~2σ — **suggestive, not yet
significant**; confirming it needs ~6× more rays/seeds at high z_s.

---

## Verdict for the framing (the honesty call the user flagged)

- The cascade's core premise holds: **each ingredient's effect is a small set of
  well-determined, θ-smooth scalar shifts** — true for both subhalos and clustering (#1, #2).
- **Do NOT claim strict orthogonality / "independent modules."** Composition is additive
  only at leading order; σ and edge show a real-direction, z_s-growing sub-additive
  interaction (up to ~37% of the full shift at z_s=5, ~2σ). The correct framing is
  **"physically-structured *conditional* corrections that compose at leading order, with a
  measurable high-z_s interaction"** — still publishable, but as conditional corrections,
  not orthogonal modules.
- **Design consequence:** a 2-level additive cascade will be accurate at z_s≲1 but will
  overshoot at z_s≳5 unless the interaction is absorbed — either by (a) fitting the *full*
  σ,edge on the (sub∧bias) grid rather than summing single-ingredient shifts, or (b)
  adding a small pairwise interaction term ΔΔσ(θ,z_s). Cheap either way at the scalar level.

## Recommended next step (#4, now warranted)

Prototype the 2-level emulator: keep the production flow as base, add a small regressor for
the scalar shifts {σ, edge, skew, tail, flux, mean}(θ, ingredients-on), and compare KL to
the monolith at matched budget. Decide up front whether to (a) sum single-ingredient shifts
(simplest; accept the z_s≳5 overshoot) or (b) fit shifts on the joint grid / add a pairwise
term (captures the interaction). Given the σ-definition caveat, define the parametric σ from
the production clipped-Var, not the IQR used for this diagnostic.

**Caveats:** shifts/interactions are conditional on this forward model and its production
clustering config (20 Mpc top-hat, staged not default-flipped); tail slope is a moderate-
shoulder proxy; NA significance is 8-seed-limited (z_s=5 σ/edge deserve a targeted
higher-precision rerun before any strong order-dependence claim).
