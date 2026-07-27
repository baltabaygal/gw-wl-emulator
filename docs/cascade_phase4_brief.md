# #4 brief — 2-level parametric-shift cascade prototype

**Purpose:** hand-off brief for building the first real cascade deliverable. This is
**not** the 6-stage ambition — it is a *2-level* emulator (production base + one additive
correction layer) proven on *two* ingredients (subhalos, clustering). Diagnostics #1–#3 +
the step-1 metric recompute are DONE and gate this; read `data/results/cascade_residual/`
(`report.md`, `report_param.md`, `report_compose.md`, `report_metric.md`) first.

## What the diagnostics settled (so this brief can be prescriptive)

1. **Pointwise-logP cascade is dead** (MC-noise-limited, no CRN). Emulate **parameter
   shifts**, not per-bin log-P. (`report.md`)
2. **Each ingredient = a few θ-smooth scalar shifts**, detectable from one 200k run and
   interpolable to the MC-noise floor with ~7 θ-points. Subhalos load on **{σ, edge,
   tail}**; clustering on **{σ, edge, skew}** (+tail/flux at z_s=5). Carry the **union
   {σ, edge, skew, tail, flux, mean}**. (`report_param.md`, `report_compose.md` #2)
3. **Corrections compose additively at leading order.** The composition interaction (NA)
   is statistically **insignificant in every σ metric** (max SNR_NA=2.0 IQR, 0.2 on the
   certified κ≤1 core); the earlier "37% at z_s=5" was an IQR small-number artifact.
   (`report_metric.md`)
4. **σ is metric-fragile for clustering** — define the parametric σ on the emulator's
   exact production support (clipped-Var / κ≤1), never a convenience proxy. (`report_metric.md`)

## Architecture — reuse, don't invent

The production emulator (`ml/autoresearch/smooth_model.py`, `-ar` worktree, branch
`autoresearch/jun16`) already layers parametric operators on the flow body: the POT tail
params (S2/S3/S8, 3-segment power law, μ_c blend), the empty-beam **edge cutoff** location
(EDGE_W), and the **flux δ** shift. The cascade makes those operators — plus a body
width/skew deformation — **functions of which ingredients are on**:

```
params(θ, ingredients) = params_base(θ)
                         + Σ_i  Δparams_i(θ)          # additive single-ingredient shifts
                         [ + Δ²params_{ij}(θ) ]       # (b)-HOOK: empty pairwise slot
```

where `Δparams_i ∈ {Δσ, Δedge, Δskew, Δtail, Δflux, Δmean}` for ingredient i∈{sub, bias}.
Reuse the existing, validated tail/edge/flux deformation code; add only (a) a small
regressor `Δparams_i(θ, z_s)` per ingredient and (b) a body width/skew deformation of the
flow. **Do not build a new PDF model.**

## Build steps

1. **Datasets** — reuse the 4-arm cascade shards where possible; for the base-params
   fits use the production training pipeline. Fit `Δparams_i(θ,z_s)` from the shift grids
   already generated (base/sub/bias/full, Om/σ8/h × z_s∈{0.5,1,5}); 7 pts/axis is enough
   (shifts interpolate to the MC floor). Extend the θ grid only if the regressor demands.
2. **σ definition (blocking):** measure σ (and the other params) with the **production
   clipped-Var / κ≤1 estimator** and a **robust anchor (mean excluding κ>1)** — use
   `sample_lensing_raw_ml`'s κ field, per `report_metric.md`. Do NOT use IQR.
3. **Shift regressors:** one small model per (ingredient, param) — the shifts are smooth
   and low-D, so a low-order polynomial / tiny MLP / GP suffices. Wire them additively
   into the base params; leave the pairwise slot present but zeroed.
4. **Acceptance gate (decides a vs b empirically):** KL vs the monolith at **matched
   budget**, **resolved by z_s AND event-population-weighted** (marginalize over the mock
   event population — the project's verdicts are body-weighted; the only hinted interaction
   sits at z_s≳5 + far tail, regions the population down-weights). Reuse `validate_kl.py`,
   `validate_pit.py`, `validate_posterior.py` (control must be ~0σ).
   - **Pass** (pure-additive KL ≤ monolith's 0.0073 population-weighted) → **ship (a)**,
     report the high-z_s overshoot as a quantified known limitation.
   - **Breach** → populate the pairwise slot, but first resolve the σ interaction with the
     κ-field + robust-anchor + production-σ estimator (§2), NOT IQR; then re-gate.
5. **Only after a gate breach** spend the ~6× extra rays at z_s=5 to pin the interaction.
   Not before — confirming a ≤2σ effect that is below the accuracy budget is premature.

## Do NOT
- Do not build the 6-stage cascade or the pairwise term pre-emptively.
- Do not use IQR σ anywhere in the real cascade (metric-fragility, `report_metric.md`).
- Do not touch `~/Desktop/halos`. Do not commit/push (user handles git).
- Do not quote far-tail/q≳99.9 as converged.

## Open decisions for the user before build
- **σ operator:** confirm the parametric σ = production clipped-Var (κ≤1) estimator.
- **Clustering config for the "bias" ingredient:** the diagnostics used the production
  20 Mpc top-hat + weak arm (staged, not default-flipped). Confirm that is the target.
- **Event population** for the weighted KL gate (which mock catalog / z_s distribution).
