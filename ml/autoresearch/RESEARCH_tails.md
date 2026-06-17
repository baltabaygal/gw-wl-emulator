# Research: heavy / power-law tails in (conditional) density estimation

Triggered by the residual we hit: a single *global* tail knob (tilt/reweight) can't fit a
tail whose amplitude varies ~10x across the prior — it nails high-structure tails but
overshoots thin low-structure tails (held-out TLSE 0.325 vs production 1.64). Question:
how does the ML/stats community handle this? Findings below.

## 1. Why the plain flow failed (it's a theorem, not a bug)

**Jaini, Kobyzev, Yu, Brubaker (2020), "Tails of Triangular Flows".** Lipschitz triangular
flows — which *includes* RealNVP and Neural Spline Flows — provably **cannot** map a
light-tailed (Gaussian) base to a heavy-tailed target. Bi-Lipschitz maps preserve the tail
class. So the original emulator's Gaussian-base NSF *had* to collapse in the tail; our switch
to a Student-t base was the right structural fix.
https://www.semanticscholar.org/paper/31558e5bf78f530f07428e6e5af59ae5a3c704da

## 2. Two established families for the heavy-tail part

**(a) Heavy-tailed base** — TAF / gTAF / **mTAF** (Laszkiewicz et al., ICML 2022). Student-t
base with learnable degrees of freedom; mTAF uses EVT to estimate each marginal's tail index
and builds a Gaussian+t base. (We did a fixed-df Student-t — this is the learnable version.)
https://proceedings.mlr.press/v162/laszkiewicz22a/laszkiewicz22a.pdf

**(b) Tail Transform Flow (TTF)** — "Flexible Tails for Normalizing Flows" (ICML 2025).
Keep a Gaussian base, put the heaviness in a **final transform**
`R(z)=μ+σ(s/λ)[erfc(|z|/√2)^(-λ) - 1]` with per-tail indices λ±. Empirically **beats**
Student-t-base flows (it keeps extreme values out of the NN layers → stable gradients). Can be
fixed from a Hill estimate (TTFfix) — and we already have that (α≈3.4). **Caveat: they did NOT
get conditional tail params to optimise** ("harder to optimise, left for future work").
https://arxiv.org/abs/2406.16971

Key point: (a) and (b) make the tail heavy, but **both are global/unconditional**. On their own
they'd hit the *same* amplitude-across-the-prior problem our tilt did.

## 3. The actual residual is "conditional EVT" (covariate-dependent tail)

Our hard part — tail amplitude varying with (z, h, Ωm, σ8) — is exactly **extreme value
regression**: model the tail above a threshold with a **Generalized Pareto Distribution (GPD)
whose parameters depend on covariates**.

**EQRN — "Neural Networks for Extreme Quantile Regression" (Pasche & Engelke).** NN outputs GPD
scale σ(x) and shape ξ(x); splice above an intermediate-quantile threshold; reconstruct extreme
quantiles from the GPD. Crucially they note you can **hold the shape ξ constant and let only the
scale vary with covariates** — which is *precisely* our case: our benchmark found the tail SLOPE
is ~universal (α≈3.3-3.5 ⇒ GPD shape ξ≈1/(α-1)≈0.42, constant) and only the **amplitude/scale**
moves with cosmology. https://arxiv.org/abs/2208.07590

## 4. What this implies for us

The literature splits our problem cleanly:
- **Slope/shape:** universal → fix it (our α≈3.4 benchmark; Jaini says you need a heavy base or
  tail-transform to represent it at all — done).
- **Amplitude:** covariate-dependent → must be **conditional**, not a global tilt. A global knob
  is the documented wrong tool here.

Two viable routes, both literature-backed:

- **R1 — Body–tail splice with a conditional GPD tail (recommended).** Flow models the body
  (already excellent) below a threshold μ_u; above it, an analytic power-law/GPD tail with
  **fixed universal shape** (ξ from our Hill fit) and **scale pinned by continuity to the flow's
  density at μ_u**. The amplitude then comes *for free* from the flow's well-fit, conditional
  body density at the threshold — no global tilt, no overshoot, smooth by construction. This is
  the standard EVT peaks-over-threshold splice; it's also the user's earlier "hybrid analytic
  tail" option, now strongly justified AND made conditional.
- **R2 — Pure flow with a *conditional* heavy-tailed base/tail-transform.** TTFfix or
  learnable/conditional-df Student-t, with the tail index/scale a function of context. Most
  elegant (single model) but the TTF/TAF authors themselves flag conditional tail params as
  hard to optimise — research-grade risk.

Recommendation: **R1**. It directly solves the amplitude residual (the thing that broke the
global tilt), is the established EVT approach, guarantees a smooth power-law tail, and reuses the
strong body flow we already have. R2 stays "pure flow" but is the harder, less certain path.

## 5. Domain SOTA: ACE-Lensing (Türker, Marra, Castro, Quartin, Borgani 2025)

arxiv 2512.01607 / github Turkero/ACE-Lensing. Same problem (emulate lensing magnification
PDF for SNe/GW), different method: N-body light-cones -> bin PDF into 5000 log bins over
**mu in [0.1, 6]** -> standardize (mu-mean)/sigma -> **PCA (4 components)** -> **XGBoost** to
interpolate components+mean+sigma over (Om, s8, w, h, z). Median KL 0.007.

Crucially for us, **they hit our exact problem and punt on the tail**:
- "extending beyond mu=6 degraded PCA performance because the high-mu tail disproportionately
  weighted the variance, downweighting the body" -> they CAP at mu=6, trim at max/1e4. (Same
  variance-domination that made our spline bound/std a problem.)
- "modeling is less reliable at very high magnifications."
- They justify it: "a local discrepancy of 50% in the high-mag tail corresponds to a weighted
  relative difference of only 2% in the magnification statistics" (L1 argument) + note baryons
  affect mu>~3 anyway (DMO sims).

Takeaways: (a) the SOTA domain emulator does NOT solve the rare strong-lensing tail -> our tail
focus is genuinely novel/valuable; (b) PCA-on-binned-PDF is WORSE for tails than our
flow-over-samples (+analytic tail); (c) their L1 metric is a useful sanity check on how much tail
error actually matters for a given downstream statistic.

## 6. The root cause is DATA sparsity, not just the model

Every model (flow, EVT, PCA) struggles with the tail because there are few tail samples
(rare events). Importance sampling / rare-event simulation is the standard fix across fields
(reliability, finance, particle physics): bias the generator toward the rare region and reweight,
turning ~10 tail events/config into hundreds. ML-for-IS and synthetic rare-event data are active
areas (refs below). Our simulator's `bias` flag is halo-clustering physics, NOT high-mu IS, so
options are: (a) brute-force more samples per config (linear cost; 100k-1M/config), focused on
high-z/high-structure where the tail lives; (b) modify the C++ to importance-sample halo
configurations toward high magnification (big tail efficiency gain, non-trivial).

## 7. Synthesis: two orthogonal levers to improve the tail

- **DATA lever (most fundamental):** more tail signal via (a) brute-force higher N at
  high-z/high-structure configs, or (b) importance-sampling the simulator. Lifts EVERY method.
- **MODEL lever:** conditional EVT — GPD with covariate-dependent SCALE (universal shape from our
  alpha~3.4 benchmark) fit to the exceedances (EQRN). Our survival-anchored splice is a simplified
  version (scale inherited from the body flow's survival); the full version fits scale to tail data
  directly and would fix the residual low-structure corner. Pure-flow alternative: TTF heavy tail.

Recommendation: combine — enrich tail data where it matters, keep body-flow + conditional
power-law tail, and (if needed) upgrade the splice amplitude from "flow survival" to a small
conditional-GPD-scale fit on the richer exceedance data.

## Sources
- Jaini et al. 2020, Tails of Triangular Flows — https://www.semanticscholar.org/paper/31558e5bf78f530f07428e6e5af59ae5a3c704da
- Laszkiewicz et al. 2022, Marginal Tail-Adaptive Normalizing Flows — https://proceedings.mlr.press/v162/laszkiewicz22a/laszkiewicz22a.pdf
- Flexible Tails for Normalizing Flows (TTF), ICML 2025 — https://arxiv.org/abs/2406.16971
- Pasche & Engelke, Neural Networks for Extreme Quantile Regression (EQRN) — https://arxiv.org/abs/2208.07590
