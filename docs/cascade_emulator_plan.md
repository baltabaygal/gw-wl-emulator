# Plan: Hierarchical Residual (Cascade) Emulator — diagnostic phase

**Purpose of this doc:** hand-off brief for a delegate AI. It gives (1) project context,
(2) the idea, (3) the known risks, (4) a concrete, bounded first task. The delegate's job
is NOT to build the full cascade. It is to run one cheap make-or-break diagnostic that
tells us whether the idea is worth pursuing. Do exactly the scoped task; report numbers.

---

## 1. Context

`gw-wl-emulator` is a project building a fast ML emulator for the gravitational-wave
weak-lensing magnification PDF, P(μ | θ), where θ is a set of cosmological parameters.
The ground truth is a C++ Monte-Carlo forward model (`gwlensing`) that stacks several
physical ingredients on top of a baseline:

- main halos (the baseline),
- subhalos / substructure,
- large-scale halo clustering (a correlated "bias" field + a conditional sub-threshold
  weak arm),
- (future) baryons, filaments, observational systematics.

There is already a **production monolithic emulator** (a normalizing-flow body + a
separate extreme-value POT tail module + an empty-beam edge cutoff + a flux calibration).
It reaches KL ≈ 0.0073 against the simulator — matching the existing ACE-Lensing emulator.
**Any new approach has to beat or match that bar.**

The forward model is expensive; the hard-to-emulate regions are the **low-μ edge**
(empty-beam cutoff, which physically *moves* when ingredients are added) and the
**high-μ tail** (heavy, ~μ⁻² asymptote).

## 2. The idea to test

Instead of one network learning the full P(μ | θ), train a **cascade** of emulators, each
responsible for one additional physical ingredient, and add them in log-space:

```
log P = log P_halo + ΔlogP_sub + ΔlogP_bias + ΔlogP_baryon + ...
```

where, e.g.,

```
ΔlogP_sub  = log P_halo+sub        − log P_halo
ΔlogP_bias = log P_halo+sub+bias   − log P_halo+sub
```

**Claimed advantages** (to be tested, not assumed):
1. Each residual has smaller dynamic range, is smoother in θ, less nonlinear → easier to fit.
2. Each residual is an interpretable physical object (isolates one effect).
3. Modular: add/replace an ingredient without retraining everything.
4. Correction datasets need fewer θ samples than the full model.

## 3. Known risks (why we test before building)

The delegate should keep these in mind — they shape what to measure:

- **The smoothness/small-dynamic-range claim is unproven and most likely to FAIL in the
  tail.** Where P_halo is tiny (far tail), log-differences are small-minus-small and can
  have *larger* range and more noise than the base. Measure per region, not globally.
- **Moving support / edge.** log P is only defined where P > 0, and the edge location
  itself shifts between stages. Pointwise log-residuals near the edge can diverge. Expect
  trouble at low μ; quantify it.
- **Residuals are conditional, not orthogonal.** Effects don't commute (host–clump
  covariance dominates the subhalo effect, for example). ΔlogP_bias-on-top-of-sub ≠
  a standalone clustering effect. This is fine but must be reported honestly.
- **Prior art:** "emulate a correction/boost as a separate object" is standard in
  cosmology (EuclidEmulator boost B(k)=P_nl/P_lin; baryonification/BCemu baryon boost).
  The novelty here is the *multi-level cascade* + application to a *PDF with a moving
  edge and heavy tail*. Frame findings against that, don't overclaim novelty.

## 4. THE TASK (scoped — do only this)

Run the **paired-run smoothness/dynamic-range diagnostic** for ONE ingredient (subhalos).
This decides whether the residual is actually easier to emulate than the base.

### Step 0 — environment (mandatory)
Use the **`test` conda env (Python 3.12)**. System python is 3.13 and will NOT import the
C++ module.
```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python
# or: source /Users/baltabay/miniforge3/bin/activate test
```
Import from repo root:
```python
import sys, numpy as np
sys.path.insert(0, 'build')          # gwlensing C++ module
import gwlensing
```
Key API: `gwlensing.sample_lnmu(...)` returns lnμ samples. Cosmology kwargs: `Om`,
`sigma8`, `h`. Subhalos: `subhalo_model=3` (production default), `subhalo_factor=1e-2`.
Clustering field: `bias_model` (0=off/legacy, 1=correlated field), `bias_window`,
`bias_Rperp`. **⚠ positional arg order of `sample_lnmu` is (z, OmegaM, sigma8, h)** — use
kwargs to avoid the h-first / σ8 mix-up documented in the repo. Confirm the exact
signature with `help(gwlensing.sample_lnmu)` and `gwlensing.get_simulator_config(...)`
before generating data.

### Step 1 — paired datasets (common random numbers)
For a modest grid of θ (start ~20–40 points across the emulator's cosmology prior; e.g.
vary Om, sigma8, h, and source redshift z_s ∈ {0.5, 1, 5}), generate two matched PDFs at
**the same random seed**:
- **A = halo-only** (subhalos OFF),
- **B = halo+sub** (subhalo_model=3, subhalo_factor=1e-2).

Use a large sample count per config (aim ≥1e5 rays, more in the tail) and build P(lnμ) on a
**shared, fixed lnμ grid** so A and B are directly subtractable. Common random numbers
between A and B suppress MC noise in the difference — this matters a lot for the residual.

### Step 2 — compute the residual and measure it
For each θ:
```
ΔlogP_sub(lnμ) = log P_B(lnμ) − log P_A(lnμ)
```
Then report, **split into three regions — body (core, κ_tot≤1-ish / bulk of mass),
low-μ edge, high-μ tail:**
1. **Dynamic range** of ΔlogP_sub vs dynamic range of the base log P_A (max−min, and a
   robust version like 5–95 percentile spread). Is the residual actually smaller?
2. **MC noise floor** of ΔlogP_sub (from seed-to-seed variation at fixed θ) vs its signal.
   If noise ≳ signal in a region, the residual is uninformative there.
3. **Smoothness in θ:** how does ΔlogP_sub vary as you move each cosmological parameter?
   A simple proxy: fit/interpolate ΔlogP at each lnμ bin across the θ grid and compare the
   curvature / interpolation error to that of the base log P_A. The core claim is that the
   residual needs fewer θ samples — test it by holding out points and measuring
   interpolation error for residual vs base.
4. **Edge behavior:** where does P_A hit zero vs P_B? Quantify how far the edge moves and
   how badly ΔlogP diverges there.

### Step 3 — deliverables
- A short script (saved under `tmp/` or `scripts/`, env-`test` runnable) that regenerates
  everything.
- A results note (markdown, under `data/results/` or `docs/`) with the numbers above and
  plots: base log P_A, ΔlogP_sub, and the three-region dynamic-range / smoothness summary.
- A one-paragraph verdict answering: **is the subhalo residual smaller in dynamic range and
  smoother in θ than the base — in the body, edge, and tail separately?** Yes/no/where.

## 5. What NOT to do
- Do NOT build the full multi-stage cascade or train any networks yet.
- Do NOT touch the `~/Desktop/halos` repo (user-exclusive).
- Do NOT commit/push. Report results; the user handles git.
- Do NOT quote far-tail / q≳99.9 numbers as converged — the method class isn't certified
  there; keep tail statements qualitative unless robustly estimated.

## 6. Follow-up (only if Step 1–3 look promising)
If the residual is demonstrably better-conditioned in the body: the likely right design is
**not** pointwise log-P residuals but residuals on the *structured/parametric* tail+edge
representation the production model already uses (σ(lnμ), tail index, edge location, flux
δ) — i.e. emulate how each ingredient shifts those parameters. Note this in the writeup as
the recommended next direction, but don't implement it in this task.
