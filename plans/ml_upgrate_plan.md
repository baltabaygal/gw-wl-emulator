# 1+6d parameterization: {h, z_eq, Ω_M, Ω_B, A_s, n_s} as runtime inputs

## Context

The emulator currently conditions on a 1+3d space (z_s; h, Ω_M, σ₈). The goal is to promote
the full cosmology vector **Θ = {h, z_eq, Ω_M, Ω_B, A_s, n_s}** to runtime inputs of the C++
simulator (with A_s as the primary amplitude parameter, σ₈ kept as an alternative mode), and
update the Python/ML plumbing so the pipeline is *ready* to train on the 1+6d space.
**No data regeneration or retraining in this round** (user decision). Ω_Λ and w(DE) deferred.

**Good news from exploration:** `OmegaB`, `zeq`, `T0`, `ns` are already runtime members of
`CosmologyParams` (cpp/lnmu_wrapper.h:6–24) and flow dynamically into the Eisenstein–Hu
transfer function (cpp/cosmology.cpp:4–50: `keq` from `Hz(zeq)`, `ksilk`/`Rd` from `OmegaB·h²`,
`T0`) and into `Δ²(k) ∝ k^(3+ns)` (cosmology.h:117–122). They're just not exposed through
pybind. Only **A_s is missing entirely**: normalization is
`deltaH8 = sigma8/sigmaC(M8, 1.0)[0]` (cosmology.h:310).

**Is 1+6d OK for ML?** Yes. The NSF conditioning dim (4→7) is architecturally trivial; sample
complexity is governed by the smoothness of the PDF response, not dimension per se, and the
three new params (Ω_B, n_s, z_eq) have weak, smooth effects on P(μ) (mild T(k) shape changes);
A_s is the dominant one and acts like σ₈². Expect to need ~2–4× more training configs
(487 → ~1000–2000) mainly so the ridge/Poisson edge & tail fits stay constrained with the
extended feature basis. Condition on **ln(10¹⁰A_s)**, not raw A_s.

## Physics: A_s → δ_H mapping (analytic, no new integrals)

The code uses a Bunn–White-style amplitude: `Δ²(k,z) = (ck/H₀)^(3+ns)·(δ_H·T(k))²·Dg(z)²`
with `Dg(z) = [g_CPT(z)/(1+z)] / 0.7869370293916` (cosmology.h:58–60; the constant is
g_CPT(z=0) at the fiducial cosmology, so `D(z)=g/(1+z)` → a in matter domination).
Equating to the standard relation `Δ²_m(k,z) = (4/25)·A_s·(ck/H₀)⁴·(k/k_p)^(ns−1)·T²(k)·D²(z)/Ω_M²`:

```
deltaH8 = (2/5) · g_fid · sqrt(As) · (c·k_p/H0)^((1−ns)/2) / OmegaM
```

with `g_fid = 0.7869370293916` (the constant already in `Dg`), `c·k_p/H0 = 306.535·kpivot/H0`
in code units, `kpivot = 0.05 Mpc⁻¹ = 5.0e-5 kpc⁻¹` (comoving; k in Deltak is comoving kpc⁻¹,
H0 = 0.000102247·h). Everything downstream (σ(M) tables, HMF, Mchar) flows through `deltaH8`
unchanged. Inverse mapping gives a derived A_s in σ₈-mode and a derived σ₈ in A_s-mode
(via `deltaH8·sigmaC(M8,1.0)[0]`) — expose both for diagnostics.

Caveats to document in code comments: (1) the code's σ₈ uses a smooth k-space window
`Ws` (cosmology.cpp:41–45), not a tophat, so derived σ₈ vs CAMB agrees only to a few %;
(2) freeing z_eq frees the radiation density (`OmegaR = OmegaM/(1+zeq)`, cosmology.h:293) —
that's the intended parametrization; (3) CPT growth ignores radiation (as before).

## Phase A — C++ core

1. **cpp/lnmu_wrapper.h** `CosmologyParams`: add `double As = -1.0;` (≤0 ⇒ σ₈-mode) and
   `double kpivot = 5.0e-5;` (kpc⁻¹). Keep `sigma8`. A_s > 0 takes precedence.
2. **cpp/cosmology.h**: add members `As`, `kpivot`; hoist the literal `0.7869370293916` into a
   named constant `gfid` used by both `Dg` and the new normalization. In `initialize0()`
   (line ~310) branch:
   ```cpp
   if (As > 0.0) deltaH8 = 0.4*gfid*sqrt(As)*pow(306.535*kpivot/H0,(1.0-ns)/2.0)/OmegaM;
   else          deltaH8 = sigma8/sigmaC(M8, 1.0)[0];
   sigma8_derived = deltaH8*sigmaC(M8, 1.0)[0];
   As_derived = pow(deltaH8*OmegaM/(0.4*gfid), 2.0) * pow(306.535*kpivot/H0, ns-1.0);
   ```
3. **cpp/lnmu_wrapper.cpp**: copy `As`, `kpivot` into the `cosmology C` object at every unpack
   site (same places `OmegaB/zeq/T0/ns` are copied, ~line 209–262 and siblings).
4. **cpp/python_bindings.cpp**: add trailing kwargs to *all* entry points (`sample_lnmu`,
   `sample_lnmu_ml`, `sample_lnmu_ml_with_diagnostics`, `sample_lensing_raw_ml`,
   `compute_lnmu_stats`, …): `As = -1.0, OmegaB = 0.0493, zeq = 3402.0, ns = 0.965`
   (existing positional args unchanged → full backward compat). Extend
   `get_simulator_config` to accept the same params and report `deltaH8`,
   `sigma8_derived`, `As_derived`, `OmegaR`.
5. Rebuild: `make build` (against the `test` env Python 3.12).

## Phase B — C++ tests (tests/, new test_cosmology_params.py)

- **Backward compat (bitwise):** default-kwarg call with fixed seed reproduces samples
  generated *before* the change (capture a reference vector first, before editing).
- **σ₈↔A_s round trip (bitwise):** run σ₈-mode, read `As_derived` from
  `get_simulator_config`, rerun in A_s-mode with it, same seed ⇒ identical lnμ samples
  (equal `deltaH8` ⇒ identical RNG stream).
- **Planck sanity:** As=2.101e-9, ns=0.9649, h=0.674, Om=0.315, Ob=0.0493, zeq=3402 ⇒
  derived σ₈ ∈ [0.75, 0.87] (EH fit + smooth-window tolerance). If far outside, hunt a
  convention bug before proceeding.
- **Exact scaling:** derived σ₈ ∝ √A_s (two A_s values).
- **Smoke:** perturbing ns / OmegaB / zeq individually changes `deltaH8`/samples.
- `make pytest` — all 75 existing tests must still pass.

## Phase C — Python/ML plumbing (this repo, current branch)

6. **New `ml/params.py`** — single source of truth, imported everywhere:
   - `FIDUCIAL = dict(h=0.674, Om=0.315, As=2.101e-9, Ob=0.0493, ns=0.965, zeq=3402.0)`
   - `PRIOR_6D` (wide box, user-approved): h (0.59, 0.76), Om (0.20, 0.40),
     A_s (1.2e-9, 3.6e-9) — sampled/conditioned as ln(10¹⁰A_s) ∈ [2.485, 3.584],
     Ob (0.035, 0.065), ns (0.90, 1.02), zeq (2500, 4500).
   - Context layout `CONTEXT_KEYS = ("z", "h", "Om", "lnAs10", "Ob", "ns", "zeq_k")`
     (z stays index 0 — train_smooth.py assumes it; zeq_k = zeq/1000).
   - Helpers: `theta_to_context(z, **params)`, `sample_prior(rng, n)`.
7. **python/generate_dataset.py**: sample the 6d prior; worker passes
   `As=…, OmegaB=…, ns=…, zeq=…` kwargs (sigma8 arg left at default, ignored when As>0);
   HDF5 stores keys `z, h, OmegaM, As, OmegaB, ns, zeq` (+ derived sigma8 as metadata);
   ID/OoD partition (lines 60–68) re-expressed with the lnAs10 box in place of σ₈.
8. **ml/data.py**: `flatten_dataset` → X shape (N, 7) per `CONTEXT_KEYS`;
   **ml/nsf_model.py / ml/train_nsf.py**: `context_dim` 4→7 (make it derive from
   `len(CONTEXT_KEYS)`).

## Phase D — autoresearch plumbing (~/Desktop/gw-wl-emulator-ar worktree; edit, don't run)

9. **train_ar.py** (`context=4` at line 76 → `len(CONTEXT_KEYS)`; `make_log_prob_fn` context
   assembly), **train_smooth.py** (index-0-is-z assumption stays valid).
10. **smooth_model.py / prepare_fix.py / fit_tail_amplitude.py**: generalize
    `edge_features` (10-elem) and `tail_features` (14-elem) to the 6-param context. Proposed
    basis (a = lnAs10, ze = zeq_k, lz = log1p(z)): backbone `[1, lz, lz², lz³]` + linear
    `[h, om, a, ob, ns, ze]` + z-mixed `[·lz for each]` + amplitude interactions
    `[a², om·a, a²·lz, om·a·lz]` (a inherits σ₈'s interaction terms since A_s ~ σ₈²).
    Coefficients refit at retrain time — this round only changes the feature functions and
    keeps them defined in one place (import from `ml/params.py` side module or duplicate
    consistently between fit and inference as now, but generated by one shared function).
11. **gen_lowz_data.py, gen_tail_counts.py, param_space_check.py, validate_kl/pit/posterior.py,
    prepare_ar.py**: replace (h, Om, s8) tuples with 6d thetas built from `FIDUCIAL` —
    COSMOS panels vary (h, Om, A_s), hold (Ob, ns, zeq) at fiducial; validate_posterior keeps
    its (h, Om) grid with the rest fixed to truth. Update `PRIOR` dicts to `PRIOR_6D`.

## Verification (end-to-end)

1. Pre-change: capture reference lnμ vectors (fixed seeds, a few (z,Θ) points) to a scratch
   file for the bitwise backward-compat test.
2. `make build && make pytest` in the `test` env — 75 old + new tests pass.
3. Round-trip + Planck-σ₈ checks (Phase B) pass.
4. PDF overlay plot: A_s-mode at Planck vs σ₈-mode at σ₈=0.811, same seed, zs=1 — curves
   indistinguishable (→ plots/).
5. 6d plumbing smoke: `generate_dataset.py` with ~5 configs × 1k samples on the 6d prior →
   tiny HDF5; `ml/data.py` flatten gives (N,7); 1-epoch `train_ar.py` run on it confirms the
   context=7 flow trains and `make_log_prob_fn` evaluates.

## Out of scope (this round)

- Data regeneration + retraining (recipe in CLAUDE.md §NSF; needs the new datasets first).
- Ω_Λ ≠ 1−Ω_M−Ω_R and w(DE): would enter `Az()` (cosmology.h:41) and require replacing the
  CPT growth approximation with the growth ODE — noted for later, per user.
- The ~/Desktop/halos repo port (user pushes halos; separate task).
- T0 stays struct-level (settable in C++, not part of Θ).