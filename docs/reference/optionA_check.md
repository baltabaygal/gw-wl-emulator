# Option A — external validation via the substructure convergence power spectrum

Status: **Layer 1 (internal exactness) PASSED, 2026-07-04. Layer 2 (external overlay vs
DRCD18) PASSED, 2026-07-04** — agreement to a few % at the variance-dominating scales,
Var(κ) ratio 1.09; the only deviation is the expected untruncated-NFW low-k excess,
now quantified and shown to be irrelevant to the κ-variance. Direct ETHOS overlay
remains optional polish (covered indirectly: DRCD18's model was validated against the
ETHOS N-body suite in their companion paper).

## What Option A is

Validate the subhalo implementation against the substructure convergence power spectrum
P_sub(k) formalism of Díaz Rivero, Cyr-Racine & Dvorkin 2018 (PRD 97, 023001,
arXiv:1707.04590) and its N-body measurement companion (Díaz Rivero et al. 2018,
PRD 98, 103517, arXiv:1809.00004; ETHOS suite). P_sub(k) is the Fourier face of exactly
the quantity our framework computes — the Campbell variance is ∫ P_sub(k) k dk / 2π —
so this compares our emergent lensing statistic against an independently derived
formalism anchored to simulations.

Two layers:
1. **Layer 1 — internal exactness:** our MC clump populations, pushed through a
   point-process P_sub(k) estimator, must reproduce our own analytic
   Campbell-in-Fourier prediction. Tests the machinery; no external content.
2. **Layer 2 — external overlay:** compare our analytic P_sub(k) against DRCD18's
   published curves (rescaled to a common mean substructure convergence κ̄_sub) and the
   ETHOS simulation band. This is the external validation.

## Layer 1 setup

- Script: `tmp/psub_check.py`; data: `tmp/psub_check.npz`.
- Population spec == production model (verified equivalent to `cpp/subhalo.cpp` by the
  mass-conservation MC of 2026-07-04): ψ_max = 1 SHMF with exp(−βψ^ω) via thinning,
  corrected Giocoli w̃_f, Han+16 anti-biased radial profile, cons14 concentrations,
  untruncated-NFW clump κ profiles (2D FT by numerical Hankel transform).
- Host: M = 10¹³ M⊙ at z_l = 0.5, source z_s = 2; g = 0.040, f_s = 0.139,
  r200 = 378 kpc; m_floor = 10⁷ M⊙ (brute; ⟨N⟩ ≈ 4045 per host).
- Aperture: annulus 5–40 kpc; estimator: direct mode sum over clump positions,
  **connected part** (complex mean field subtracted per (k, azimuth) — the smooth radial
  gradient of substructure density across the annulus otherwise leaks into P̂);
  1200 realizations, 8 azimuths.

## Layer 1 results

Aperture mean count: MC 89.06 vs analytic 88.42 (**0.7%**).

| k [1/kpc] | P_MC [kpc²] | P_analytic | ratio |
|---|---|---|---|
| 0.18 | 1.994e-3 | 1.841e-3 | 1.08 |
| 0.26 | 6.98e-4 | 6.62e-4 | 1.06 |
| 0.54 | 8.31e-5 | 8.11e-5 | 1.03 |
| 0.77 | 2.86e-5 | 2.79e-5 | 1.02 |
| 1.12 | 9.09e-6 | 9.16e-6 | 0.99 |
| 1.6–10 | — | — | 0.75–1.32 (scatter) |

Residuals are understood estimator artifacts, not model errors:
- **Low-k (+5–8%):** window leakage — a steep (red) spectrum convolved with the finite
  annulus window (Δk ≈ 2π/35 kpc⁻¹ ≈ 0.18) biases the lowest bins upward; the excess
  decays with k exactly as observed and vanishes by k ≈ 1.
- **High-k scatter:** heavy-tailed MC noise dominated by the few most massive clumps;
  the azimuth-based error bars underestimate there (azimuths of one realization are
  correlated). Paper version: bootstrap over realizations.
- Clean band k ≈ 0.5–1.2 kpc⁻¹: agreement to ~2%.

**Layer-1 verdict: PASSED.** The population sampler → P_sub(k) machinery is internally
exact at the level the estimator permits.

Debugging record (both artifacts of the v1 checker, caught and fixed): a coarse radial
grid inside the aperture biased the analytic count by 9%; the v1 estimator omitted the
mean-field subtraction, inflating low-k power by up to 2.2×.

## Layer 2 — external overlay vs DRCD18 (DONE 2026-07-04)

Script: `tmp/psub_overlay.py`; data: `tmp/psub_overlay.npz`; figure:
`plots/figures/psub_overlay_drcd18.png`.

**Conventions pinned from the paper** (eqs. 4, 28, 44, 50): mass-normalized profile FT
κ̃(k→0)=1; P_1sh(k) = κ̄_sub/(⟨m⟩Σ_cr)·∫dm m²P_m(m)|κ̃(k,m)|²; fiducial z_l=0.5→z_s=1,
κ̄_sub=0.02, dN/dm ∝ m^−1.9 on [10⁵,10⁸] M⊙, tNFW (their eq. 31) with τ=r_t/r_s=15,
r_s = 0.1 (m/10⁶)^{1/3} kpc; 2-subhalo term neglected (subdominant per their §III C, and
zero in our Poisson-occupancy model).

**Reproduction of their model — all published self-check targets hit:**
Σ_cr = 3.00e9 M⊙/kpc² (they quote ~3e9); plateau 1.215e-4 kpc² vs their quoted 1.2e-4
(and g₀=κ̄_sub m_eff/Σ_cr = 1.218e-4, m_eff=⟨m²⟩/⟨m⟩=1.83e7); k_trunc = 0.144 vs 0.14;
k_scale = 21.5 vs 21.5; high-k slope approaching −4. The comparison machinery is faithful.

**Our model at matched window and κ̄_sub** (untruncated NFW, cons14 c(m,z), slope −1.82;
profile extent regulated at x_max=200 r_s, sensitivity checked with x_max=50):

| k [1/kpc] | ours/DRCD18 |
|---|---|
| 0.01–0.07 | 3.4–6.6 (regulator-sensitive: untruncated outskirts) |
| 0.17 | 2.1 |
| 0.42 | 1.34 |
| 1.1–18 | **0.98–1.10** |
| 46 | 1.35 (both entering the k⁻⁴ asymptote) |

**Bottom line for our observable:** Var(κ) = ∫P k dk/2π agrees to **+9%**
(2.27e-5 vs 2.08e-5), and is regulator-independent to 0.1% — the variance integrand
peaks at k ≈ 3.2 kpc⁻¹, deep in the agreeing band. The untruncated-NFW excess is
confined to k ≲ 0.3 kpc⁻¹ where it contributes negligibly to the ray variance. The +9%
combines the SHMF-slope difference (m_eff: 2.15e7 vs 1.83e7) and profile-relation
differences (cons14 gives r_s(10⁶ M⊙) = 0.077 kpc vs their 0.1), partially compensating.

**Verdict:** our subhalo model's emergent lensing power spectrum agrees with the
independently derived, simulation-validated DRCD18 formalism to ≤10% on the κ-variance
and to a few % at the scales that dominate it — comfortably inside the factor ≲2 band
that formalism itself achieves against the ETHOS N-body measurements (arXiv:1809.00004).
External chain: this work ≈ DRCD18 (direct, here); DRCD18 ≈ ETHOS sims (their companion
paper) ⇒ consistent with N-body at the formalism's own validation level.
**Truncation systematic (paper text):** neglecting tidal truncation affects P_sub only
at k ≲ 0.3 kpc⁻¹ and changes the substructure κ-variance by <10% at fixed κ̄_sub — the
deferred BMO profile is not required for the current observable.

Optional polish (not blocking): direct overlay of the ETHOS measured band; repeat at a
GW-relevant host mass (10¹³–10¹⁴ M⊙) — the formalism is host-mass-agnostic at fixed
κ̄_sub and mass window, so no change is expected.

## Caveats

- Layer 1 uses a numpy replica of the sampler *specification*; the C++ ↔ spec
  equivalence is established separately (mass-conservation MC, invariant suite).
  Optional belt-and-braces upgrade: a small C++ utility dumping (m, x, y) clump catalogs
  from `Subhalo::addClumps` itself, re-running the estimator on real production draws.
- P_sub(k) validates the *lens-plane* substructure statistics. The LOS accumulation and
  P(μ) machinery are covered by the host-layer analytic screen and the convergence
  studies (`docs/subhalo_combining.md`, `docs/variance_derivation.md`).
