# analytic/ — semi-analytic sGL magnification PDF vs the C++ engine

Implementation of the compound-Poisson (Lévy–Khintchine) framework of
`~/Downloads/sGL_analytic_draft.tex` (2026-07-27), tested end-to-end against the
production C++ Monte Carlo in halo-only scope. Physics chain inherited from the
validated `scripts/comparisons/analytic_pdf.py` / `analytic_chain_spec.md`
(caustic-resolved deposit-binned cross-section included); NEW here are the
Λ(k) exponent, the Fourier inversion to P(ξ), the engine-convention cosmology
mode, and the direct PDF overlay against the MC.

## Files

- `sgl.py` — the chain: `Cosmology` (background, EH98 transfer, σ(M), ST HMF,
  NFW/Dutton-Macciò) → `R_of_xi` (jump measure) → `tilt_source` (Esscher
  e^{−ξ}, source plane) → `Lambda_of_k` → `P_of_xi` / `dP_dmu`, plus exact
  moment formulas (`sigma_DL_over_DL`, `moments`).
  Two selectable convention modes:
  - `window="tophat", transfer="nowiggle"` — the spec / paper convention;
  - `window="smoothk", transfer="eh98"` — the ENGINE convention: σ₈ imposed
    through the smooth-k filter Ws (like `cosmology.cpp::sigmaC`) + full EH98
    with wiggles and the engine's z_eq-based k_eq (`OmegaR = Ω_M/(1+z_eq)` ⇒
    `k_eq = (H0/c)√(2Ω_M(1+z_eq))`). Reproduces the engine's σ(1e12) to
    2e-4 and its dn/dlnM grid points to ≤0.3%.
- `checkpoints.py` — holds `sgl.py` to (a) every `analytic_chain_spec.md`
  checkpoint (all pass; the R(ξ≳0.2) and σ_DL excesses over the SPEC table are
  the known 2026-07-24 caustic erratum — the spec's numbers are the flawed
  ones) and (b) the engine probe dump `tmp/engine_checkpoints.txt`.
- `run_mc.py` — halo-only C++ MC (filaments/bias/ell/subhalo OFF), raw
  (κ,γ₁,γ₂) per ray, subprocess-sharded seeds. 400k rays ≈ 10 s at z_s=1.
- `compare_pdf.py` — the overlay + metrics. Rebuilds `sample_lnmu` from raw
  rays (anchor selectable: `robust` = mean κ over κ≤1 rays, `batch` = engine
  legacy), source-plane weights 1/μ, and compares against the inverted P(ξ)
  and the mean-matched P(ξ−Δ) (Δ = MC's source-plane mean of lnμ).
- `data/` — MC npz + cached R(ξ) + `compare_*.json`; `figures/` — overlays.

## Result (2026-07-27, 400k rays/z_s, engine mode, robust anchor)

`figures/pdf_overlay_engine_robust.png`

| z_s | Var ratio MC/an | σ_DL ratio | N_lens an/eng | weak-band Var an/eng | JSD | JSD mean-matched |
|-----|------|-------|-------|-------|---------|---------|
| 0.5 | 0.973 | 0.987 | 1.040 | 1.000 | 8.5e-5 | 2.0e-4 |
| 1.0 | 0.957 | 0.978 | 1.031 | 1.009 | 3.4e-4 | 6.2e-4 |
| 2.0 | 0.952 | 0.973 | 1.008 | 1.032 | 9.2e-4 | 1.1e-3 |
| 5.0 | 0.945 | 0.964 | 1.036 | 1.015 | 2.2e-3 | 1.8e-3 |

- **Draft's validation claims reproduced**: lens count <4% (draft: <3%),
  σ_DL 1.3–3.6% (draft: 1–3%), and the engine's own sub-threshold Gaussian
  variance matched to ≤3% by ∫ξ²R below ξ=2κ_thr.
- **PDF overlay**: JSD 8e-5 → 2e-3 from z_s 0.5 → 5 (reference floor: 240k-shard
  sampling floors elsewhere in this repo are ~2–3e-4; emulator KL is 7e-3).
  Probability-weighted |MC/analytic − 1| is 2% at z_s=0.5, 11% at z_s=5
  (6.5% after mean-matching).
- **Residual anatomy** (grows with z_s, i.e. with optical depth — consistent
  with the draft's own "scalar reduction" caveat):
  1. *Mean convention*: engine anchors ⟨κ⟩=0 per batch; the compensated Lévy
     exponent sets ⟨ξ⟩=0. At z_s=5 the MC's source-plane mean is −0.015 vs
     analytic −0.002; mean-matching removes about half the body residual.
  2. *Scalar composition*: summing per-lens ξ_i vs forming ξ from summed
     (κ,γ). 4·Var(κ)_MC / Var(ξ)_an = 0.82 (z_s=1) → 0.74 (z_s=5): beyond
     leading order ξ ≈ 2κ the two differ at O(κ²), which the low-μ edge
     (deep void limit, many overlapping underdense cells) feels most — visible
     as the blue points' edge deficit at z_s=5.
  3. MC var sits 3–6% BELOW analytic: same sign and size as the spec's §6
     known behaviour (scalar chain slightly over-counts vs vector process at
     high optical depth).

**Verdict**: the framework holds at the draft's claimed accuracy in the body
and moments for z_s ≲ 2; at z_s = 5 the scalar reduction is visibly the
limiting approximation (edge + 5–6% variance), not the jump measure R(ξ)
itself (microscopic checks all ≤4%).

### ⟨N⟩ is NOT the cause of the z_s=5 residual (2026-07-27)

`test_nhalos.py` → `figures/nhalos_convergence_zs5.png`. Raising the engine's
explicit-lens budget by 10× (κ_thr 9.0e-4 → 1.1e-4, so the Gaussian
sub-threshold stand-in shrinks by an order of magnitude):

| ⟨N⟩ | κ_thr | Var MC/an | σ_DL MC/an | JSD | body \|Δ\| |
|-----|-------|------|-------|---------|-------|
| 100 | 9.0e-4 | 0.945 | 0.964 | 2.2e-3 | 11.4% |
| 300 | 3.4e-4 | 0.941 | 0.963 | 2.0e-3 | 10.7% |
| 1000 | 1.1e-4 | 0.950 | 0.967 | 1.8e-3 | 10.0% |

The three MC curves lie on top of each other; the ratio panel is unchanged.
The explicit/Gaussian split is already converged at the production ⟨N⟩=100 —
consistent with the weak-band variance check (analytic/engine = 1.015 at
z_s=5). The residual is intrinsic to the scalar reduction plus the mean
convention, and no engine setting removes it.

### Where the residual actually lives (2026-07-27)

`test_composition.py` → `figures/composition_vs_zs.png`. κ and γ are **exactly
additive** in both codes, so Campbell's theorem gives their variances from the
jump measure with no scalar reduction anywhere: Var(κ) = ∫κ²dR,
⟨|Σγ⃗|²⟩ = ∫γ²dR. Comparing those separates an error in R(ξ) from an error in
the κ→μ composition.

| z_s | N an/eng | Var(κ) an/MC | ⟨γ²⟩ an/MC, κ>κ_thr | ⟨γ²⟩ an/MC, full band | Var(lnμ) an/MC | σ_DL an/MC |
|-----|-------|-------|-------|-------|-------|-------|
| 0.5 | 1.017 | 1.030 | 1.030 | 1.068 | 1.027 | 1.013 |
| 1   | 1.017 | 1.042 | 1.031 | 1.099 | 1.045 | 1.023 |
| 2   | 1.016 | 1.050 | 1.030 | 1.149 | 1.051 | 1.027 |
| 5   | 1.016 | 1.034 | 1.030 | 1.241 | 1.058 | 1.037 |
| 7   | 1.018 | 1.045 | 1.033 | 1.279 | 1.068 | 1.044 |
| 8   | 1.017 | 1.039 | 1.032 | 1.292 | 1.067 | 1.045 |
| 10  | 1.017 | 1.039 | 1.031 | 1.309 | 1.070 | 1.048 |

Three separate things were conflated in the earlier reading:

1. **The jump measure R(ξ) is right, uniformly in z_s.** Lens count +1.7% and
   like-for-like ⟨γ²⟩ +3.0–3.3% are *flat* from z_s=0.5 to 10. Var(κ) ≈ +4% is
   just the count excess propagated (Var ∝ N for a Poisson sum). Nothing about
   the analytic degrades at high z_s.
2. **⚠ ENGINE finding — the sub-threshold arm has no shear.**
   `lensing.cpp:779-781` fills the Gaussian background with
   `kappa = PkappaW(mt)` but `gamma1 = gamma2 = 0`. Under the fixed-⟨N⟩=100
   rule κ_thr grows with z_s (3.8e-5 → 1.4e-3 over z_s 0.5→10), so the omitted
   band grows and the engine's ⟨γ²⟩ deficit reaches **31% at z_s=10**. This is
   an engine approximation, not an analytic error — restricting the analytic
   to κ>κ_thr collapses the ratio back to a flat 1.03. It barely moves P(lnμ)
   because γ enters ξ only at O(γ²), which is why the ⟨N⟩ scan below shows
   nothing, but it should be quoted if ⟨γ²⟩ is ever used directly.
3. **The scalar reduction is real but small**: Var(lnμ) drifts +2.7% → +7.0%
   over z_s 0.5 → 10, i.e. only ~3% on top of the flat additive floor, and it
   **saturates by z_s ≈ 5** (the z_s = 5/7/8/10 ratio panels are essentially
   identical). σ_DL likewise 1.3% → 4.8%.

The large-looking ±20% swing in the ratio panels is pointwise, on a steep
narrow distribution: about half is the mean-subtraction convention
(mean-matching drops body |Δ| from 11–14% to 6.5–7.5%), the rest is the
composition. No integrated quantity is off by more than 7%.

### Closing the budget: the gap was mostly the ENGINE's grid (2026-07-27)

Three targeted experiments, each isolating one candidate.

**(a) Background shear — NOT the cause.** `test_background_shear.py` measures
the shear the engine's weak arm omits, ⟨γ²_sub⟩ = ∫γ²dR(full) − ∫γ²dR(κ>κ_thr),
and injects it back into the MC rays as a 2D Gaussian. It closes only
**0.2 / 5.2 / 8.3%** of the Var(lnμ) gap at z_s = 1 / 5 / 10. Real but minor —
γ enters ξ only at O(γ²).

**(b) Scalar reduction, in isolation.** `test_scalar_reduction.py` draws BOTH
compositions from the SAME realizations of the analytic chain's own lens
population, so nothing else can differ. Var(scalar)/Var(vector):

| z_s | 1 | 5 | 10 |
|-----|---|---|----|
| ratio | 1.0004 | 1.0144 | 1.0220 |

So the scalar reduction is worth ~0% at z_s=1, +1.4% at z_s=5, +2.2% at z_s=10.
(The mean rows in that script's output compare two *anchoring* conventions,
not compositions — see the caveat there.)

**(c) ⚠ The dominant term was the engine's (NM, Nz) grid, not the analytic.**
At fixed κ_thr the engine's own expected lens count is still rising at its
production resolution, while the analytic's is converged:

| grid | engine ⟨N⟩ (z_s=5) | | analytic grid | analytic ⟨N⟩ |
|------|------|---|------|------|
| NM=Nz=100 (production) | 100.54 | | Nz=40, NM=48 | 102.18 |
| NM=Nz=200 | 102.42 | | Nz=80, NM=96 | 102.02 |
| NM=Nz=400 | 103.37 | | Nz=160, NM=160 | 101.98 |

Re-running the MC on the finer grid moves it onto the analytic:

| z_s | engine grid | Var(lnμ) an/MC | σ_DL an/MC | JSD | body \|Δ\| |
|-----|------|------|------|------|------|
| 1 | NM=Nz=100 | 1.045 | 1.023 | 3.5e-4 | 4.1% |
| 1 | NM=Nz=400 | **1.008** | **1.005** | 2.8e-4 | 4.1% |
| 5 | NM=Nz=100 | 1.058 | 1.037 | 2.3e-3 | 11.4% |
| 5 | NM=Nz=400 | **1.024** | **1.021** | 2.2e-3 | 11.4% |

**Final budget for the z_s=5 Var(lnμ) gap (+5.8% total):** engine grid
resolution +3.4%, scalar reduction +1.4%, shear-free weak arm +0.3%,
unexplained ~0.7%. At z_s=1 the gap collapses to +0.8% once the engine grid is
converged — i.e. essentially all of it was the engine.

Note the JSD and body |Δ| barely move under (a)–(c): the ratio-panel *shape*
is set by the mean-subtraction convention, which is independent of every
variance term above.

**Implication for the wider project:** the production engine's Nz=NM=100 grid
under-counts lenses by ~3% at z_s=5, worth ~2.7% in Var(κ) and ~1.6% in σ_DL.
This refines CLAUDE.md item 12 ("Nz=100 converged") — the JSD verdicts there
are unaffected (JSD moves only 2.3e-3 → 2.2e-3), but *variance-level* claims at
z_s ≳ 5 carry this bias.

### Cost (z_s=1, engine mode, single core)

| stage | time |
|-------|------|
| `Cosmology` setup (once per θ) | 0.04 s |
| `R_of_xi` per z_s | 3.9 s |
| Fourier inversion to P(ξ) | 7.2 s |
| moments / σ_DL from an existing R | 0.1 ms |

≈11 s per (θ, z_s) for a full PDF — ~10³× faster than the 400k-ray MC, but
~10⁴–10⁵× slower than an ML emulator evaluation.

## Repro

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python
$PY analytic/checkpoints.py                       # spec + engine checkpoints
$PY analytic/run_mc.py --zs 1.0 --nray 400000     # per z_s (0.5/1/2/5 done)
$PY -W ignore analytic/compare_pdf.py --zs 0.5 1.0 2.0 5.0 --mode engine
```

Grids: `R_of_xi` Nz=40, NM=48, Mmax=1e16 — converged to 0.02% in Var/σ_DL
against Nz=120, NM=120, Mmax=1e17 (checked at z_s=1 and 5, 2026-07-27).
