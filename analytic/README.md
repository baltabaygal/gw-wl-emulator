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
