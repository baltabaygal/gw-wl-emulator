# Subhalo combining process — design

Adds substructure to the `halos` lensing pipeline with minimal interference.
OFF by default; when off, **no RNG draws change**, so existing results reproduce exactly.
Analytic backbone (Poisson/Campbell variance, cluster-process corrections, the top-heavy
m^{+0.5} mass scaling behind the m_floor/subhalo_factor cutoffs):
`docs/variance_derivation.md`.

## κ/γ definition (agreed)
Each lensing halo becomes a **reduced smooth host + its discrete subhalos**, conserving mass:
```
κ_halo = κ_host((1−f_s)·M)  +  Σ_i κ_sub(m_i),     ⟨Σ_i m_i⟩ = f_s·M
```
The reduction is essential: the HMF mass `M` is the *total* (inclusive) halo mass, so the
subhalos are carved out of it, not added on top. Skipping the reduction injects ~f_s·M ≈ 20%
spurious mass per halo and biases the variance/skewness at order f_s. `meankappa` (taken
downstream in `sample_lnmu`) removes the mean; the reduction fixes the higher moments.

Each subhalo's κ,γ is added at the **2D ray→clump separation** `d = |r⃗_LoS − R⃗_clump|`,
with ε=0 (circular clumps), using the **single-angle γ projection (cos φ_d, sin φ_d) of
the original Vaskonen code** — kept by decision (2026-07-02, supervisor preference for
consistency with the original approach). For the record: the spin-2 form is
(γ1,γ2) = −γ_t(cos 2φ, sin 2φ); a convention-free Jacobian test
(`tmp/shear_convention_check.py`) shows the single-angle form is exact for independent
lens angles, and off by ~0.5% on ⟨γ²⟩ (≲1% on |γ| quantiles) for correlated host+clump
configurations at representative amplitudes. Revisit only if shear statistics become a
primary observable.

## Per-host subhalo pipeline (precomputed on the (M,z) grid)
1. `z_f`  — Giocoli+2007 (JvdB14 eq. 25): δc(z_f)=δc(z0)+w̃_f√(σ²(M/2)−σ²(M)), w̃_f=√(2ln(α_f+1)), α_f=0.815 e^{−2f³}/f^0.707 = 0.815 e^{−1/4}/0.5^0.707 at f=½ (⇒ w̃_f≈1.19; **fixed 2026-07-02** — was mistyped as e^{−1}, overestimating f_s by ~22–26%).
2. `N_τ`  — eq. 24, reduces to a pure z-integral (Hz cancels): N_τ=∫_{z}^{z_f} 6.006·(Δvir/178)^{1/2}/(1+z′) dz′, Δvir=Bryan-Norman.
3. `f_s`  — eq. 26: f_s=0.3563/N_τ^{0.6}−0.075 (mass fraction is top-heavy ⇒ ~floor-independent).
4. `γ`    — eq. 23 (GSL incomplete Γ), anchored to f_s at ψ_res=1e-4.
5. SHMF   — dN/dlnψ=γψ^α e^{−βψ^ω}, (α,β,ω)=(−0.82,50,4), ψ=m/M. Over the clump range
            ψ∈[m_floor/M, 0.1] the exponential ≈ 1, so masses are sampled as a **pure
            power-law** (analytic inverse-CDF) — no per-bin mass table needed.
6. `N_sub` — ∫ dN/dlnψ above the floor (number is bottom-heavy ⇒ floor-dependent; Poisson mean).
7. reduced host NFW at (1−f_s)M; host r200 and c for the radial profile.

## Spatial profile (user-specified, anti-biased)
Subhalo number density per shell ∝ x²/(1+c x)² · B(x), x=r/r200,
B(x)=1/√((x/0.54)^{−2.5}+1)  (central depletion; →1 at large x).
Per active bin, the inverse radial CDF x(u) is tabulated (128 pts); a clump's 3D radius is
r=x·r200, then projected to 2D (random direction).

## Resolution
Two modes (both `subhalo_model=1`, the reduced-host option B):
- **`subhalo_brute=True`** — every clump above `m_floor = 1e7` is drawn. Ground truth;
  thousands+ of clumps for massive hosts, runtime grows accordingly.
- **dynamic split (default)** — only clumps whose reach r_thr(m) exceeds the **host-center
  distance r** are drawn (floor = inverse of the monotone r_thr table at r). This
  under-resolves clumps that land closer to the ray than r; the bias is absorbed by
  lowering `subhalo_factor` (rescales the clump κ_thr) until Var(κ) converges to brute —
  `scripts/subhalo_factor_convergence.py`. A worst-case max(0, r−r200) floor is NOT usable
  instead: NFW κ diverges on-axis, so it degenerates to brute for every ray inside r200
  (found 2026-07-02 when a first version of that criterion made `subhalo_factor` inert).
  Unresolved subhalos stay in the reduced smooth host, so mass is conserved per sightline —
  the same ψ_lo is used for the host reduction (`lensing.cpp`) and the clump draw
  (`addClumps`); keep them in sync.

## Code touchpoints
- new: `cpp/subhalo.{h,cpp}` (class `Subhalo`: `precompute(cosmology&)`, `addClumps(...)`).
- `lensing.h`: `LensingConfig{ bool subhalo; double m_floor; }`; declare NFW kernels; `Subhalo` member.
- `lensing.cpp` `sample_lnmu_raw`: build tables if subhalo on; per-bin use reduced host; call `addClumps` in both host-add branches.
- `python_bindings.cpp`: expose `subhalo`, `m_floor`.
- `cpp/CMakeLists.txt`: add `subhalo.cpp`.

## Pre-publication reassessment (2026-07-03)
Independent re-review of the current tree (`cpp/subhalo.{h,cpp}`, `cpp/lensing.cpp`) by
Claude + codex (read-only), plus numerical invariants (`tmp/reassess_invariants.py`):
- **Mass conservation EXACT.** The fraction removed from the host in `lensing.cpp`
  (`f_s_res`) and the clump count/masses drawn in `addClumps` use the *same* `psi_lo`
  (same `r_thr` lookup); ⟨Σ m_clump⟩ = f_s_res·M analytically (ratio 1.00000) and by MC
  (0.999). Power-law inverse-CDF for clump mass and the anti-biased inverse radial CDF
  both verified correct for α<0.
- **Invariants pass:** subhalo=OFF bit-reproducible and leaves κ_nosub≡κ; subhalo=ON
  deterministic at fixed seed; flux invariant ⟨1/μ⟩ = 1.001 both ON and OFF (substructure
  does not break normalization).
- **Fix applied:** invalid `subhalo_model` (∉{0,1}) now throws instead of silently adding
  clumps without reducing the host (would inject ~f_s·M). Published runs use model=1.
- **Non-issues for the published config** (model=1, threads=1, dynamic floor or brute@1e7),
  documented not to block: (i) host-on-ray r=0 is prob 2⁻⁶⁴ and now clamped on both host
  (`kappagammaNFWeps` xeps floor) and clump (`x_c` floor) paths; (ii) legacy model=0
  `gslope` is unclamped at r=0 but non-default; (iii) `subhalo_threads>1` uses independent
  RNG streams → not bit-reproducible across thread counts (default =1); (iv) Poisson int
  overflow only at unphysically low brute `m_floor`. See codex log for details.

## Validation (2026-07-02; config = original Vaskonen conventions + corrected w̃_f)
- host-only ⟨κ²⟩ = 7.23e-3 (zs=5, 40k) matches the analytic screen's host term (7.234e-3)
  to 0.1%.
- **Dynamic split vs brute** (excess = var(κ)−var(κ_nosub), paired control variate,
  bootstrap errors). Two scans, 2026-07-03: `scripts/subhalo_factor_convergence.py`
  (zs=1, 40k, brute reference) and `scripts/subhalo_factor_redshift_check.py`
  (zs=0.5–10, factors 1e-5…1e-2; z=10 checked down to 1e-6). The plateau is z-dependent —
  1e-4 suffices at zs≤2 but sits ~7–8% low at zs=5–10 — so **default fixed at
  `subhalo_factor = 1e-5`, the cross-redshift plateau** (flat vs 3.16e-6/1e-6 at z=10;
  ~100–140 s per 40k, ~20× cheaper than brute). Old default 1.0 captured only ~14% of the
  substructure variance at zs=1. **Gap closed 2026-07-04** (`tmp/z1_tail_scan.py`,
  `data/subhalo_factor_z1_tail.csv`; fit by codex in `tmp/fit_factor_convergence.py` +
  `_report.txt`): deep tail f=3.16e-6…1e-7 (4 points) rides the brute band; brute pooled
  over 3 seeds = 6.35±0.22e-5 (the earlier single seed-42 brute, 6.77e-5, was a high
  fluctuation); recruitment-model extrapolation E_inf agrees with pooled brute to +0.14σ
  (χ²/dof=0.51); model-free f≤1e-5 pool −0.77σ. At the adopted 1e-5 the single point is
  89±6% of pooled brute (−1.8σ); the f≤1e-4 plateau mean is 96%. Paper figure:
  `scripts/plot_subhalo_factor_paper.py` →
  `plots/figures/subhalo_factor_convergence_paper.{pdf,png}`.
  Older text: residual vs brute at the plateau = near-ray clumps of far hosts
  (recruited only as f^(−1/3)). High factors give a significantly NEGATIVE excess
  (−1.3e-5 at f=10): smooth-swap softening — the host loses central Σ but the Han+16
  anti-biased clump profile returns less mass near the ray. `m_floor` sensitivity
  (1e7→1e5, zs=1): flat — the 1e7 floor is deep in the convergent regime
  (dVar/dlnm ∝ m^{+0.5}, variance is top-heavy in subhalo mass).
  Data: `data/subhalo_factor_convergence_z1.npz`, `data/subhalo_factor_redshift_check.*`,
  `data/subhalo_z10_low_check.*`, `data/subhalo_mfloor_sensitivity_z1.npz` + PNGs in
  `plots/figures/`. TODO: tail quantiles (q99/q99.9) alongside variance;
  Nhalos=100→300 probe of unsampled-host clumps.
- NOTE: an earlier "94% captured / 11× faster" claim here was measured with the broken
  max(0, r−r200) floor (effectively brute-vs-brute) — retracted.

**Do NOT compare the MC Δ⟨κ²⟩ directly to the screen's `dK2c`.** The screen's clump term is
the *unclustered Poisson component only* (full-plane Campbell integral, host at full M).
The MC additionally contains within-host clump clustering and host–clump covariance
(positive, dominant) and the (1−f_s)M host reduction (negative) — at zs=5 the full MC
effect is +14.9% on ⟨κ²⟩ vs the screen's Poisson component +3.4%. An earlier version of
this section claimed they should match; that was wrong.

Screen fiducials with the corrected w̃_f: zs=5: Δ⟨κ²⟩_c/⟨κ²⟩=3.43%, Δ⟨κ³⟩_c/⟨κ³⟩=1.45%;
zs=1 (gate fiducial): 3.16%, 1.45% (`dK2c, dK3c = 2.307e-5, 1.077e-6` in `subhalo_gate.py`).

## Deferred
BMO-truncated clump profile + subhalo c(m,z);
eq.-4 elliptic-collapse host MF (`papers/misc/2504.20043v4.pdf`); merger-tree clump clustering.
