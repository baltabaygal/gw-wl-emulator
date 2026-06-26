# Subhalo combining process — design

Adds substructure to the `halos` lensing pipeline with minimal interference.
OFF by default; when off, **no RNG draws change**, so existing results reproduce exactly.

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
mirroring the existing γ projection (`cos φ_d, sin φ_d`), with ε=0 (circular clumps).

## Per-host subhalo pipeline (precomputed on the (M,z) grid)
1. `z_f`  — Giocoli+2012 (JvdB14 eq. 25): δc(z_f)=δc(z0)+w̃_f√(σ²(M/2)−σ²(M)), w̃_f=√(2ln(α_f+1)), α_f=0.815 e^{−1}/0.5^0.707.
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
Brute-force to `m_floor = 1e7` for now (no resolved/unresolved split — optimization deferred).
This means thousands of clumps for massive hosts; runtime grows accordingly (accepted).

## Code touchpoints
- new: `cpp/subhalo.{h,cpp}` (class `Subhalo`: `precompute(cosmology&)`, `addClumps(...)`).
- `lensing.h`: `LensingConfig{ bool subhalo; double m_floor; }`; declare NFW kernels; `Subhalo` member.
- `lensing.cpp` `sample_lnmu_raw`: build tables if subhalo on; per-bin use reduced host; call `addClumps` in both host-add branches.
- `python_bindings.cpp`: expose `subhalo`, `m_floor`.
- `cpp/CMakeLists.txt`: add `subhalo.cpp`.

## Validation
With features OFF + subhalo ON, the C++ Δ⟨κ²⟩/⟨κ²⟩, Δ⟨κ³⟩/⟨κ³⟩ must reproduce the analytic
screen (+3.9%, +1.8%). See `memory/subhalo_gate_verdict.md`.

## Deferred
Resolved+unresolved hybrid (speed); BMO-truncated clump profile + subhalo c(m,z);
eq.-4 elliptic-collapse host MF (`papers/misc/2504.20043v4.pdf`); merger-tree clump clustering.
