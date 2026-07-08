# Vaskonen 2026 — Weak lensing of bright standard sirens: prospects for σ₈

> **TL;DR (read-before-you-open):** Shows that the *scatter* in GW standard-siren
> distances from weak lensing carries cosmological information — a 10% measurement of
> σ₈ is feasible with ~300 ET neutron-star binaries (30% with 12 LISA MBHBs). Introduces
> the C++ Monte-Carlo that computes dP/dμ by summing discrete NFW halos + filaments along
> the line of sight. **This is the engine this whole repo emulates.**

| | |
|---|---|
| **Authors** | Ville Vaskonen (Padova / INFN / Tallinn) |
| **Year** | 2026 · MNRAS 547, 1–7 |
| **arXiv** | [2601.06023v2](https://arxiv.org/abs/2601.06023) |
| **Category** | core |
| **Relevance** | ⭐⭐⭐⭐⭐ — the paper this project is built on |
| **Read status** | reference (keep open) |
| **Code** | https://github.com/vianvask/halos |

## Read this if…
- You need the exact definition/derivation of the magnification PDF dP/dμ being emulated.
- You're touching the C++ engine (halo + filament LOS model, mass functions, κ/γ).
- You want the science target: which cosmological params (h, Ω_M, σ₈) the scatter constrains.

## Skip / defer if…
- You only need ML/emulator mechanics (see methods/) — the physics here is upstream of that.

## What it does
Computes the lensing magnification distribution dP/dμ with the stochastic (discrete-lens)
approach of Kainulainen & Marra (2009, 2011), implemented in C++. The matter along the LOS
is modelled as an ensemble of discrete lenses in redshift slices; per realization, lenses
are placed near the source and the total magnification is summed under the weak-lensing
approximation. Repeating over many realizations builds the PDF. Faster than N-body ray-tracing
and resolves the high-μ tail that N-body misses.

## Key ideas / equations to reuse
- μ = 1 / ((1−κ)² − γ²), with κ = Σⱼ κ(θⱼ), γ = √(γ₁²+γ₂²), summed over lenses (eq. 1).
- **Halos:** pseudo-elliptical NFW; κ, γ from Golse & Kneib (2002); Δ=200; c(M,z) from
  Dutton & Macciò (2014); ellipticity ε(M,z) from Allgood+ (2006).
- **Filaments:** cylindrical ρ ∝ (1+r²/r_s²)⁻¹; r_s=1 Mpc(M/10¹⁴)^⅓, L=20 Mpc(...)^⅓.
- Mass functions via excursion-set (Bond+1991) with Shen+2006 collapse barriers;
  {p,q}={0.3,0.8} halos, {0,0.7} filaments (eq. 2).
- Non-linear μ (eq. 1) retained to capture single-lens-dominated high-μ events.

## Headline results
- ET + ~300 NS-NS with EM counterparts → **σ₈ to 10%**.
- LISA + ~12 MBHBs with EM counterparts → **σ₈ to 30%**.
- Lensing adds constraining power *beyond* the mean distance-redshift relation (which fixes
  h, Ω_M): the non-Gaussian scatter is sensitive to σ₈.

## Caveats the author flags
- Non-linear structure modelling carries theoretical uncertainty (Mpetha+2024, Alfradique+2024).
- The dP/dμ shape needs refinement beyond this demonstration — exactly the gap the emulator +
  substructure work in this repo is addressing.

## Links
- Related: [[Trker_2025_AccurateCosmologicalEmulatorProbabilityDistribution]] (ACE baseline),
  [[BaltzMarshallOguri_2009_TruncatedNFW]] (lens profiles),
  [[JiangVdB_2014_SubhaloMassFunction]] (substructure the repo adds on top).
- In-repo: the whole `cpp/` engine + `CLAUDE.md` cosmology constants derive from this.

#paper #core #standard-sirens #weak-lensing #sigma8
