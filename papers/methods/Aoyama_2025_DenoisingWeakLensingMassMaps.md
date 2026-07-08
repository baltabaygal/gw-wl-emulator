# Aoyama, Osato & Shirasaki 2026 — Denoising weak-lensing mass maps with a diffusion model (vs GAN)

> **TL;DR (read-before-you-open):** Trains a diffusion model (and a GAN) to remove shape
> noise from WL convergence maps, then checks which preserves cosmological summary statistics
> (power spectrum, bispectrum, 1-pt PDF, peak/minima counts, scattering transform). **Diffusion
> beats GAN** on almost all statistics down to small scales. A methods reference for
> diffusion-based denoising and for how to validate a generative WL model on 1-pt statistics.

| | |
|---|---|
| **Authors** | Shohei D. Aoyama, Ken Osato, Masato Shirasaki (Chiba / RIKEN / NAOJ) |
| **Year** | 2026 · PASJ |
| **arXiv** | [2505.00345v4](https://arxiv.org/abs/2505.00345) |
| **Category** | methods (moved from core — it's about mass-map denoising, not the μ-PDF) |
| **Relevance** | ⭐⭐⭐ — diffusion technique + statistics-validation methodology |
| **Read status** | unread |

## Read this if…
- You want a worked comparison of diffusion vs GAN for a WL generative task.
- You need a checklist of WL summary statistics to validate a generative/emulator model
  (power spectrum, bispectrum, 1-pt PDF, peak & minima counts, scattering transform).

## Skip / defer if…
- You're not doing 2D map-level generation — this repo emulates a 1-pt μ-PDF, not maps.

## What it does
Denoises WL convergence "mass maps" (removing shape noise) with two ML generative models —
a **diffusion model (DM)** and a **GAN** — both trained on 39,000 mock maps, then applied to
1,000 noisy test maps. Recovers the true convergence map at large scales and evaluates the
recovered summary statistics down to small scales.

## Key findings to reuse
- **Diffusion outperforms GAN** on nearly all statistics down to small scales.
- With diffusion, the angular power spectrum is recovered to ℓ ≲ 6000 (noise dominates from
  ℓ ≃ 2000).
- Diffusion trades higher training cost for numerically stable training, better recovery of
  cosmological statistics, and easy sampling of many realizations once trained.

## Links
- Related: [[Boruah_2025_DiffusionBasedMassMapReconstruction]] (diffusion for mass-map
  reconstruction), [[Shirasaki_2026_MLWeakLensingCosmology]] (ML-for-WL review, shared author),
  [[Trker_2025_AccurateCosmologicalEmulatorProbabilityDistribution]] (1-pt PDF emulation).

#paper #methods #diffusion #gan #weak-lensing #denoising
