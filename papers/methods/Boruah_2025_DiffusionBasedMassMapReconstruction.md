# Boruah, Jacob & Jain 2025 — Diffusion-based mass-map reconstruction from weak-lensing data

> **TL;DR (read-before-you-open):** A single diffusion model used for *both* fast WL simulation
> and the inverse problem of reconstructing mass maps from noisy shear, via Diffusion Posterior
> Sampling (DPS). Key technical point: naive DPS gives **biased** inference; they fix it by
> reweighting the likelihood term at early sampling steps. Methods reference for
> score/diffusion-based inference and covariance estimation for higher-order statistics.

| | |
|---|---|
| **Authors** | Supranta S. Boruah, Michael Jacob, Bhuvnesh Jain (UPenn) |
| **Year** | 2025 (Feb 7) |
| **arXiv** | [2502.04158v1](https://arxiv.org/abs/2502.04158) |
| **Category** | methods |
| **Relevance** | ⭐⭐ — diffusion/DPS technique; tangential to the 1-pt μ-PDF emulator |
| **Read status** | unread |

## Read this if…
- You want a diffusion model that serves as both a forward simulator and an inverse-problem solver.
- You need the DPS bias-correction trick (reweight likelihood at early diffusion steps).
- You're estimating covariances for higher-order WL statistics from generated samples.

## Skip / defer if…
- You only need 1-pt magnification-PDF emulation — this is field/map-level, not point-source μ.

## What it does
Trains one **Diffusion Posterior Sampling** model to (a) simulate WL shear/convergence fields
and (b) reconstruct high-resolution (sub-arcminute) mass maps from noisy shear. Finds standard
DPS yields biased inference and corrects it by reweighting the likelihood term at early
sampling steps of the reverse diffusion.

## Key findings to reuse
- Reconstructed maps have the correct power spectrum and a range of non-Gaussian statistics.
- Applications: simulation-quality mock maps, covariance estimation for higher-order statistics,
  and finding filaments/voids/clusters from noisy shear.

## Links
- Related: [[Aoyama_2025_DenoisingWeakLensingMassMaps]] (diffusion vs GAN denoising),
  [[Shirasaki_2026_MLWeakLensingCosmology]] (ML-for-WL review).

#paper #methods #diffusion #dps #mass-map #weak-lensing
