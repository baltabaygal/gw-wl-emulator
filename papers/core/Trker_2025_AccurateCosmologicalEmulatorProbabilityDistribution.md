# Türker, Marra, Castro, Quartin & Borgani 2026 — Accurate cosmological emulator for the lensing magnification PDF of point sources (ACE)

> **TL;DR (read-before-you-open):** This is **ACE** — the PCA+XGBoost emulator that predicts
> the lensing magnification PDF over (Ω_M, σ₈, w, h, z≤6) from N-body light cones, at median
> KL divergence 0.007 vs simulation. It's the direct baseline the NSF emulator in this repo is
> benchmarked against, and it's installed in the `test` conda env as `ace_lensing`.

| | |
|---|---|
| **Authors** | Tunç Türker, Valerio Marra, Tiago Castro, Miguel Quartin, Stefano Borgani (Trieste / Vitória) |
| **Year** | 2026 · A&A (accepted) |
| **arXiv** | [2512.01607v2](https://arxiv.org/abs/2512.01607) |
| **Category** | core |
| **Relevance** | ⭐⭐⭐⭐⭐ — the emulator baseline this repo competes with / validates against |
| **Read status** | reference |
| **Code** | `ace_lensing` (public; installed in env `test`) |

## Read this if…
- You want the accuracy bar to beat/match (median KL 0.007) and how they measure it.
- You need the ACE parameter space + redshift range (Ω_M, σ₈, w, h; 0.2 ≤ z ≤ 6).
- You're deciding PCA-component counts / an XGBoost interpolation approach for PDFs.

## Skip / defer if…
- You only need the physics of dP/dμ — that's [[Vaskonen_2026_GWWeakLensingSigma8]] upstream.

## What it does
Builds cosmological N-body sims → past light cones → convergence & shear maps → lensing
magnification PDFs. Compresses PDFs with **PCA**, then interpolates PCA coefficients across
the cosmological parameter space with **XGBoost** (gradient-boosted trees). Picks the optimal
number of PCA components trading accuracy vs stability.

## Key numbers to reuse
- **Median KL divergence 0.007** vs held-out sims — the number this repo's NSF model matches.
- Parameters: (Ω_M, σ₈, w, h); redshift 0.2 ≤ z ≤ 6.
- Focus: **1-point** statistics (full non-Gaussian PDF), complementary to 2-pt shear analyses.
- Motivation matches this project: unbiased cosmological inference from magnification for SNe
  and **GW distances** (biases if PDF is mismodelled: Pierel+2021, Shan+2021).

## Caveats the authors flag
- DMO sims — baryonic physics can shift the high-μ PDF (Castro+2018); future work = hydro.
- Accuracy may need more training data to fully generalize.

## Links
- Related: [[Vaskonen_2026_GWWeakLensingSigma8]] (alt engine for the same PDF),
  [[Aoyama_2025_DenoisingWeakLensingMassMaps]] (also 1-pt WL statistics, ML).
- In-repo: `ace_lensing/`, `scripts/ace_gap.py`, memory `ace_lensing_baseline.md`.

#paper #core #emulator #lensing-pdf #ace #baseline
