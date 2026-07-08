# Baltz, Marshall & Oguri 2009 — Analytic models of plausible gravitational lens potentials (truncated NFW / BMO)

> **TL;DR (read-before-you-open):** The **BMO truncated-NFW** paper. Gives analytic lensing
> expressions (κ, γ, potential, flexion, all derivatives) for a smoothly-truncated NFW profile
> whose density falls off as r⁻⁵ or r⁻⁷ beyond the tidal radius — ensuring **finite total mass**
> and non-divergent lensing, unlike the infinite-mass NFW. This is the reference for the
> truncated-clump profiles used when modelling stripped subhalos.

| | |
|---|---|
| **Authors** | Edward A. Baltz, Phil Marshall, Masamune Oguri (KIPAC/Stanford, UCSB) |
| **Year** | 2009 · JCAP (arXiv 2007, v3 2008) |
| **arXiv** | [0705.0682v3](https://arxiv.org/abs/0705.0682) |
| **Category** | context |
| **Relevance** | ⭐⭐⭐ — the truncated-NFW lensing profile for subhalo clumps |
| **Read status** | unread |

## Read this if…
- You need analytic κ/γ (and higher derivatives) for a **truncated** NFW lens.
- You're modelling stripped subhalos and need a finite-mass, non-divergent profile.
- You want the r⁻⁵ / r⁻⁷ outer-slope functional forms and their normalization.

## Skip / defer if…
- The current model uses untruncated NFW clumps (a known caveat in this repo) — this paper is
  the upgrade path, not the current implementation.

## What it does
Provides analytic lens potentials for a hybrid model: ellipsoidal symmetry + universal DM
profile (+ Sérsic baryons). Central contribution is **smoothly-truncated NFW** profiles with
outer slope −5 or −7 beyond the tidal radius, giving finite mass and analytically calculable
lens-potential derivatives to all orders. Shows how observables differ from the infinite NFW,
highlights flexion, and frames how truncation models stripped haloes.

## Key results to reuse
- Closed-form κ, γ, deflection, flexion for truncated NFW (the "BMO" profile).
- Truncation → finite total mass and non-divergent lensing → needed for ray-tracing/catastrophes.
- Decreasing tidal radius = modelling tidal stripping of subhalos.

## Links
- Related: [[JiangVdB_2014_SubhaloMassFunction]], [[vdBoschTormenGiocoli_2005_SubhaloMassLoss]],
  [[HanColeFrenkJing_2016_SubhaloSpatialDistribution]] (what gets stripped/where).
- In-repo: NFW κ conventions in `CLAUDE.md`; caveat "untruncated NFW clumps" in subhalo model.

#paper #context #nfw #truncated-nfw #lensing #subhalo
