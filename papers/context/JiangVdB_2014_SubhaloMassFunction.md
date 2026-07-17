# Jiang & van den Bosch 2014 — Statistics of Dark Matter Substructure I: Model and Universal Fitting Functions

> **TL;DR (read-before-you-open):** The **JvdB14** semi-analytic subhalo model + universal
> fitting functions for the **evolved and unevolved subhalo mass functions** (SHMF), valid for
> any host mass, redshift, and ΛCDM cosmology. This is *the* SHMF this repo's substructure gate
> is built on — lift the fitting-function numbers from here, not from memory.

| | |
|---|---|
| **Authors** | Fangzhou Jiang, Frank C. van den Bosch (Yale) |
| **Year** | 2014 · MNRAS (arXiv 2014) |
| **arXiv** | [1403.6827](https://arxiv.org/abs/1403.6827) |
| **Category** | context |
| **Relevance** | ⭐⭐⭐⭐ — the SHMF the subhalo module uses directly |
| **Read status** | unread (fitting functions cited in code) |

## Read this if…
- You need the evolved/unevolved SHMF fitting-function coefficients (the ones in the code).
- You want the model's assumptions: Parkinson+2008 merger trees, orbit-averaged mass-loss.
- You need the mass-loss result (avg subhalo loses >80% of infall mass in first radial orbit).

## Skip / defer if…
- You only need spatial distribution (see [[HanColeFrenkJing_2016_SubhaloSpatialDistribution]])
  or the artificial-disruption correction ([[Green_2021_TidalEvolutionSubstructureArtificialDisruption]]).

## What it does
A fast semi-analytic model of DM subhalo evolution: masses & redshifts at accretion from
Parkinson+2008 merger trees, then evolved with a simple **orbit-averaged mass-loss rate**.
Treats subhalos of all orders, scatter in orbits/concentrations, and a recipe converting
subhalo mass → max circular velocity. Reproduces N-body subhalo mass & velocity functions.

## Key results to reuse
- **Universal fitting functions** for evolved and unevolved SHMFs — valid for any host mass,
  redshift, ΛCDM cosmology. (These are the numbers to lift for the substructure gate.)
- Average subhalo loses **>80%** of its infall mass during its first radial orbit.
- Total subhalo mass fraction tightly correlates with the host's "dynamical age".

## Links
- Related: [[vdBoschTormenGiocoli_2005_SubhaloMassLoss]] (predecessor mass-loss model),
  [[HanColeFrenkJing_2016_SubhaloSpatialDistribution]] (spatial),
  [[Green_2021_TidalEvolutionSubstructureArtificialDisruption]] (disruption correction),
  [[SalvadorSol_2025_AccurateComprehensiveApproachSubstructureIv]] (competing model).
- In-repo: `cpp/subhalo.cpp`, `scripts/subhalo_*.py`, memory `subhalo_gate_verdict.md`.

#paper #context #subhalo #shmf #jvdb14
