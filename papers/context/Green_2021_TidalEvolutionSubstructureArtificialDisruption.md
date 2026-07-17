# Green, van den Bosch & Jiang 2021 — Tidal evolution of DM substructure II: artificial disruption on subhalo mass functions & radial profiles

> **TL;DR (read-before-you-open):** Artificial (numerical) subhalo disruption in DM-only
> sims suppresses subhalo mass functions by up to ~2× and biases radial profiles at the
> 10–20% level. They build a fast semi-analytic subhalo evolution model (SatGen-based,
> calibrated on the DASH simulations) that is *free* of this artefact — a clean alternative
> to N-body for substructure statistics.

| | |
|---|---|
| **Authors** | Sheridan B. Green, Frank C. van den Bosch, Fangzhou Jiang (Yale / Caltech) |
| **Year** | 2021 · MNRAS 000, 1–18 |
| **arXiv** | [2103.01227v2](https://arxiv.org/abs/2103.01227) |
| **Category** | context |
| **Relevance** | ⭐⭐⭐ — informs subhalo-model fidelity (§4 gate), not on the critical path |
| **Read status** | unread |

## Read this if…
- You're deciding how much to trust N-body-calibrated subhalo mass functions (SHMFs).
- You want a semi-analytic substructure model (SatGen) as an upgrade to the JvdB14 SHMF.
- You need the argument that *mass resolution*, not artificial disruption, drives the radial bias.

## Skip / defer if…
- The subhalo gate already concluded substructure is sub-dominant (few % vs the 50–86%
  ACE−Vaskonen gap) — this is background justification, not a required input.

## What it does
Augments a semi-analytic subhalo evolution model with an improved tidal-stripping treatment
calibrated on the DASH database of idealized high-resolution subhalo simulations (disruption-free).
Also builds a *model* of artificial disruption matched to the Bolshoi simulation, so the two can
be compared directly. Predicts SHMFs, number-density profiles, and substructure mass fractions
with and without artificial disruption.

## Key findings to reuse
- Artificial disruption suppresses SHMF / affects profiles at the **10–20%** level; earlier
  claims of factor-of-2 suppression are ameliorated once orbit integration is included.
- **Splashback haloes** (~half the subhalo population) require orbit integration to model —
  can't be captured by a snapshot cut.
- **Resolution, not artificial disruption, is the primary cause of the radial number-density
  bias** seen in DM-only sims. ⇒ semi-analytic modelling is the accurate route.

## Links
- Related: [[JiangVdB_2014_SubhaloMassFunction]] (the SHMF this repo uses),
  [[vdBoschTormenGiocoli_2005_SubhaloMassLoss]] (mass-loss physics),
  [[SalvadorSol_2025_AccurateComprehensiveApproachSubstructureIv]] (competing substructure model).
- In-repo: relevant to `docs/subhalo/subhalo_combining.md` and the §4 substructure gate.

#paper #context #subhalo #substructure #satgen
