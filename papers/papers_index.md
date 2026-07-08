# Papers Index

_Last updated: 2026-07-07_

Each entry links to its **note** (`.md`) — read the note first to decide whether to open the PDF.
Relevance: ⭐⭐⭐⭐⭐ = built on it · ⭐ = tangential. Notes lead with a TL;DR + "read this if / skip if".

**Categories:** `core` = the PDF/emulator we build & benchmark · `methods` = ML/statistical
techniques · `context` = astrophysics background (WL, NFW, subhalos) · `misc` = tangential.

---

## Core — the thing we're building / competing with

| Paper | ★ | What it is |
|---|---|---|
| [Vaskonen 2026](core/Vaskonen_2026_GWWeakLensingSigma8.md) | ⭐⭐⭐⭐⭐ | The C++ engine this repo emulates: GW standard-siren weak-lensing dP/dμ → σ₈. |
| [Türker 2026 (ACE)](core/Trker_2025_AccurateCosmologicalEmulatorProbabilityDistribution.md) | ⭐⭐⭐⭐⭐ | ACE — PCA+XGBoost emulator of the same μ-PDF; the KL≈0.007 baseline. |

## Methods — ML / statistical techniques

| Paper | ★ | What it is |
|---|---|---|
| [Aoyama 2026](methods/Aoyama_2025_DenoisingWeakLensingMassMaps.md) | ⭐⭐⭐ | Diffusion vs GAN denoising of WL mass maps; how to validate on 1-pt statistics. |
| [Boruah 2025](methods/Boruah_2025_DiffusionBasedMassMapReconstruction.md) | ⭐⭐ | One diffusion model for WL simulation + mass-map reconstruction (DPS bias fix). |
| [Shirasaki 2026](methods/Shirasaki_2026_MLWeakLensingCosmology.md) | ⭐⭐ | Review chapter: ML applications in weak-lensing cosmology. |

## Context — astrophysics background (WL fundamentals + subhalo substructure)

| Paper | ★ | What it is |
|---|---|---|
| [Jiang & vdB 2014](context/JiangVdB_2014_SubhaloMassFunction.md) | ⭐⭐⭐⭐ | JvdB14 SHMF fitting functions — the subhalo mass function the code uses. |
| [Han+ 2016](context/HanColeFrenkJing_2016_SubhaloSpatialDistribution.md) | ⭐⭐⭐⭐ | Unified subhalo spatial + mass distribution (SubGen); where to place clumps. |
| [Baltz+ 2009](context/BaltzMarshallOguri_2009_TruncatedNFW.md) | ⭐⭐⭐ | BMO truncated-NFW lensing profile (finite-mass clumps). |
| [vdB, Tormen & Giocoli 2005](context/vdBoschTormenGiocoli_2005_SubhaloMassLoss.md) | ⭐⭐⭐ | Origin of the subhalo mass-loss ansatz; non-universal SHMF. |
| [Green+ 2021](context/Green_2021_TidalEvolutionSubstructureArtificialDisruption.md) | ⭐⭐⭐ | Artificial disruption in sims; SatGen disruption-free substructure model. |
| [Salvador-Solé+ 2025 (CUSP)](context/SalvadorSol_2025_AccurateComprehensiveApproachSubstructureIv.md) | ⭐⭐ | CUSP analytic substructure + dynamical friction; deferred SHMF alternative. |
| [Hunde+ 2026](context/Hunde_2026_CaughtCosmicWebEnvironmentalImpacts.md) | ⭐⭐ | Cosmic-web environment dependence of subhalo populations / boosts. |
| [Prat & Bacon 2025](context/Prat_2025_WeakGravitationalLensing.md) | ⭐⭐ | Weak-lensing fundamentals review (κ, γ, cosmic shear, 3×2pt). |

## Misc — tangential

| Paper | ★ | What it is |
|---|---|---|
| [Urrutia+ 2025](misc/Urrutia_2025_StarlightJwstImplicationsStarFormation.md) | ⭐ | JWST star formation vs DM models (FDM/WDM/PBH bounds); shared Vaskonen author. |
| [Shlesinger 2017](misc/Shlesinger_2017_OriginsApplicationsMontrollWeissContinuous.md) | ⭐ | Montroll-Weiss CTRW / anomalous-diffusion review (math background). |
