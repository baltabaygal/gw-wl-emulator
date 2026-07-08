# van den Bosch, Tormen & Giocoli 2005 — The Mass Function and Average Mass-Loss Rate of Dark Matter Subhaloes

> **TL;DR (read-before-you-open):** The foundational semi-analytic subhalo model this line of
> work (→ JvdB14) descends from. Key result: the **average subhalo mass-loss rate** can be
> written as a simple function of redshift and the instantaneous subhalo/host mass ratio, and
> the resulting SHMF is **not universal** — its slope & normalization depend on M/M* (host
> formation-time dependence).

| | |
|---|---|
| **Authors** | Frank C. van den Bosch, Giuseppe Tormen, Carlo Giocoli (ETH Zürich / Padova) |
| **Year** | 2005 · MNRAS (arXiv astro-ph/0409201) |
| **arXiv** | [astro-ph/0409201](https://arxiv.org/abs/astro-ph/0409201) |
| **Category** | context |
| **Relevance** | ⭐⭐⭐ — origin of the mass-loss ansatz; direct ancestor of the SHMF used |
| **Read status** | unread |

## Read this if…
- You want the average mass-loss-rate ansatz (function of z and m/M only) and its calibration.
- You need the argument that the SHMF is non-universal (depends on M/M*, host formation time).
- You're tracing where the repo's subhalo mass-loss physics originally comes from.

## Skip / defer if…
- You just need the updated fitting functions — go to [[JiangVdB_2014_SubhaloMassFunction]].

## What it does
Semi-analytic model for DM subhalo mass functions: accretion masses from merger trees, then
mass loss from combined dynamical friction + tidal stripping + tidal heating, modelled as an
**orbit-averaged** rate independent of parent halo mass. Calibrated against high-resolution
cluster-sized N-body SHMFs.

## Key results to reuse
- Average mass-loss rate depends only on redshift and instantaneous m/M.
- **SHMF is non-universal**: slope & normalization depend on M/M* (massive hosts form later →
  less time for mass loss → different SHMF).
- Galaxy-sized haloes (M≃10¹²h⁻¹M⊙) have ~3× lower subhalo mass fraction than clusters (10¹⁵).
- Subhalo mass fraction depends most strongly on accretion history in the last ~1 Gyr.

## Links
- Related: [[JiangVdB_2014_SubhaloMassFunction]] (successor),
  [[Green_2021_TidalEvolutionSubstructureArtificialDisruption]],
  [[HanColeFrenkJing_2016_SubhaloSpatialDistribution]].
- In-repo: `cpp/subhalo.cpp` (Giocoli+2007 w̃_f factor is from this lineage).

#paper #context #subhalo #mass-loss #shmf
