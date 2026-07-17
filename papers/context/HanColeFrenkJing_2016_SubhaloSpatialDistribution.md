# Han, Cole, Frenk & Jing 2016 — A unified model for the spatial and mass distribution of subhaloes (SubGen)

> **TL;DR (read-before-you-open):** A single analytic model reproducing both the universal
> subhalo mass function *and* the radial number-density (spatial) distribution, from one
> physically-motivated description of tidal-stripping mass loss. Predicts the joint
> distribution of subhalos in (final mass, infall mass, radius) → gives the subhalo spatial
> profile, lensing profile, and annihilation boost. Ships the **SubGen** Python code.

| | |
|---|---|
| **Authors** | Jiaxin Han, Shaun Cole, Carlos S. Frenk, Yipeng Jing (Durham / SJTU) |
| **Year** | 2016 · MNRAS (arXiv 2015) |
| **arXiv** | [1509.02175v2](https://arxiv.org/abs/1509.02175) |
| **Category** | context |
| **Relevance** | ⭐⭐⭐⭐ — the radial/spatial distribution of clumps (where to place subhalos) |
| **Read status** | unread |
| **Code** | SubGen — http://icc.dur.ac.uk/data/ |

## Read this if…
- You need the **radial distribution** of subhalos inside a host (this repo places clumps by r).
- You want the joint (final mass, infall mass, radius) distribution to sample clump populations.
- You want the resolution of the subhalo/galaxy "anti-bias" confusion (it's a selection effect).

## Skip / defer if…
- You only need the mass function without spatial dependence — [[JiangVdB_2014_SubhaloMassFunction]].

## What it does
Explains universal subhalo mass + radial distributions from a simple analytic model of tidal
stripping that reduces subhalo mass with decreasing halocentric distance. Starting point: the
spatial distribution of subhalos at fixed **infall** mass is ~identical to the host mass profile;
tidal stripping then reshapes it. Yields the full joint distribution of final mass, infall mass,
and radius.

## Key results to reuse
- Universal SHMF ∝ dN/dln m = A M_host m⁻ᵅ with α ≃ 0.9 (host-mass independent, exp. high-mass tail).
- Derived quantities from the joint distribution: subhalo spatial distribution, **gravitational
  lensing profile**, DM annihilation profile & boost factor.
- The "anti-bias" (subhalos less concentrated than DM) is a simple **selection effect**.

## Links
- Related: [[JiangVdB_2014_SubhaloMassFunction]] (mass function),
  [[vdBoschTormenGiocoli_2005_SubhaloMassLoss]] (mass-loss),
  [[Green_2021_TidalEvolutionSubstructureArtificialDisruption]] (radial-bias resolution effects).
- In-repo: r-dependent resolved/unresolved split in the subhalo model (`docs/subhalo/subhalo_combining.md`).

#paper #context #subhalo #spatial-distribution #subgen
