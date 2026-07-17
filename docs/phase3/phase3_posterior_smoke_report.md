# Phase 3 Posterior Smoke Report

Smoke catalog: `mixed_uniform_central_N1000_seed610001_57cb44ad1ef0f9d4.npz`

Configuration: central mixed catalog, `N=1000`, stratified uniform redshift support with 3 support points, 2D `(OmegaM, sigma8)` grid at resolution 12 with `h` fixed to truth, simulator `nsim_per_z=10000`.

## Integrity

- Simulator commit: `59c107efc320264b2a6d91bf31a89e62760b5ca7`
- NSF checkpoint hash: `9d3a1cacb36bc53d...`
- Preprocessing stats hash: `a8dd1d0aad6fab32...`
- Simulator grid time: 235.15 s
- NSF grid time: 0.426 s

The smoke run completed without NaNs, cache mismatches, or posterior serialization failures.

## Posterior Agreement

- Posterior JSD: 0.607845
- Total variation distance: 0.947312
- 68% credible-region overlap: 0.000
- 95% credible-region overlap: 0.167

| Parameter | Sim mean | NSF mean | Normalized shift | Sim 68% CI | NSF 68% CI |
|---|---:|---:|---:|---|---|
| OmegaM | 0.334804 | 0.225592 | 8.027 | [0.327273, 0.345455] | [0.200000, 0.218182] |
| sigma8 | 0.746767 | 1.018658 | 7.788 | [0.722727, 0.759091] | [1.050000, 1.050000] |

## Smoke Decision

The smoke benchmark is technically clean but scientifically fails the posterior-agreement criterion. The NSF posterior is shifted toward low `OmegaM` and high `sigma8` relative to the simulator reference.

Coverage and larger main runs are deferred until this mismatch is understood.

