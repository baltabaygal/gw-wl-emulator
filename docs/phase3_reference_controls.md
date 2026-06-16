# Phase 3 Reference Controls

Catalog: `mixed_uniform_central_N1000_seed610001_57cb44ad1ef0f9d4.npz`  N=1000  truth OmegaM=0.3 sigma8=0.85

## 1. Sim-vs-sim reproducibility (same nsim, different seed)
If JSD here is large, the reference cannot reproduce itself and the gate is broken.

| Pair | JSD | TV |
|---|---:|---:|
| sim_nsim10000_seed100 vs sim_nsim10000_seed777 | 0.677294 | 0.99067 |
| sim_nsim10000_seed100 vs sim_nsim10000_seed2024 | 0.621212 | 0.963999 |
| sim_nsim10000_seed777 vs sim_nsim10000_seed2024 | 0.433062 | 0.821502 |

## 2. MC convergence (lo vs hi nsim, same seed)
- nsim 10000 vs 40000: JSD=0.363910, TV=0.737452

## 3. Truth recovery (MAP / mean per posterior)
Injected truth: OmegaM=0.3, sigma8=0.85

| Posterior | OmegaM MAP | OmegaM mean | sigma8 MAP | sigma8 mean |
|---|---:|---:|---:|---:|
| sim_nsim10000_seed100 | 0.3273 | 0.3348 | 0.7591 | 0.7468 |
| sim_nsim10000_seed777 | 0.3636 | 0.3620 | 0.7227 | 0.7262 |
| sim_nsim10000_seed2024 | 0.3818 | 0.3417 | 0.6864 | 0.7735 |
| sim_nsim40000_seed100 | 0.3818 | 0.3507 | 0.6864 | 0.7325 |
| nsf_compatible | 0.2000 | 0.2159 | 1.0500 | 1.0326 |

## 4. NSF vs hi-stat simulator
- JSD=0.649501, TV=0.970858

