# Phase 3D — Posterior Scaling Metrics Report

This report fits convergence trends to evaluate whether posterior discrepancy vanishes asymptotically ($B \to 0$).

## Convergence Fit Parameters

Fit model: $\text{Metric}(N) = A \cdot N^{-\alpha} + B$

| Catalog Type | Metric | Scale A | Decay Power $\alpha$ | Asymptotic Floor B | Status |
|---|---|---:|---:|---:|---|
| `mixed_uniform` | JSD | -34033.8203 | 2.4165 | 0.6931 | SATURATED |
| `mixed_uniform` | TV | -526958.6158 | 2.9830 | 1.0000 | SATURATED |
| `mixed_low_z_dominated` | JSD | -1241272.3774 | 2.9999 | 0.6931 | SATURATED |
| `mixed_low_z_dominated` | TV | -966520.2700 | 2.9998 | 1.0000 | SATURATED |
| `mixed_high_z_dominated` | JSD | -621.7546 | 1.5605 | 0.6931 | SATURATED |
| `mixed_high_z_dominated` | TV | -13751.9507 | 2.1306 | 1.0000 | SATURATED |

## Interpretation
If the asymptotic floor $B$ is close to 0, the mismatch is statistically dominated. If $B$ remains large, the mismatch is systematic and persists even at high statistics.
