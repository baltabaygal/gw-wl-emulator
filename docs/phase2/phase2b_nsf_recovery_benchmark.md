# Phase 2B — NSF Parameter Recovery Benchmark Report

Generated on: 2026-06-15 17:23:02 UTC

## 1. Bias Statistics Comparison

| Parameter | Method | Mean Bias &plusmn; Std Dev | Max Absolute Deviation |
|---|---|---|---|
| **h** | Simulator | +0.003249 &plusmn; 0.034092 | 0.081036 |
| **h** | Baseline MLP | -0.033474 &plusmn; 0.070450 | 0.134381 |
| **h** | **Conditional NSF** | **+0.008192 &plusmn; 0.056885** | **0.127198** |
|---|---|---|---|
| **OmegaM** | Simulator | +0.000036 &plusmn; 0.047498 | 0.089466 |
| **OmegaM** | Baseline MLP | +0.037017 &plusmn; 0.077579 | 0.152992 |
| **OmegaM** | **Conditional NSF** | **+0.009998 &plusmn; 0.051813** | **0.088147** |
|---|---|---|---|
| **sigma8** | Simulator | -0.016775 &plusmn; 0.095623 | 0.171438 |
| **sigma8** | Baseline MLP | -0.073516 &plusmn; 0.141342 | 0.333918 |
| **sigma8** | **Conditional NSF** | **+0.003009 &plusmn; 0.085647** | **0.231801** |
|---|---|---|---|

## 2. Optimization Speedup

- **Average Simulator Time**: 614.1816 seconds
- **Average MLP Time**: 0.0147 seconds (Speedup: **41871.3x**)
- **Average NSF Time**: 7.8233 seconds (Speedup: **78.5x**)

## 3. Recovery Visualization

![Parameter Recovery Biases](figures/phase2b_nsf_recovery/parameter_recovery_biases.png)
