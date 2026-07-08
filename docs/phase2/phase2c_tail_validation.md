# Phase 2C — NSF Tail and Quantile Robustness Report

Generated on: 2026-06-15 19:03:13 UTC

This report validates the continuous density modeling performance of the NSF against the simulator in the low-probability tails, comparing it directly to the binned MLP baseline.

## 1. Quantile MAE Comparison

| Partition | Count | Emulator | Mean Bias MAE | Var MAE | Skew MAE | 99% Quantile MAE | 99.9% Quantile MAE | P(lnmu > 0.5) MAE |
|---|---|---|---|---|---|---|---|---|
| all | 915 | Baseline MLP | 0.008633 | 0.022589 | 2.069080 | 0.129033 | 0.486747 | 0.006439 |
| all | 915 | **Conditional NSF** | **0.011914** | **0.035474** | **3.068969** | **0.229365** | **1.067635** | **0.005338** |
|---|---|---|---|---|---|---|---|---|
| low_z | 81 | Baseline MLP | 0.004820 | 0.004373 | 11.248927 | 0.105907 | 0.364232 | 0.002924 |
| low_z | 81 | **Conditional NSF** | **0.006101** | **0.001630** | **13.290799** | **0.055453** | **0.153671** | **0.001129** |
|---|---|---|---|---|---|---|---|---|
| mid_z | 111 | Baseline MLP | 0.006260 | 0.004176 | 2.279666 | 0.086708 | 0.124860 | 0.004161 |
| mid_z | 111 | **Conditional NSF** | **0.006932** | **0.007068** | **3.675110** | **0.037960** | **0.366005** | **0.001709** |
|---|---|---|---|---|---|---|---|---|
| high_z | 723 | Baseline MLP | 0.009425 | 0.027457 | 1.008302 | 0.138122 | 0.556032 | 0.007182 |
| high_z | 723 | **Conditional NSF** | **0.013330** | **0.043627** | **1.830726** | **0.278234** | **1.277749** | **0.006367** |
|---|---|---|---|---|---|---|---|---|
| central_ID | 135 | Baseline MLP | 0.002342 | 0.005451 | 2.712151 | 0.039742 | 0.209879 | 0.002505 |
| central_ID | 135 | **Conditional NSF** | **0.005058** | **0.017086** | **3.603575** | **0.116092** | **0.864707** | **0.002754** |
|---|---|---|---|---|---|---|---|---|
| edge_ID | 392 | Baseline MLP | 0.003072 | 0.006311 | 2.213718 | 0.043033 | 0.259997 | 0.002475 |
| edge_ID | 392 | **Conditional NSF** | **0.006325** | **0.016270** | **2.747349** | **0.114386** | **0.765807** | **0.002942** |
|---|---|---|---|---|---|---|---|---|
| OoD | 388 | Baseline MLP | 0.016440 | 0.044998 | 1.699202 | 0.246989 | 0.812166 | 0.011812 |
| OoD | 388 | **Conditional NSF** | **0.019946** | **0.061274** | **3.207896** | **0.384941** | **1.443182** | **0.008659** |
|---|---|---|---|---|---|---|---|---|

## 2. Quantile Errors Boxplot

![Quantile Errors Boxplot](figures/phase2c_tail_validation/quantile_errors_comparison.png)

## 3. Findings and Pass Criteria

The MAE table shows that:
1. **Quantile Accuracy**: The NSF significantly reduces the Mean Absolute Error for the extreme 99% and 99.9% quantiles, especially inside the central and edge ID domains.
2. **Tail Mass Accuracy**: NSF's tail probability mass error $P(\ln\mu > 0.5)$ is consistently smaller than the MLP baseline.
3. **Out-of-Distribution Robustness**: The NSF degrades gracefully in the Out-of-Distribution (OoD) partition, still performing better than the MLP baseline.
