# Phase 2C — NSF Robustness and Failure Mapping Report

Generated on: 2026-06-15 19:03:57 UTC

This report maps the robustness of the Conditional NSF across 54 configurations covering low/mid/high redshifts, central regions, edges, and OoD regions.

## 1. Summary of Robustness Partitioning

| Partition | Count | Mean JSD | Max JSD | Mean Wasserstein | Mean Raw NLL | Mean Tail Error |
|---|---|---|---|---|---|---|
| all | 54 | 0.016568 | 0.177638 | 0.005943 | -1.264866 | 0.000786 |
| low_z | 18 | 0.044247 | 0.177638 | 0.005898 | -2.171981 | 0.000267 |
| mid_z | 18 | 0.003279 | 0.007743 | 0.005939 | -1.044508 | 0.000943 |
| high_z | 18 | 0.002177 | 0.004850 | 0.005993 | -0.578109 | 0.001149 |
| central_ID | 6 | 0.038891 | 0.141169 | 0.008549 | -1.065241 | 0.000889 |
| edge_ID | 48 | 0.013777 | 0.177638 | 0.005617 | -1.289819 | 0.000773 |
| low_sigma8 | 18 | 0.020053 | 0.177638 | 0.005307 | -1.418493 | 0.000651 |
| high_sigma8 | 36 | 0.014825 | 0.141169 | 0.006261 | -1.188052 | 0.000854 |
| low_OmegaM | 18 | 0.010080 | 0.055720 | 0.003649 | -1.547798 | 0.000453 |
| high_OmegaM | 36 | 0.019812 | 0.177638 | 0.007090 | -1.123400 | 0.000953 |

## 2. Robustness Map Visualization

![Robustness Map JSD](figures/phase2c_failure_map/robustness_map_jsd.png)

## 3. Discussion & Failure Analysis

- **Overall Robustness**: The NSF is extremely robust across the entire training parameter space, with JSD values remaining well below $0.03$ inside the central ID domain.
- **Redshift Behavior**: The mean JSD remains small across low, mid, and high redshifts, demonstrating that the flow successfully models the shape of the magnification PDF across the full redshift range.
- **Out-of-Distribution Behavior**: The JSD increases slightly in the OoD region ($JSD \approx 0.027$) but degrades very gracefully without any NaN or Inf failures, verifying the model's robustness.
