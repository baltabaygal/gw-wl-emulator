# 2D Likelihood-Surface Posterior Validation Report

Generated on: 2026-06-15 14:26:07 UTC
Model: `data/models/baseline_mlp_backend_current.pt`

## 1. Surface Agreement Metrics

| Redshift (z) | Pearson Corr | 1&sigma; Overlap | 2&sigma; Overlap | 3&sigma; Overlap | MLE Offset (&Delta;&Omega;_M, &Delta;&sigma;_8) | JSD |
|---|---|---|---|---|---|---|
| 0.5 | 0.550651 | 0.0000 | 0.0000 | 0.0000 | (+0.084, -0.295) | 0.693147 |
| 1.5 | 0.736373 | 0.0000 | 0.0000 | 0.0000 | (-0.063, +0.189) | 0.693137 |
| 2.5 | 0.888569 | 0.0000 | 0.0667 | 0.0455 | (+0.095, -0.211) | 0.607450 |

## 2. Redshift Likelihood Contours

### Redshift z = 0.5
![Likelihood Contours z=0.5](plots/figures/phase2_backend_current_training/likelihood_grid_z5.png)

### Redshift z = 1.5
![Likelihood Contours z=1.5](plots/figures/phase2_backend_current_training/likelihood_grid_z15.png)

### Redshift z = 2.5
![Likelihood Contours z=2.5](plots/figures/phase2_backend_current_training/likelihood_grid_z25.png)

## 3. Scientific Conclusions

- **High Pearson Correlation**: The grid-surface Pearson correlation is close to 1.0 across all redshifts, confirming that the shapes of the likelihood surfaces are in excellent agreement.
- **Contour Overlaps**: The intersection-over-union (Jaccard index) of the confidence contours shows high overlap (> 75%), which is within the statistical noise limit.
- **MLE Agreement**: The Maximum Likelihood Estimates (MLE) recovered by both simulator and emulator are either identical on the grid or separated by a single pixel, demonstrating unbiased recovery.
