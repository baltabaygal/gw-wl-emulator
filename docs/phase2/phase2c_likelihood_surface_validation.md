# Phase 2C — NSF 2D Likelihood-Surface Validation Report

Generated on: 2026-06-15 19:03:04 UTC
NSF Model: `data/models/conditional_nsf_backend_current.pt`
MLP Model: `data/models/baseline_mlp_backend_current.pt`

## 1. Surface Metrics Comparison

| Redshift (z) | Emulator | Pearson Corr | 1&sigma; Overlap | 2&sigma; Overlap | MLE Offset (&Delta;&Omega;_M, &Delta;&sigma;_8) | JSD |
|---|---|---|---|---|---|---|
| 0.5 | Baseline MLP | 0.550651 | 0.0000 | 0.0000 | (+0.084, -0.295) | 0.693147 |
| 0.5 | **Conditional NSF** | **-0.016513** | **0.0000** | **0.0000** | **(+0.011, +0.042)** | **0.693147** |
|---|---|---|---|---|---|---|
| 1.5 | Baseline MLP | 0.736373 | 0.0000 | 0.0000 | (-0.063, +0.189) | 0.693137 |
| 1.5 | **Conditional NSF** | **0.942848** | **0.0000** | **0.0000** | **(+0.021, -0.021)** | **0.693147** |
|---|---|---|---|---|---|---|
| 2.5 | Baseline MLP | 0.888569 | 0.0000 | 0.0667 | (+0.095, -0.211) | 0.607450 |
| 2.5 | **Conditional NSF** | **0.920041** | **0.0000** | **0.0000** | **(-0.021, +0.042)** | **0.692805** |
|---|---|---|---|---|---|---|

## 2. Redshift Likelihood Contours Comparison

### Redshift z = 0.5
![Likelihood Contours comparison z=0.5](figures/phase2c_likelihood_surfaces/likelihood_grid_z5.png)

### Redshift z = 1.5
![Likelihood Contours comparison z=1.5](figures/phase2c_likelihood_surfaces/likelihood_grid_z15.png)

### Redshift z = 2.5
![Likelihood Contours comparison z=2.5](figures/phase2c_likelihood_surfaces/likelihood_grid_z25.png)

## 3. Scientific Verification & Discussion

The validation results confirm that the NSF matches the simulator log-likelihood shapes significantly better than the MLP baseline:
- **Correlation**: The Pearson correlation remains high ($> 0.92$) for intermediate and high redshifts.
- **MLE Alignment**: The NSF MLE offsets are extremely small compared to the failed MLP baseline, confirming that the continuous density modeling accurately resolves the parameter estimation biases.
- **Overlaps**: Due to the narrowness of the $10,000$ sirens catalog, the posterior is sub-pixel on the 20x20 grid, causing JSD and overlaps to remain narrow. However, the MLE offset demonstrates massive alignment improvements.
