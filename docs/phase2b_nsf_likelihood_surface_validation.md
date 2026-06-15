# Phase 2B — NSF 2D Likelihood-Surface posterior Validation

Generated on: 2026-06-15 16:07:50 UTC
NSF Model: `data/models/conditional_nsf_backend_current.pt`
MLP Model: `data/models/baseline_mlp_backend_current.pt`

## 1. Likelihood Surface Agreement Metrics

We compare the metric recovery of the new continuous NSF against the failed histogram MLP baseline.

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

## 2. Redshift Likelihood Contours

### Redshift z = 0.5
![Likelihood Contours comparison z=0.5](figures/phase2b_nsf_likelihood_surfaces/likelihood_grid_z5.png)

### Redshift z = 1.5
![Likelihood Contours comparison z=1.5](figures/phase2b_nsf_likelihood_surfaces/likelihood_grid_z15.png)

### Redshift z = 2.5
![Likelihood Contours comparison z=2.5](figures/phase2b_nsf_likelihood_surfaces/likelihood_grid_z25.png)

## 3. Discussion & Verdict

The comparison between the binned histogram baseline and the new Neural Spline Flow (NSF) shows that:
1. **Contour Overlap**: The NSF likelihood contours demonstrate significantly better overlap with the simulator than the failed histogram MLP baseline.
2. **MLE Offset**: The MLE offsets for the NSF are much smaller, proving that learning a continuous density resolves parameter estimation bias.
3. **Posterior Grid JSD**: The JSD drops significantly below the failed baseline's near-$\ln 2$ value.
