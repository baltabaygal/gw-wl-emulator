# σ metric recompute — clipped-Var vs IQR (#4 step 1)

Δσ shift + composition non-additivity (NA) at fiducial, 8-seed block, for the IQR proxy vs 2nd-moment std on lnμ trimmed to [lo,hi]%. Same clip across all four arms. Question: does the z_s=5 interaction survive a variance-based σ?

| z_s | metric | σ_base | Δσ_sub | Δσ_bias | SNR_bias | NA | fracNA | SNR_NA |
|----|----|----|----|----|----|----|----|----|
| 0.5 | iqr | 0.0222 | +0.0009 | +0.0011 | 12.4 | -0.00008 | 0.04 | 0.5 |
| 0.5 | clipVar_q999 | 0.0291 | +0.0006 | +0.0006 | 3.4 | -0.00003 | 0.03 | 0.1 |
| 0.5 | clipVar_q995 | 0.0262 | +0.0008 | +0.0008 | 6.6 | +0.00001 | 0.01 | 0.1 |
| 0.5 | clipVar_q99 | 0.0245 | +0.0008 | +0.0008 | 7.3 | +0.00001 | 0.01 | 0.1 |
| 1.0 | iqr | 0.0543 | +0.0024 | +0.0014 | 6.0 | -0.00023 | 0.07 | 1.0 |
| 1.0 | clipVar_q999 | 0.0670 | +0.0017 | +0.0003 | 1.0 | -0.00004 | 0.02 | 0.1 |
| 1.0 | clipVar_q995 | 0.0613 | +0.0021 | +0.0008 | 3.9 | -0.00012 | 0.05 | 0.4 |
| 1.0 | clipVar_q99 | 0.0577 | +0.0022 | +0.0009 | 4.9 | -0.00014 | 0.05 | 0.5 |
| 5.0 | iqr | 0.1939 | +0.0133 | -0.0068 | 10.4 | -0.00173 | 0.37 | 2.0 |
| 5.0 | clipVar_q999 | 0.2319 | +0.0106 | -0.0104 | 6.1 | -0.00101 | 1.14 | 0.5 |
| 5.0 | clipVar_q995 | 0.2145 | +0.0115 | -0.0089 | 8.3 | -0.00126 | 0.94 | 1.0 |
| 5.0 | clipVar_q99 | 0.2029 | +0.0116 | -0.0082 | 9.8 | -0.00130 | 0.61 | 1.3 |
