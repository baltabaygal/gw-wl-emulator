# Flat vs adaptive kappa-threshold PDF acceptance

**Verdict: FAIL** for flat `kappathr_flat=1e-4`.

Subhalos were disabled; all runs used seed 123 and the default cosmology (OmegaM=0.315, sigma8=0.811, h=0.674). Confidence intervals on tail quantiles are distribution-free 95% order-statistic intervals.

| z_s | rule | valid | <1/mu> | sigma(lnmu) | delta sigma | KL to adaptive | q99 | q99.9 | q99.99 |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | adaptive | 1,000,000 | 1.000041 | 0.0114992 | +0.000% | 0 | 1.03419 | 1.09446 | 1.24718 |
| 0.2 | flat_1e-4 | 1,000,000 | 1.000094 | 0.0136605 | +18.795% | 0.00359 | 1.03505 | 1.10096 | 1.26927 |
| 0.2 | flat_1e-3 | 999,997 | 1.000545 | 0.0189459 | +64.758% | 0.12 | 1.03578 | 1.12122 | 1.51395 |
| 1 | adaptive | 999,993 | 1.000458 | 0.0763584 | +0.000% | 0 | 1.28693 | 1.81546 | 4.127 |
| 1 | flat_1e-4 | 999,993 | 1.000426 | 0.0760904 | -0.351% | 0.000212 | 1.28686 | 1.83289 | 3.99229 |
| 1 | flat_1e-3 | 999,982 | 1.002306 | 0.0812673 | +6.429% | 0.00788 | 1.30116 | 1.91048 | 5.85233 |
| 10 | adaptive | 299,616 | 1.011678 | 0.325348 | +0.000% | 0 | 3.18039 | 14.2007 | 117.907 |
| 10 | flat_1e-4 | 299,660 | 1.007058 | 0.321547 | -1.168% | 0.00494 | 3.01309 | 11.5058 | 98.159 |
| 10 | flat_1e-3 | 299,676 | 1.008351 | 0.325464 | +0.035% | 0.00105 | 3.107 | 13.9625 | 137.475 |

## Acceptance checks

- z_s=0.2: **FAIL**; flux=pass, sigma=fail, KL=fail, tail overlap: q99=no, q99.9=no, q99.99=yes.
- z_s=1.0: **PASS**; flux=pass, sigma=pass, KL=pass, tail overlap: q99=yes, q99.9=yes, q99.99=yes.
- z_s=10.0: **FAIL**; flux=fail, sigma=fail, KL=fail, tail overlap: q99=no, q99.9=yes, q99.99=yes.

The same-distribution KL null medians (from multinomial replicas of the adaptive histogram) are recorded in the JSON and CSV outputs. The acceptance KL is `KL(flat || adaptive)` with 400 shared ln(mu) bins and a Jeffreys 0.5-count pseudocount.

## Raw-kappa diagnostic

The z_s=10 flux-conservation sanity check fails for all three rules, including adaptive, so it is a common high-redshift sampler issue rather than evidence specific to the threshold switch. The KL, sigma, and q99 comparisons independently fail there.

A separate 200,000-draw raw-kappa check (`scripts/figures/check_kappathr_raw_kappa.py`) shows that untrimmed second moments are tail-noisy, but it does not rescue the literal pure-split premise. With default bias enabled, the flat-1e-4 raw sigma_kappa point differences relative to adaptive are +7.32%, +4.09%, and +2.39% at z_s=0.2, 1, and 10. After retaining only centered |kappa| < 1, the corresponding differences are +7.32%, +0.11%, and +1.09%.

The live implementation uses the threshold-dependent encounter table for both the lognormal bias width and filament count/radius (`cpp/lensing.cpp`, around lines 401-402 and 434-439). Only the NFW sub-threshold host field receives `sigmakappaW`; filaments below the threshold are not Gaussian-compensated. Therefore changing kappa_thr is not a pure NFW computational repartition under the full default model, which is consistent with the observed redshift-dependent PDF gaps.

**Decision:** do not lock in flat 1e-4 as science-neutral on this evidence. The z_s=1 case passes, but the full three-redshift acceptance criterion does not.

Figure: `plots/kappathr_pdf_acceptance.png`

Raw samples: `data/results/kappathr_pdf_acceptance_samples.npz`

Machine-readable summary: `data/results/kappathr_pdf_acceptance_summary.json`

Raw-kappa metrics: `data/results/kappathr_raw_kappa_metrics.csv`
