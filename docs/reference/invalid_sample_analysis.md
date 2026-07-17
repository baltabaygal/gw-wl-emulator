# Invalid Sample Analysis

## Summary Statistics
- Total samples scanned: 72000
- Total invalid samples: 11059
- Global invalid fraction: 0.153597
- detA <= 0 count: 4
- detA <= 0 fraction: 0.000056
- nonfinite mu count: 11055
- mu <= 0 count: 4
- overflow mu count: 0

## Parameter Dependence
- Invalid fraction vs z trend: **increasing**
- Invalid fraction vs sigma8 trend: **flat_or_decreasing**

## detA Diagnostics
- See figures: `detA_histogram.png`, `detA_near_zero_zoom.png`, `detA_cdf_abs.png`.
- detA <= 0 corresponds to critical-curve crossing / strong-lensing regime where mu diverges or flips sign.

## Interpretation
- If invalid fractions increase with both z and sigma8, this supports a **physical-tail origin** (nonlinear structure and high-path-length lensing).
- nonfinite/overflow mu events should mostly coincide with detA near 0, consistent with magnification divergence.
- Persistent NaN kappa/gamma without detA crowding near 0 would suggest numerical pathologies.

## Validity Domain Estimate
- Practical weak-lensing-safe region criterion used: invalid fraction < 1%.
- Grid points passing criterion:
  - z=0.2, sigma8=0.4, f_invalid=0.0010
  - z=0.2, sigma8=0.6, f_invalid=0.0000
  - z=0.2, sigma8=0.8, f_invalid=0.0005
  - z=0.2, sigma8=1.0, f_invalid=0.0000
  - z=0.2, sigma8=1.2, f_invalid=0.0005
  - z=0.2, sigma8=1.4, f_invalid=0.0005
  - z=0.5, sigma8=0.4, f_invalid=0.0020
  - z=0.5, sigma8=0.6, f_invalid=0.0005
  - z=0.5, sigma8=0.8, f_invalid=0.0010
  - z=0.5, sigma8=1.0, f_invalid=0.0015
  - z=0.5, sigma8=1.2, f_invalid=0.0015
  - z=0.5, sigma8=1.4, f_invalid=0.0010
  - z=1, sigma8=0.6, f_invalid=0.0050
  - z=1, sigma8=0.8, f_invalid=0.0035
  - z=1, sigma8=1.0, f_invalid=0.0010
  - z=1, sigma8=1.2, f_invalid=0.0015
  - z=1, sigma8=1.4, f_invalid=0.0015

## Recommendation
- For Phase 2 baseline emulator, model only valid ln(mu) samples and explicitly report invalid/filtered fraction.
- Keep strong-lensing-like events (detA <= 0) out of the first weak-lensing emulator scope, and treat them as separate regime in future work.
