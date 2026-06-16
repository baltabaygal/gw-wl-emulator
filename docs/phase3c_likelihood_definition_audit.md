# Phase 3C — Likelihood Definition Audit

This document compares the exact mathematical and implementation details of the likelihood evaluation paths for the simulator reference and the Neural Spline Flow (NSF) model.

## Likelihood Comparison Matrix

| Property | Simulator Reference | NSF Path (Continuous) | NSF Path (Simulator Compatible) | Mismatch Impact |
|---|---|---|---|---|
| **Event-level Formula** | Histogram bin probability mass: $P_i = N_{\text{bin}(x_i)} / N_{\text{total}}$ | Continuous probability density: $p(x_i \mid z_i, \theta)$ | Integrated/binned flow probability: $P_i = \int_{E_k}^{E_{k+1}} p(x \mid z_i, \theta) dx$ | **High**: Density vs. probability mass introduces a scale difference of $\approx \log(\text{bin\_width}) \approx -3.91$. |
| **Support Truncation** | Hard bounds at bin range boundary $[-1.0, 1.0]$. Events outside are clamped. | Infinite support $(-\infty, \infty)$ of continuous flow. | Hard bounds at bin range boundary $[-1.0, 1.0]$. Events outside are clamped. | **High**: Tail/edge events can have arbitrarily low/unbounded density under raw NSF, biasing posteriors. |
| **Density vs. Bin Mass** | Bin probability mass. | Continuous density. | Bin probability mass (integrated). | **High**: Non-constant offset if shape varies; absolute values are incompatible. |
| **Support Floor** | Explicit $10^{-12}$ floor applied to bin probabilities: $P_i \ge 10^{-12}$. | None. Log-density can be arbitrarily negative or non-finite. | Explicit $10^{-12}$ floor applied to integrated bin probabilities: $P_i \ge 10^{-12}$. | **High**: Avoids extreme leverage of tail events and improves numerical stability. |
| **Low-Probability Handling** | Floored to $10^{-12}$ ($\log P_i \ge -27.63$). | Raises exception on non-finite values; no lower bound. | Floored to $10^{-12}$ ($\log P_i \ge -27.63$). | **High**: Essential to match zero-probability behaviour in unphysical parameter regions. |
| **Normalization Constants** | Normalization is intrinsic to probability mass ($\sum P_k = 1$). | Normalized as a probability density function over continuous support. | Normalization is matched to the sum of the binned flow probabilities. | **Medium**: Restores exact normalization comparability. |
| **Redshift Handling** | Redshift rounded to 2 decimal places: $\text{round}(z, 2)$. | Exact float $z$ from catalog. | Redshift rounded to 2 decimal places: $\text{round}(z, 2)$. | **Low/Medium**: Negligible on discrete support but matches catalog membership grouping. |
| **Catalog Aggregation** | $\sum_{i} \log(P_{\text{bin}(x_i)} + 10^{-12})$ grouped by rounded $z$ slices. | $\sum_{i} \log p(x_i \mid z_i, \theta)$. | $\sum_{i} \log(P_{\text{bin}(x_i)} + 10^{-12})$ grouped by rounded $z$ slices. | **High**: Corrects the accumulation structure. |
| **Implicit Clamping** | Clips to $[E_{\text{min}} + 10^{-9}, E_{\text{max}} - 10^{-9}]$. | None. | Clips to $[E_{\text{min}} + 10^{-9}, E_{\text{max}} - 10^{-9}]$. | **High**: Aligns edge-case treatment. |

## Material vs. Constant Offsets

1. **Material Mismatches (Requires Action)**:
   - **Continuous density vs. discrete bin mass**: This is not a flat constant offset because the shape of the density changes across cosmologies ($\theta$). The integral over the bin width varies, so this changes the shape of the likelihood surface.
   - **Zero-probability tail behavior**: The simulator floors probabilities to $10^{-12}$ while the NSF assigns extremely small values (e.g. $10^{-100}$ or less) in far-tail regions. This causes tail events to dominate the log-likelihood sum and shift the posterior mode.
   - **Clamping**: Events outside the bin boundaries are clamped to the boundaries, forcing them into the edge bins. The NSF assigns them continuous tail densities which are extremely small.

2. **Constant Offsets (Harmless)**:
   - **Units and overall normalizations**: A pure constant offset in log-likelihood (e.g., a constant $\log(\Delta x)$ across all cosmologies) would cancel out upon posterior normalization. However, because the bins are non-uniform and the shape changes, this offset is parameter-dependent and must be aligned.
