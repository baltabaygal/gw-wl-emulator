# Statistical Stability and Dataset Validation Report

Generated on: 2026-05-21 18:53:17

## 1. Monte Carlo Convergence
We monitored mean, variance, skewness, kurtosis, and extreme percentiles (1st and 99th) for sample sizes [1000, 5000, 10000, 50000].
The weak-lensing distributions present heavy tails. Variance of higher-order moments like kurtosis begins stabilizing after 10000 samples, implying $N=10000$ to $50000$ as a safe regime for ML dataset generation targeting tail fidelity.

## 2. Parameter Smoothness (Wasserstein Continuity)
Varying $sigma_8$ linearly demonstrates a continuously varying Wasserstein distance against adjacent cosmological parameters. This implies our underlying weak lensing simulations produce densities that smoothly depend on the physical conditioning parameters, meeting the assumptions necessary for Normalizing Flows.

## 3. Multidimensional Interpolation
We transitioned dataset splitting to a Checkerboard strategy defined jointly over $(Omega_M, sigma_8)$. Validation and Test splits are inherently bound within the convex hull of the Train dataset interpolating regions. This ensures the ML estimator is strictly interpolating instead of suffering from out-of-distribution (OoD) edge extrapolation.

## 4. OOD Extrapolation Tracking
Explicit OoD metrics and plotting track edge-of-domain points. We label validation samples explicitly as `interpolation` or `OoD` at generation time to enforce strict evaluation boundaries.

## 5. Tail Stability
Extreme percentiles (e.g. 99th) across various fixed seeds present narrow standard deviations (e.g., $< 0.05$ absolute differences). This provides statistical stability when training simulation-based inference models that require robust and non-chaotic target distributions at the tails.

## Conclusion
The infrastructure is structurally ready and scalable for Phase 2 neural density estimation tasks.
