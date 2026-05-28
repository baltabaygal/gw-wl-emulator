# Transport Validation Bootstrap

This branch initializes the validation program for the effective transport theory of weak-lensing magnification PDFs.

## Immediate goals

1. Reproduce the Vaskonen 2026 magnification PDFs from the existing Monte Carlo infrastructure.
2. Validate the spectral extraction pipeline on synthetic data with known transport generators.
3. Test whether the reconstructed transport spectrum admits separability:

   Lambda(k,z) ~ lambda(z) F(k)

4. Attempt leave-one-out prediction across source redshifts.

## Phase 1: Numerical correctness

Before testing real lensing PDFs, validate:

- FFT normalization
- inverse reconstruction
- Gibbs suppression
- recovery of known synthetic generators
- stability under UV cutoff variation

The first control case should be a manufactured propagator:

    P_hat(k,z) = exp[ Lambda(k) z ]

with known Lambda(k).

## Phase 2: Real PDF extraction

For each redshift:

1. Convert dP/dmu -> P(x), x = ln(mu)
2. Interpolate to uniform x-grid
3. FFT to P_hat(k)
4. Extract:

    Lambda_eff(k,z) = (1/z) log P_hat(k,z)

5. Apply reliability mask

## Phase 3: Separability tests

Test:

    Lambda(k,z) ~= lambda(z) F(k)

using:

- spectral collapse
- semigroup consistency
- convolution closure
- leave-one-out prediction

## Scientific criterion

The framework has real content only if:

- one universal spectral shape F(k) exists,
- only the amplitude evolves strongly with redshift,
- and held-out PDFs can be predicted without direct fitting.

Otherwise the framework reduces to a single-redshift reparameterization.
