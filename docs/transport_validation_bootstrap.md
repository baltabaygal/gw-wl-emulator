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

## Update (2026-05-29): Cross-z and cross-sigma8 result

Using the existing nine NPZ products from:

- z = {0.5, 1.0, 2.0, 5.0}
- sigma8 = {0.75, 0.81, 0.87}

with no new Monte Carlo draws, we built a global common-k band from the
intersection of reliable modes and evaluated the full low-dimensional structure.

Primary results:

- joint_modes = 26
- global_rank1_energy = 0.9997863585
- global_s2_over_s1 = 0.014574
- median_relative_residual (rank-1 recon) = 0.01601
- max_relative_residual (rank-1 recon) = 0.20948

The max relative residual mode is localized at:

- (z, sigma8, k) = (0.5, 0.75, 0.785398163397)
- true value = -0.00565665
- rank-1 value = -0.00684159
- absolute error = 0.00118494

Amplitude manifold:

- A_rank1_energy = 0.9999854613

This supports:

    A(z, sigma8) ~= g(z) alpha(sigma8)

with fitted sigma8 exponents (log A = gamma log sigma8 + const):

- gamma(0.5) = 1.4952
- gamma(1.0) = 1.3969
- gamma(2.0) = 1.4188
- gamma(5.0) = 1.5097

Leave-one-out predictive test (C.3 style):

- Hold out one run, fit global shape + separable amplitude model on remaining 11.
- Predict held-out spectrum without refitting that run.

Aggregate hold-out metrics:

- median(median relative error, all modes) = 0.02079
- median(median relative error, bulk modes) = 0.02453
- median(median relative error, tail modes) = 0.00858
- max relative error over all hold-outs/modes = 0.23104
- median correlation(true, predicted) = 0.999905
- min correlation(true, predicted) = 0.999404

Current interpretation:

    Lambda(k,z,sigma8) ~= g(z) alpha(sigma8) F(k)

is now supported at high precision for the current simulator and reliable-k band.

Artifacts:

- global_transport_lowdim_summary.json
- global_transport_lowdim_data.npz
- global_transport_loo_prediction.json
