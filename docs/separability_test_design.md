# Spectral Separability Test Design

## Core hypothesis

The effective transport framework has real content only if:

    Lambda(k, z) ~= lambda(z) F(k)

with:

- one approximately universal spectral shape F(k)
- and a redshift-dependent amplitude schedule lambda(z)

If this fails strongly, the framework reduces to a single-redshift
reparameterization with no predictive power.

---

# Phase 1: Extract spectra

For each redshift:

1. Sample ln(mu) realizations from gwlensing
2. Construct P(x), x = ln(mu)
3. FFT to obtain:

       P_hat(k)

4. Extract:

       Lambda_eff(k, z) = (1 / z) log P_hat(k, z)

5. Apply reliability mask:

       |P_hat(k)| > 2e-2

Only reliable modes participate in the comparison.

---

# Phase 2: Spectral collapse

Normalize each spectrum by a characteristic amplitude.

Candidate choices:

- max |Re Lambda|
- plateau amplitude
- RMS over reliable band

Define:

    Lambda_tilde(k, z) = Lambda(k, z) / A(z)

The central question:

> do all normalized spectra collapse onto one curve?

This is the strongest evidence for a universal propagator structure.

---

# Phase 3: Quantitative collapse metric

At each reliable k:

1. compute mean normalized spectrum
2. compute variance across redshifts
3. define:

       collapse_error(k)

The framework survives only if:

- collapse error remains small and smooth
- shape variations are much smaller than amplitude variations

---

# Phase 4: Semigroup consistency

The transport picture predicts:

    P_hat(k, z1 + z2)
    ~=
    P_hat(k, z1) P_hat(k, z2)

or more generally:

    P_hat(k, z)
    =
    exp[ integral Lambda(k, z') dz' ]

Violation indicates:

- hidden variables
- non-Markovian structure
- or failure of effective closure.

---

# Phase 5: Leave-one-out prediction

This is the actual falsification test.

Procedure:

1. fit universal shape F(k)
2. fit lambda(z) using all but one redshift
3. predict held-out redshift
4. reconstruct PDF
5. compare to Monte Carlo result

A true transport theory predicts held-out PDFs.

A reparameterization cannot.

---

# Interpretation logic

## Strong success

- spectral collapse works
- semigroup approximately holds
- held-out prediction succeeds

=> effective transport structure exists.

## Weak success

- collapse approximately works
- prediction partially succeeds

=> useful compression but not full theory.

## Failure

- shapes vary strongly with redshift
- semigroup fails
- leave-one-out prediction fails

=> no universal transport operator exists.

---

# Important warning

Single-redshift reconstruction is NOT validation.

The FFT and inverse FFT trivially cancel.

Only:

- cross-redshift prediction
- and universal spectral structure

provide non-circular evidence.
