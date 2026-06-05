# Compound-Poisson Generator Investigation

This note sharpens the transport-validation program in the current
`transport-validation` branch.

## What survives from the new picture

The empirical result

    Lambda_eff(k, z, sigma8) ~= g(z) alpha(sigma8) F(k)

still matters, but the simulator suggests a stronger microscopic question:

> can we derive the transport generator directly from the stochastic
> halo-sampling process, instead of fitting it from the output PDF?

That is the right next investigation.

## What the code is exactly doing

The Monte Carlo driver in [cpp/lensing.cpp](/Users/baltabay/Desktop/gw-wl-emulator/cpp/lensing.cpp:427)
does the following for each realization:

1. Fix a threshold `kappa_thr` through `Nhalos`.
2. Add a weak-lensing Gaussian background `kappa_W`.
3. Loop over redshift and mass cells.
4. Draw halo and filament event counts in each cell.
5. Sample event geometry and add contributions to:

       Y = (kappa, gamma1, gamma2)

6. Convert the final summed field to:

       mu = 1 / ((1 - kappa + <kappa>)^2 - gamma^2)
       x = ln mu

with `<kappa>` estimated from the ensemble and subtracted afterward.

## Important correction: the exact additive variable is not `x = ln mu`

The new compound-Poisson idea is almost right, but one detail matters:

- the code is additive in `Y = (kappa, gamma1, gamma2)`,
- not additive in `x = ln mu`.

Each event contributes linearly to `(kappa, gamma1, gamma2)`, but `ln mu`
is only computed after the full sum. Therefore

    x != sum_i xi_i

for the current simulator, unless one introduces an additional weak-lensing
or single-jump approximation.

So the exact Lévy-Khintchine object is the characteristic exponent of the
raw additive field:

    Psi(q, z)
    = int dJ R(J, z) [exp(-i q . J) - 1]

where `J = (delta kappa, delta gamma1, delta gamma2)`.

## What is exact, and under which switches

### Cleanest exact setting

Use:

- `bias = 0`
- `filaments = 0`
- raw field `(kappa, gamma1, gamma2)` before mean-kappa subtraction

Then each `(z_l, M)` cell is sampled with Poisson counts and independent
marks. In that regime the additive raw field is a genuine compound Poisson
sum, plus the explicit Gaussian weak-lensing background already present in
the code.

One implementation caveat matters:

- for cells with very small expected count, the simulator uses a Bernoulli
  shortcut `U < lambda * barN` instead of drawing the full Poisson count
  distribution,
- only larger-rate cells use an explicit Poisson draw.

So the current production code is best described as a Poisson/Bernoulli
hybrid approximation to the ideal compound Poisson process. The exact
Lévy-Khintchine generator is therefore the clean continuum target, while the
current Monte Carlo may show small but real deviations from it even in the
`bias=0`, `filaments=0` baseline.

The exact generator is then:

    Psi(q, z_s)
    = -1/2 sigma_kappaW(z_s)^2 q_kappa^2
      + sum_cells lambda_cell(z_s)
        <exp(-i q . J) - 1>_cell

where the mark average is over impact parameter, polar angle, and halo
orientation.

### When `bias = 1`

The counts are modulated by a lognormal factor:

    lambda -> lambda * exp(delta_b - sigma_b^2 / 2)

That is no longer ordinary Poisson. It is closer to a Cox / doubly-stochastic
Poisson process, so the simple Lévy exponent only holds conditionally on the
bias field.

### When using the published `ln mu` output

Two extra complications enter:

- `ln mu` is a nonlinear pushforward of the additive field,
- the mean-kappa subtraction couples the samples through the estimated
  ensemble mean.

Neither destroys the investigation, but both mean that

    Lambda_x(k) = (1 / z) log P_hat_x(k, z)

is not the direct Lévy exponent of independent `ln mu` jumps.

## Recommended validation sequence

### Stage A: exact generator in additive-field space

Validate the theorem where it is genuinely exact:

1. work with `bias=0`, `filaments=0`
2. measure the Monte Carlo characteristic function of

       Y = (kappa, gamma1, gamma2)

3. compare it to a theoretical generator built from the same cell rates and
   mark distributions

This is the clean generator-versus-generator test.

In practice, with the current implementation, "clean" means:

- exact at the level of the intended stochastic continuum model,
- and approximate at the level of the realized simulator because of the
  low-rate Bernoulli shortcut.

### Stage B: push forward to `ln mu`

After Stage A works, transform the raw samples to

    x = ln mu(kappa, gamma1, gamma2)

and compare the induced `x`-space spectrum to the existing
`Lambda_eff(k, z)` pipeline.

This isolates the genuine nonlinearity from any mistake in the underlying
stochastic generator.

### Stage C: turn on realism one ingredient at a time

Then add back:

1. filaments
2. bias / clustering
3. mean-kappa subtraction and source-plane conventions

That gives a controlled map from exact baseline to full production model.

## Implementation note

This branch now exposes a raw-field Python API:

    gwlensing.sample_lensing_raw(...)

returning arrays for:

- `kappa`
- `gamma1`
- `gamma2`
- `mean_kappa`

This is the right observable for the exact stochastic-process validation.

## Bottom line

The generator program is still the right next move, but the exact object is
slightly different from the original claim:

- exact theory lives first in additive field space `(kappa, gamma1, gamma2)`,
- `ln mu` is a derived nonlinear observable,
- the pure compound-Poisson statement is cleanest in the `bias=0`,
  `filaments=0` baseline.

That is still strong news: it means the current empirical transport result may
be the pushed-forward shadow of a more fundamental, exactly defined generator.
