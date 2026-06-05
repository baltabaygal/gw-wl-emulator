# AGENTS.md

## Repo Context

- Repository: `/Users/baltabay/Desktop/gw-wl-emulator`
- Active branch for this investigation: `transport-validation`
- The sibling directory `/Users/baltabay/Desktop/gw-wl-new` is empty; active work has been happening here.

## Working Environment

- Preferred Python environment: `conda` env `test`
- Build command:

```bash
conda run -n test bash ./scripts/build.sh
```

- The Python extension is loaded from:

```text
build/gwlensing*.so
```

## Key Scientific Reframing

### Track A: Generator Validation

The main correction was:

- the exact additive variable in the simulator is **not** `x = ln(mu)`
- the exact additive process lives first in:
  - `kappa`
  - `gamma1`
  - `gamma2`

So the fundamental stochastic object is a jump/Lévy process in additive field space, and `ln(mu)` is a nonlinear push-forward observable.

### Track B: Microscopic Jump Measure

The main objective became:

- measure the single-event jump distribution directly
- estimate the tail exponent `alpha`
- determine whether `alpha` is a real observable or a fitting artifact

## New/Important APIs

### LOS-summed raw samples

Python binding:

- `gwlensing.sample_lensing_raw(...)`

Returns per-realization:

- `kappa`
- `gamma1`
- `gamma2`
- `mean_kappa`

This is for additive-field validation, not single-event tails.

### Event-level samples

Python binding:

- `gwlensing.sample_lensing_events(...)`

Returns one record per sampled encounter with:

- `realization`
- `is_filament`
- `z`
- `M`
- `b`
- `kappa`
- `gamma1`
- `gamma2`

This is the key API for Track B.

Relevant implementation files:

- `/Users/baltabay/Desktop/gw-wl-emulator/cpp/lensing.h`
- `/Users/baltabay/Desktop/gw-wl-emulator/cpp/lensing.cpp`
- `/Users/baltabay/Desktop/gw-wl-emulator/cpp/lnmu_wrapper.h`
- `/Users/baltabay/Desktop/gw-wl-emulator/cpp/lnmu_wrapper.cpp`
- `/Users/baltabay/Desktop/gw-wl-emulator/cpp/python_bindings.cpp`

## Track A Summary

### What was tested

- direct comparison of theory generator vs simulator generator in additive `kappa` space
- clean baseline:
  - `filaments=False`
  - `bias=False`
  - `ell=False`
  - `exact_poisson=True`

### Main conclusions

- the residuals discussed in the validation scripts are **theory-vs-simulator Lambda residuals**
- Bernoulli truncation is **not** the dominant source of mismatch
- much of the earlier discrepancy was numerical:
  - band-edge effects
  - insufficient theory-side quadrature resolution
  - log-branch contamination at large `q`

### Practical status

- the additive-field generator picture survived multiple attempts to kill it
- Track A is scientifically mature enough to support the generator interpretation
- remaining discrepancies are small and mostly tied to numerical/diagnostic details rather than a failure of the basic picture

## Track B Summary

### Core object

For each event, define:

```text
xi = -log((1 - kappa)^2 - gamma^2)
```

with:

```text
gamma = sqrt(gamma1^2 + gamma2^2)
```

Track B focuses on the single-event jump measure `R(xi)`.

### Scripts added

- `/Users/baltabay/Desktop/gw-wl-emulator/python/measure_jump_measure.py`
  - direct event-level measurement of `R(kappa)` and `R(xi)`
  - bulk/tail plots
  - simple tail-family fits

- `/Users/baltabay/Desktop/gw-wl-emulator/python/measure_alpha_grid.py`
  - fixed-window grid measurement of `alpha(z, sigma8)`

- `/Users/baltabay/Desktop/gw-wl-emulator/python/measure_alpha_relative_window.py`
  - turnover-relative window measurement of `alpha(z, sigma8)`

- `/Users/baltabay/Desktop/gw-wl-emulator/python/measure_alpha_estimators.py`
  - compares multiple alpha estimators on the same grid

- `/Users/baltabay/Desktop/gw-wl-emulator/python/measure_alpha_tail_estimators.py`
  - compares relative-window vs Hill/Pareto threshold estimators

### Clean measurement settings

Unless explicitly stated otherwise, Track B measurements used:

- `filaments=False`
- `bias=False`
- `ell=False`
- `exact_poisson=True`
- usually `Nreal=5000`

### Main results so far

#### 1. Heavy-tail structure is real

Direct event-level plots show:

- a broad approximately power-law regime in `R(xi)`
- followed by turnover/cutoff

This supports a truncated-heavy-tail interpretation qualitatively.

#### 2. Fixed-window alpha looked non-universal

Using a common absolute fit window `1e-4 < xi < 1e-2`, mean `alpha` values were approximately:

- `z=0.5`: `1.083`
- `z=1.0`: `1.090`
- `z=2.0`: `1.120`
- `z=5.0`: `1.198`

At fixed redshift, `sigma8` dependence was weak.

#### 3. Most of that redshift trend was a measurement artifact

The turnover scale `xi_break` moves with redshift, so the fixed absolute window samples different parts of the tail.

Using a turnover-relative window, the strong monotonic trend mostly disappeared. Relative-window means became approximately:

- `z=0.5`: `1.044`
- `z=1.0`: `1.015`
- `z=2.0`: `1.021`
- `z=5.0`: `0.888`

Interpretation:

- strong redshift evolution of `alpha` is **not established**
- a universal `alpha ~ 1` is plausible for `z=0.5, 1, 2`
- `z=5` is still measurement-sensitive

#### 4. Alpha is not yet estimator-independent

Current estimator comparison shows:

- fixed-window estimator is biased by turnover motion
- relative-window estimator is presently the most credible
- local-slope and current threshold-based estimators are not yet stable enough to define the observable cleanly

So the current bottleneck is:

- make `alpha` estimator-independent before attempting CGMY/truncated-Lévy closure

## Current Best Interpretation

- Track A: the additive-field generator identity is established well enough for science
- Track B: there is strong evidence for a heavy-tail exponent of order unity, but the precise value of `alpha` is still estimator-sensitive
- the most likely emerging picture is:

```text
alpha ~ 1
```

but that is not yet promoted to a final physical claim

## Recommended Next Steps

1. Stabilize the tail observable.
   - Sweep threshold choice for Hill/Pareto estimators.
   - Look for plateaus in `alpha(threshold)`.

2. Compare multiple defensible estimators on the same support.
   - relative-window fit
   - Hill estimator
   - Pareto/power-law MLE

3. Only after `alpha` becomes estimator-independent:
   - attempt CGMY / truncated-Lévy closure
   - compare `alpha`, Lévy index `Y`, and spectral behavior

## Important Artifacts Already Written

- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid.csv`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid.json`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_vs_z.png`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_vs_sigma8.png`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_histogram.png`

- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_relative.csv`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_relative.json`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_relative_vs_z.png`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_relative_vs_sigma8.png`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_grid_relative_histogram.png`

- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_estimators.csv`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_estimators.json`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_estimators_vs_z.png`

- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_tail_estimators.csv`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_tail_estimators.json`
- `/Users/baltabay/Desktop/gw-wl-emulator/alpha_tail_estimators_vs_z.png`

- `/Users/baltabay/Desktop/gw-wl-emulator/jump_measure_alpha_z0p5_N5k.png`
- `/Users/baltabay/Desktop/gw-wl-emulator/jump_measure_alpha_z0p5_N5k_summary.json`
- `/Users/baltabay/Desktop/gw-wl-emulator/jump_measure_alpha_z0p5_N5k_events.npz`
