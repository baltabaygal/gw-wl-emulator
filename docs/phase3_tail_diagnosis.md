# Phase 3 — High-μ Tail Diagnosis (NSF emulator)

Stress-testing the magnification PDF at a high-structure cosmology
(h=0.72, Ω_M=0.38, σ₈=1.00) over a wide redshift range revealed that the NSF
emulator's `dP/dμ` collapses sharply at **μ ≈ 3.4**, independent of redshift,
with a spike-then-cutoff artifact. The simulator's strong-lensing tail extends
well beyond this (μ ≳ 5–8 at high z). See `ml/plot_dpdmu_comparison.py` and
`ml/diagnose_tail.py`.

## Quantified tail mass — P(μ > threshold), simulator vs NSF

Cosmology (0.72, 0.38, 1.00), simulator nsim=400k, emulator integrated over lnμ.

| z | P(μ>1.5) sim/nsf | P(μ>2) sim/nsf | P(μ>3) sim/nsf | P(μ>4) sim/nsf | P(μ>5) sim/nsf | NSF cutoff μ |
|--:|--|--|--|--|--|--|
| 2.0 | 0.0348 / 0.0333 | 0.0104 / 0.0083 | 0.0034 / 0.0005 | 0.0020 / 0.0000 | 0.0014 / 0.0000 | 3.31 |
| 3.5 | 0.0753 / 0.0700 | 0.0269 / 0.0207 | 0.0092 / 0.0021 | 0.0052 / 0.0000 | 0.0036 / 0.0000 | 3.35 |
| 5.0 | 0.1030 / 0.0909 | 0.0418 / 0.0293 | 0.0151 / 0.0030 | 0.0086 / 0.0000 | 0.0062 / 0.0000 | 3.37 |
| 8.0 | 0.1342 / 0.1196 | 0.0619 / 0.0464 | 0.0248 / 0.0047 | 0.0146 / 0.0000 | 0.0101 / 0.0000 | 3.39 |

- Body and near-tail (μ<2): match within ~10–25%.
- μ>3: NSF misses ~80% of the mass.
- μ>4: NSF mass is exactly **0** at all z (hard cutoff).

## Root cause — spline `bound` too small (architecture, not data)

The cutoff μ (3.31–3.39) is essentially **constant in redshift**, at standardized
lnμ ≈ 4.9. That is the zuko `NSF` rational-quadratic spline `bound` (default
**5.0**). The spline knots only span standardized lnμ ∈ [−5, 5]:

- lnμ_mean = 0.0133, lnμ_std = 0.2432 → spline domain lnμ ∈ [−1.20, 1.23] → **μ ∈ [0.30, 3.42]**.
- Beyond the bound the transform is linear (identity tails); the conditioner cannot
  shape the density there, so it collapses with a boundary artifact.

The histogram bins run to lnμ = 2.5 (μ = 12.2), so the spline cannot represent the
upper half of the model's own declared support. Covering the bin range needs
`bound ≈ (2.5 − 0.013)/0.243 ≈ 10`. This affects **all** checkpoints (set in
`ml/nsf_model.py`); it is pre-existing and not introduced by the log-z retrain.

## Secondary finding — stale preprocessing stats

Both the log-z and legacy checkpoints store identical normalization
(z-mean 4.579, lnμ-std 0.2432), which is impossible if computed from the log-z
dataset (z-mean ≈ 2.5). `train_nsf.py` is reusing stale preprocessing stats rather
than recomputing per-dataset. Non-fatal (train and inference share the same
constants, so the model is self-consistent and performs well — z=0.5 TV = 0.051),
but the z-conditioning is off-center and should be fixed.

## Recommended fix (one retrain)

1. `bound ≈ 10` in the NSF constructor (cover the full bin range in standardized lnμ).
2. `bins ≈ 12` (more spline knots for tail resolution).
3. Recompute preprocessing stats from the actual training dataset.

Science impact of the current gap is modest (P(μ>3) ≲ 2.5% even at z=8), so it is
not urgent for σ₈-from-bulk inference, but it is a real correctness fix for the
high-μ strong-lensing tail and high-z.
