# Autoresearch run: high-z far-tail fix for the NSF emulator

Run on branch `autoresearch/jun16` (worktree `gw-wl-emulator-ar`), env: conda `test`
(py3.12, torch 2.12, zuko 1.6), device MPS. Method adapted from karpathy/autoresearch:
edit one hackable trainer, train, score against a fixed simulator metric, keep/revert.

## Problem

Production flow `conditional_nsf_backend_current.pt` under-covers the strong-lensing
far tail at high z: survival `P(mu>t)` collapses to 0 at `mu ~ 3.4` for every z, while
the simulator keeps mass to `mu ~ 5-8`. The far tail **is in the training data**, so
this is a model/representation problem.

## Root cause (verified)

zuko RQS spline `bound=5.0` + `lnmu_std~0.16-0.24` ⇒ the spline only covers `mu <~ 3.4`.
Beyond it the transform is identity into a **standard-normal base**, whose Gaussian tail
decays far faster than the true `dP/dmu ~ mu^-3`. Direct check: at standardized lnmu=6,
Normal base log-density = -16.3 (collapsed) vs StudentT(df=4) = -6.8 (heavy tail kept).

## Metric (fixed, `prepare_ar.py`)

- **TLSE** (primary): mean `|log10 S_nsf(mu>t) - log10 S_sim(mu>t)|` over high-z eval
  points (central + high-structure corner, z=2,3.5,5,8) and t=2,3,4,5.
- **BodyNLL** (guard): flow NLL on simulator samples in the science window.
- Ground truth = simulator survivals cached once (`cache/groundtruth.npz`).

## Experiment log (`results.tsv`)

| exp | change | TLSE | BodyNLL | status |
|-----|--------|------|---------|--------|
| baseline | bins8 bound5 normal | 2.805 | -0.209 | keep |
| 1 | bound 5→8 | 1.894 | -0.338 | keep |
| 2 | **StudentT base df=4** | 0.539 | -0.529 | keep |
| 3 | bound 8→12 | 0.293 | -0.526 | keep |
| 4 | bins 8→16 | 0.200 | -0.534 | keep |
| 5 | bins 16→24 | 0.259 | -0.535 | discard (tail undertrained) |
| 6 | tail_weight=10 | 0.368 | -0.509 | discard (overshoot) |
| 7 | tail_weight=3 | 0.143 | -0.531 | keep |
| 8 | **tail_weight=2** | **0.046** | **-0.536** | **WINNER** |

TLSE 2.805 → 0.046 (~60×). Body fidelity and val NLL also improved (no tradeoff).

## Winner config

`bins=16, bound=12, transforms=6, hidden=128, base=StudentT(df=4), tail_weight=2`
(upweight lnmu>1.5 ×3, with weighted validation selection). Saved:
`ml/autoresearch/models/flow_ar.pt`.

## Held-out validation (`validate_ar.py`, points NOT in the metric panel)

Cosmologies `(0.60,0.22,0.70)` and `(0.74,0.39,1.03)`, z = 1.5, 2.5, 4.0, 6.5:

- **Production mean TLSE = 1.643** (survival = 0.00000 for all mu>4).
- **Winner mean TLSE = 0.225** (7.3× better; tracks the simulator tail).

See `validation_dpdmu.png`: production cuts off at mu≈3; winner follows the simulator
to mu≈6-8 at z = 2, 3.5, 5, 8 (high-structure corner).

## Caveats / next levers

- Held-out TLSE (0.225) > panel TLSE (0.046): some panel-tuning; still a large win.
- `tail_weight=2` slightly *overshoots* the thinnest tails (low-structure, low z).
  A z/structure-dependent weight, or `tail_weight≈1.5`, could trade that off.
- Not promoted to production (isolation from the other agent). To promote: retrain on
  the production dataset/stats and copy into `data/models/`.
