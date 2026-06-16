# autoresearch — NSF emulator high-z far-tail fix

Adapted from karpathy/autoresearch. The LLM runs its own research loop to improve
the weak-lensing magnification emulator, specifically the **strong-lensing far tail
at high redshift**, which the production Conditional NSF under-covers.

## The problem (grounded in diagnostics)

The production flow (`data/models/conditional_nsf_backend_current.pt`) models
`p(lnmu | z, h, OmegaM, sigma8)` with a Zuko rational-quadratic NSF. Diagnosis:

- `lnmu_std = 0.2432`, spline `bound = 5.0` → spline knots only span ±5σ →
  `mu ∈ [0.30, 3.42]`. Beyond that the transform is identity into a **standard
  normal base**, whose Gaussian tail decays far faster than the true
  `dP/dmu ∝ mu^-3` strong-lensing tail.
- Consequence: NSF survival `P(mu>t)` collapses to ~0 at `mu ≈ 3.4` for *every* z,
  while the simulator keeps real mass to `mu ≈ 5–8` at high z (P(mu>5) ≈ 0.6–1%).
- The far tail **is present in the training data** (mu up to thousands), so this is
  a model/representation problem, not missing data.

## Goal

Lower the **SCORE** metric (lower = better) reported by `train_ar.py`, which combines:

- **TLSE** (primary): Tail Log-Survival Error — mean over high-z eval points and
  thresholds t∈{2,3,4,5} of `|log10 S_nsf(mu>t) - log10 S_sim(mu>t)|`. Directly
  measures the orders-of-magnitude tail gap.
- **BodyNLL** (guard): mean NLL of the flow on simulator samples in the science
  window `lnmu ∈ [-0.5, 2.5]`. Must not regress — guards against trading body
  fidelity for tail fidelity.
- `SCORE = TLSE + 0.3 * max(0, BodyNLL - BODY_BUDGET)`.

The ground truth (cached simulator survival + samples) is fixed in `prepare_ar.py`
and must NOT be modified — it is the honest metric.

## Files

- `prepare_ar.py` — **FIXED, do not modify.** Caches simulator ground truth and
  computes the SCORE/TLSE/BodyNLL metric from a trained model. The honest harness.
- `train_ar.py` — **the file you edit.** Model architecture (base distribution,
  bins, bound, transforms, hidden width), optimizer, loss (incl. tail reweighting),
  training loop. Trains on `datasets_logz_1k`, saves to `models/`, then prints the
  metric block.
- `results.tsv` — experiment log (tab-separated). NOT committed; leave untracked.
- `models/` — model checkpoints (gitignored).

## What you CAN do (in `train_ar.py`)

Everything about the model and training is fair game: base distribution (Normal,
StudentT, ...), spline `bins`/`bound`, `num_transforms`, `hidden_features`, lr,
weight decay, epochs, batch size, loss reweighting of tail samples, lnmu
standardization/transform, learning-rate schedule.

## What you CANNOT do

- Modify `prepare_ar.py` (the metric / ground truth).
- Modify anything outside `ml/autoresearch/` (shared with another agent).
- Write to shared paths (`data/models/*`, the shared preprocessing stats). All
  outputs stay under `ml/autoresearch/`.

## The experiment loop

LOOP:
1. Note the git state (branch `autoresearch/jun16`, current commit).
2. Edit `train_ar.py` with one experimental idea.
3. `git commit -am "<idea>"`.
4. Run: `KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. <py> ml/autoresearch/train_ar.py > ml/autoresearch/run.log 2>&1`
5. Read: `grep "^SCORE:\|^TLSE:\|^BodyNLL:" ml/autoresearch/run.log`. Empty → crash;
   `tail -50 run.log` for the trace, fix if trivial else discard.
6. Log to `results.tsv`.
7. If SCORE improved (lower), keep (advance the branch). Else `git reset --hard`
   back to the previous kept commit.

First run = baseline (reproduce the production config) to establish BODY_BUDGET and
the SCORE reference.
