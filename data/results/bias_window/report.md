# bias_window — window generalization of the clustering layer (2026-07-20)

Implements `docs/bias_window_design_plan.md`: the bias field's smoothing window
becomes a selectable enum, with the spherical top-hat as the candidate headline
and a Gaussian as the shape-robustness check.

```
bias_window = 0   transverse disk,   W(x) = 2 J1(x)/x,               x = k_perp R   (legacy, bitwise default)
              1   spherical top-hat, W(x) = 3(sin x - x cos x)/x^3,  x = |k| R
              2   Gaussian,          W(x) = exp(-x^2/2),             x = |k| R
```

**Status: all gates passed; NO default flipped.** `bias_window = 0` remains the
default pending user/supervisor sign-off (bundle with the pending R_perp = 8441
sign-off — see "Open decisions"). Nothing committed; working tree left for review.

MC tables: `tables.md` (regenerate with `bias_window_scan.py report`).
Figure: `plots/bias_window_scan.png`. Raw shards: `mc/` (976 + 32 npz).

---

## 1. Headline results

1. **The window generalization is safe.** The legacy path is bitwise unchanged
   (verified against a pristine-HEAD build, not just self-consistency), and every
   validation layer passes for the two new windows.
2. **At the production R_perp = 8.44 Mpc the window shape does not move P(lnmu).**
   JSD(top-hat, disk) sits **at the shard-half floor** at z_s = 0.5, 1 and 5.
   The measurable difference is a **+3.1% / +1.3% / +0.8%** shift in clipped
   Var(lnmu) — statistically real (SE 0.29%) but small.
3. **The candidate default R is unchanged: 8441 kpc = R_L(1e14 M_sun).** The
   clustering-variance-weighted signal scale is a property of the halo
   population, not of the window: all three windows give *identical* weighted
   R_L quantiles (§4.2).
4. **Window shape is not a free knob once R is fixed by variance.** A Gaussian
   at the variance-matched radius R_G = 3972 kpc is indistinguishable from the
   top-hat at R_TH = 8441 kpc (JSD at floor, z_s = 1 and 5) — the
   robustness statement for the paper.
5. **The top-hat earns an exact identity the disk cannot have:**
   sigma^2_point(delta_1D) = sigma^2(R) holds to 1.0000 (0.9997 including
   production's mode cutoff). The disk window misses it by 19%, and 4% of its
   field variance is removed by the numerically-defined cutoff alone (§3.5).

---

## 2. Files touched

C++ (production):
- `cpp/lensing.h` — `LensingConfig::bias_window` + enum semantics, the
  "R is used as given" note for the Gaussian.
- `cpp/lensing.cpp` — `biasWindow2()` (top-hat/Gaussian W~^2, series below
  x = 1e-2); `BiasField1D::window` + `build(..., int win = 0)`; window applied
  at |k| inside the k_par loop; two guards (range, requires `bias_model = 1`);
  call site passes `cfg.bias_window`.
- `cpp/lnmu_wrapper.h/.cpp` — `SamplingParams::bias_window` → `LensingConfig`
  (2 sites). *Not in the plan's file list; the plumbing goes through this layer.*
- `cpp/python_bindings.cpp` — `bias_window` on **all five** entry points
  (`sample_lnmu`, `sample_lnmu_ml`, `sample_lnmu_ml_with_diagnostics`,
  `sample_lensing_raw_ml`, `compute_lnmu_stats`; the plan said three) +
  `get_simulator_config`.

Python replicas (verbatim ports — the gates depend on them staying verbatim):
- `playground/bias_field/validate_field_covariance.py` — `Wtophat`, `Wgauss`,
  `window2_iso`, `window=` on `cpp_field`/`notebook_cov`/`compare`/`weak_check`,
  `--gridcheck` and `--window=N` modes.
- `scripts/convergence/bias_field_prototype.py` — `_Wiso2`, `LOSField(C, window)`.
- `paper_prod/scripts/plot_fig_clustering_field.py` — `P1D_curve(..., window)`.
  **Figure NOT regenerated** (orchestrator scope, post-flip).

New:
- `tests/test_bias_window.py` — 14 tests.
- `playground/bias_window_field_probe.cpp` — dumps the *production* BiasField1D
  covariance (includes `lensing.cpp`).
- `playground/bias_field/check_cpp_vs_replica.py` — C++-vs-replica gate.
- `scripts/convergence/bias_window_sigmaR.py` — sigma(R), matched R_G, identity check.
- `scripts/convergence/bias_window_weighted_scale.py` — weighted signal scale per window.
- `scripts/convergence/bias_window_scan.py` — the MC scan (this directory).

Incidental: `data/models/test_nsf_checkpoint_smoke_history.json` is rewritten by
`tests/test_train_nsf_smoke.py` on every full-suite run — test residue, not an
intentional edit; `git checkout` it if unwanted.

---

## 3. Gates

### 3.1 Bitwise default — PASS (stronger than specified)

The plan asked for a comparison "against a pre-change run". Rather than rely on
self-consistency, HEAD was checked out into a scratch worktree, built
independently, and compared stream-for-stream:

| check | result |
|:--|:--|
| pristine-HEAD build vs patched build, 8 configs x 4 arrays (kappa, kappa_weak, gamma1, lnmu) | **32/32 bitwise equal** |
| same, with `bias_window=0` passed EXPLICITLY | **32/32 bitwise equal** |
| `tests/test_cosmology_params.py` (incl. `test_backward_compat_bitwise` vs the pre-6d reference npz) | 11/11 pass |
| `tests/test_bias_window.py` | 14/14 pass |
| full suite | 104 passed, 1 skipped, **1 pre-existing failure** (`test_phase3b_local_density`, stale docs path — documented in CLAUDE.md) |

Configs covered: legacy (bias_model=0); counts-only field at z_s = 0.5/1/5;
non-default R_perp 3000 and 20000; weak arm at z_s = 1/5.

### 3.2 k_perp quadrature convergence — PASS, no constants changed

`validate_field_covariance.py --gridcheck` doubles both grid constants
(nkperp 2048→4096, kperp_hi 60/Rw→120/Rw) and compares the P_1D table
(gate: < 1e-4). z_s = 1:

| window | R = 3000 | R = 8441 | R = 20000 |
|:--|--:|--:|--:|
| disk | 8.9e-06 | 1.2e-05 | 1.6e-05 |
| top-hat | 2.6e-05 | 3.3e-05 | 4.3e-05 |
| Gaussian | 5.3e-13 | 6.0e-14 | 2.2e-13 |

The disk window is the slowest-decaying of the three (W~^2 ~ x^-3, vs x^-4 for
the top-hat and exponential for the Gaussian), so the constants tuned for it
bound the new windows. No per-window constants needed.

### 3.3 Covariance layer (replica vs independent reference) — PASS

`validate_field_covariance.py` (all three windows, z_s = 1 and 5, R = R_L(M) for
M = 1e5..1e20): |dCov|/diag <= **9.0e-6** (worst case 4.0e-5 at N_max = 4, the
mode floor), dsig <= 4.5e-6, Cholesky reproduces Cov to 1e-12,
**<lambda> = 0.9984-1.0011** (gate: < 0.5%/shell). Weak-arm layer:
sum(v) = sigma_W^2 to 5.5e-16, log-table interp errS <= 3.3e-3, errV <= 5.3e-4,
<V>/sigma_W^2 = 0.997-1.001. All in family with the disk-window reference
(~3e-6, 0.2%).

### 3.4 Production C++ vs replica — PASS (gap the plan's gate left open)

The layer above compares two *python* integrals — it validates the derivation,
not the shipped C++. For window 0 the bitwise stream check covers the C++ side,
but **windows 1/2 are new code with no stream reference**: a window applied to
k_perp instead of |k| would still be deterministic, finite and non-inert, and
would pass every other gate. `playground/bias_window_field_probe.cpp` (which
`#include`s `lensing.cpp`, so its `BiasField1D` *is* production's) dumps sig2 and
the shell covariance; `check_cpp_vs_replica.py` compares them:

| | gate | achieved (3 windows x z_s {1,5} x R {3000, 8441, 20000}) |
|:--|--:|--:|
| max \|dCov\|/diag | 5e-6 | **<= 7.4e-13** |
| max \|sigma_cpp/sigma_py - 1\| | 5e-6 | **<= 1.4e-13** |

Roundoff-level. The chain notebook-reference <-> replica <-> production C++ is
closed for all three windows.

### 3.5 Identity check: sigma^2_point(delta_1D) == sigma^2(R)

For an isotropic window the pencil-projected field's point variance must equal
the 3D smoothed variance (the cylindrical measure reassembles d^3k). At
R = 8441 kpc (`bias_window_sigmaR.py`):

| window | sigma_point (untruncated) | sigma(R), 3D | ratio | with k_max = 2pi/R | ratio |
|:--|--:|--:|--:|--:|--:|
| top-hat | 0.972430 | 0.972430 | **1.0000** | 0.972116 | 0.9997 |
| Gaussian | 0.578349 | 0.578349 | **1.0000** | 0.578349 | 1.0000 |
| disk | 1.159783 | 0.972430* | 1.1927 | 1.112951 | 1.1445 |

\* the disk is not a 3D window; quoted against the top-hat's sigma(R) to show
the size of the miss.

Two things follow. (a) An independent end-to-end confirmation that the C++
applies the window at |k| — this test is *designed* to fail if it were on
k_perp, and window 0 duly fails it. (b) The mode cutoff is doing real physical
work for the disk window and essentially none for the isotropic ones: truncating
at k_max removes **4.0%** of the disk field's sigma but only **0.03%** of the
top-hat's. That is the "a physical scale is coupled to a numerical cutoff"
concern, quantified.

### 3.6 Weak-arm composition — PASS

Determinism (seed-for-seed), the `bias_model != 1` guard throw, and the
`bias = 0` no-op are covered by `tests/test_bias_window.py` for windows 1 and 2,
with and without `bias_weak`.

**Split invariance** (moving kappa_thr via <N> = 100 -> 300 only moves mass
between the explicit and weak domains, both riding the same field), 240k/arm.
A **disk control was added** so the top-hat's residual is judged internally
rather than against the 2026-07-16 record:

| z_s | top-hat JSD | disk control JSD | floors |
|--:|--:|--:|:--|
| 1 | 3.36e-04 | 3.40e-04 | 1.8-2.4e-04 |
| 5 | 5.56e-04 | 5.91e-04 | 3.0-3.9e-04 |

The residual is **identical to the disk's** (slightly smaller at z_s = 5), and
both match the accepted 2026-07-16 disk numbers (3.1e-4, 5.3e-4). It is the
known Cox-split approximation class — the split is exact in mean and variance,
but the shuffled band trades Poisson skewness for a Gaussian — and the window
generalization neither causes nor worsens it. Excess over floor: ~1.3e-4 (z_s=1),
~2.4e-4 (z_s=5), both within the plan's `<= 3e-4` criterion.

---

## 4. Physics

Protocol: 16 subprocess shards x 15k = 240k realizations/arm, `kappa_anchor=1`,
fixed-<N>=100, subhalo off; JSD on P(lnmu) with 200 bins; floor = JSD between
shard halves. Var_clip = Var(lnmu) after a 10-sd median clip. Full tables in
`tables.md`.

### 4.1 Top-hat R-scan

Same R grid as `rperp_pdf_scan` (R_L(M), M = 1e7..1e20), now at z_s = 0.5, 1, 5.
The scan reproduces the disk-window shape: the clustering signal rises
monotonically as R shrinks and dies into the floor for R >~ 18-39 Mpc.

At the production R_perp = 8.44 Mpc:

| z_s | sigma(lnmu) top-hat | nobias | legacy | JSD vs nobias | floor |
|--:|--:|--:|--:|--:|--:|
| 0.5 | 0.0399 | 0.0304 | 0.0365 | 1.56e-03 | 1.5e-04 |
| 1 | 0.0761 | 0.0695 | 0.0780 | 7.97e-04 | 2.0e-04 |
| 5 | 0.2359 | 0.2306 | 0.2522 | 2.78e-04 | 3.3e-04 |

Note the clustering signal at z_s = 5 and the production R is **at the floor**
for both windows (top-hat 2.78e-4, disk 2.94e-4, floor 3.3e-4) — consistent with
the known result that the clustering share is a low-z_s effect.

**Window at fixed R (top-hat vs disk), matched 120k-vs-120k JSD so the floors
are directly comparable:**

| z_s | JSD (half A) | JSD (half B) | floor (top-hat) | floor (disk) | verdict |
|--:|--:|--:|--:|--:|:--|
| 0.5 | 1.57e-04 | 1.72e-04 | 1.50e-04 | 1.60e-04 | at floor |
| 1 | 2.16e-04 | 2.01e-04 | 1.99e-04 | 2.17e-04 | at floor |
| 5 | 2.41e-04 | 3.28e-04 | 3.28e-04 | 3.43e-04 | at floor |

Clipped variance *is* sensitive enough to resolve the difference:
Var_clip(top-hat)/Var_clip(disk) = **1.031 / 1.013 / 1.008** at z_s = 0.5/1/5
(SE 0.29%).

**Why the top-hat gives slightly MORE Var(lnmu) despite LESS field variance.**
At fixed R the top-hat's per-shell variance is ~8% *below* the disk's, so the
naive expectation is a smaller effect. But kappa integrates the field
*coherently* along the line of sight, so it responds to sum_ij Cov_ij, not to
the point variance:

| z_s | trace = sum_i sig2_i (top-hat/disk) | coherent = sum_ij Cov_ij (top-hat/disk) |
|--:|--:|--:|
| 0.5 | 0.916 | **1.056** |
| 1 | 0.922 | **1.058** |
| 5 | 0.933 | **1.059** |

Mechanism: at small x the top-hat filters *less* (W = 1 - x^2/10 vs the disk's
1 - x^2/8) and at large x it filters *more* (x^-4 vs x^-3). It therefore keeps
more of the large-scale, LOS-coherent power — the part lensing integrates — while
discarding more of the small-scale power that averages out. The measured
clustering increment over `nobias` is 1.13-1.20x the disk's, i.e. the same
direction and somewhat larger than the +5.8% coherent-power difference; the
amplification is expected (lambda is lognormal, plus weak<->strong cross terms)
but is *not* claimed here as a quantitative accounting.

### 4.2 Clustering-variance-weighted signal scale — window-independent

`bias_window_weighted_scale.py`. The 2026-07-16 weight w = (barN kappabar b Dg)^2
is reproduced exactly (target q50/q90/q99 = 8.3/14/19 Mpc at z_s = 0.2, 6.1/11/17
at z_s = 1, 3.3/7.6/12 at z_s = 5, 2.8/7.1/12 at z_s = 10):

| z_s | weight | q50 | q90 | q99 |
|--:|:--|--:|--:|--:|
| 0.2 | pop (2026-07-16) | 8.25 | 14.19 | 19.35 |
| 0.5 | pop | 7.06 | 13.13 | 17.91 |
| 1 | pop | 6.05 | 11.25 | 16.57 |
| 5 | pop | 3.25 | 7.63 | 12.15 |
| 10 | pop | 2.79 | 7.06 | 12.15 |

Adding the field-consistent shell factor sigma_i^2(window) changes the quantiles
by <~ 1 grid step and gives **identical results for disk, top-hat and Gaussian**
at every z_s (full table in the script output). This is structural, not
numerical: within a shell every cell rides the same field, so the window enters
only as a per-shell factor and cancels out of the weighted R_L distribution.

**Candidate default under the top-hat: unchanged, R_perp = 8441 kpc = R_L(1e14).**
The honest caveat from 2026-07-16 carries over verbatim: on internal grounds
anything in ~4-12 Mpc is defensible (the weighted median drifts 8.3 -> 2.8 Mpc
over z_s 0.2 -> 10). Not changed here — user/supervisor decision.

### 4.3 Gaussian at matched variance

sigma^2_G(R_G) = sigma^2_TH(8441 kpc) gives **R_G = 3972.1 kpc = 3.972 Mpc**
(R_G/R_TH = **0.4706**, sigma = 0.972430 both sides) — inside the plan's
expected 0.4-0.5 band. `bias_Rperp` is used as given; there is no internal
rescaling, so this number must be passed explicitly.

Comparison run at (window=2, bias_Rperp=3972.1), matched 120k-vs-120k:

| z_s | JSD(A) | JSD(B) | floor (gauss) | floor (top-hat) | sigma(lnmu) G / TH | Var_clip G / TH |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1.96e-04 | 1.82e-04 | 1.85e-04 | 1.99e-04 | 0.0749 / 0.0761 | 0.988 |
| 5 | 3.27e-04 | 3.04e-04 | 3.47e-04 | 3.28e-04 | 0.2353 / 0.2359 | 1.012 |

**At floor at both redshifts.** Once R is fixed by matching the smoothed
variance, the window's shape is immaterial to P(lnmu) — the robustness sentence
for the paper. (Consistent with §4.1's coherence table: at matched variance the
Gaussian's coherent power is within 1% of the top-hat's.)

### 4.4 Fiducial magnitudes and the low-mu edge

Weak arm on vs counts-only, at R_perp = 8441:

| z_s | Var_clip increase, top-hat | disk (this run) | 2026-07-16 disk record | JSD(joint, counts-only) top-hat | disk |
|--:|--:|--:|:--|--:|--:|
| 0.5 | **+23.8%** | +27.1% | +20-24% | 6.82e-03 | 6.61e-03 |
| 1 | **+19.0%** | +19.6% | +18-20% | 4.95e-03 | 4.40e-03 |
| 5 | +15.7% | +13.5% | — | 2.41e-03 | 1.96e-03 |

The top-hat reproduces the disk-window sizing; the disk arm in this run
reproduces the 2026-07-16 record at z_s = 1 and sits marginally above it at
z_s = 0.5 (different anchor/clip convention — this run uses `kappa_anchor=1`).

**Low-mu edge** (q01/q05 of lnmu), the quantity the emulator's edge/flux
calibration is tied to:

| z_s | counts-only disk -> top-hat (q01) | joint disk -> top-hat (q01) | counts-only -> joint, top-hat (q01) |
|--:|--:|--:|--:|
| 0.5 | -0.0373 -> -0.0375 (-0.0002) | -0.0439 -> -0.0442 (-0.0003) | -0.0375 -> -0.0442 (**-0.0067**) |
| 1 | -0.0931 -> -0.0930 (+0.0001) | -0.1066 -> -0.1073 (-0.0007) | -0.0930 -> -0.1073 (**-0.0143**) |
| 5 | -0.3111 -> -0.3108 (+0.0003) | -0.3410 -> -0.3433 (-0.0023) | -0.3108 -> -0.3433 (**-0.0325**) |

**Verdict: the window change does not move the edge** (shifts <= 0.0007 in lnmu
at z_s <= 1, i.e. 20x smaller than the weak arm's). The already-recorded
requirement stands unchanged: the *weak arm* moves the edge and the emulator's
edge/flux calibration must be refit after retraining. No new recalibration is
introduced by `bias_window`.

---

## 5. Caveats

- Tail columns in `tables.md` (q99, q99.9) are reported for completeness only.
  Per the standing rules they are **not** converged quantities in this method
  class and must not be quoted as such.
- Var_clip uses a 10-sd median clip, not the kappa_tot <= 1 core mask (the scan
  driver stores lnmu only). It is the right instrument for *relative* comparisons
  between arms at fixed protocol, which is all it is used for here.
- The 4.1 "coherent power" explanation is a mechanism, not a calibrated model;
  the 1.13-1.20x clustering increment vs 1.058x coherent power is not accounted
  for quantitatively.
- z_s = 5 clustering at the production R is at the JSD floor for both windows, so
  the z_s = 5 window comparison has little dynamic range by construction.

## 6. Open decisions (user / supervisor — deliberately not resolved here)

1. **Flip the default 0 -> 1?** Everything needed is in hand: gates pass, the
   legacy path is bitwise safe, the PDF impact at the production R is at floor,
   and the top-hat is the window the paper's sigma(R) already refers to. Bundle
   with the pending R_perp = 8441 sign-off for Ville.
2. **R under the top-hat.** The weighted-scale calc is window-independent, so
   8441 kpc = R_L(1e14) carries over unchanged. The ~4-12 Mpc latitude noted on
   2026-07-16 is unchanged too.
3. **Gaussian: paper-visible or internal?** It is a clean one-line robustness
   statement ("at matched sigma(R) the window shape is immaterial: JSD at the
   sampling floor"), but it needs the R_G != R_TH explanation to avoid confusing
   a reader.
4. Whether the paper's R quote stays "R_L(1e14)" — unaffected by this work.
