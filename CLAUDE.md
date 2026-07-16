# CLAUDE.md — gw-wl-emulator

Fast ML emulator for the GW weak-lensing magnification PDF, built on the C++ Monte-Carlo
engine from Vaskonen (2026). See `README.MD` for the project overview and `memory/MEMORY.md`
(auto-loaded) for cross-session context.

## 0. This file is a LIVING document
Loaded into every session as project instructions. It is **not fixed** — improve it as
tools, flags, and workflows change; date the changes. Keep **only verified commands**
(ones actually run); delete anything that turned out wrong. If the user teaches you a
preference, capture it here.

## Environment — IMPORTANT

Use the **`test` conda env (Python 3.12)** for anything touching the C++ module or ACE.
The system `python3` on PATH is **3.13** and will NOT work (the C++ module is built for
cpython-3.12, and ACE lives in `test`).

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python      # conda env "test", Python 3.12
# or:  source /Users/baltabay/miniforge3/bin/activate test
```

The env has numpy/scipy/matplotlib, `xgboost`, the built `gwlensing` C++ module, and `ace_lensing`.

### Importing the two compiled/installed packages (from the repo root)

```python
import sys, numpy as np
sys.path.insert(0, 'build')          # gwlensing  (C++ pybind module, build/gwlensing.cpython-312-darwin.so)
sys.path.insert(0, 'ace_lensing')    # ACE: point at the INNER package dir, not the repo subdir
import gwlensing
from ace_lensing import predict_pdf, predict_sigma, predict_mean
```

- **Gotcha — ACE cwd shadowing:** the repo subdir `ace_lensing/` is a namespace dir that
  shadows the real package `ace_lensing/ace_lensing/`. Running `import ace_lensing` from the
  repo root without the `sys.path.insert(0,'ace_lensing')` above gives
  `ImportError: cannot import name 'predict_sigma'`. The insert (or running from any non-repo
  cwd) fixes it. Do **not** `pip install .` ACE — it pins an incompatible numpy (see memory).
- `predict_sigma(...)` returns an array — wrap with `np.ravel(np.asarray(s))[0]`.
- `predict_pdf(Om,h,w,s8,z)` → `(pdf, mu)` over a μ grid (~5000 pts).
- ACE/gwlensing emit harmless xgboost / config warnings on import — ignore.

### `gwlensing` public API
`sample_lnmu`, `sample_lnmu_ml`, `sample_lnmu_ml_with_diagnostics`,
`sample_lensing_raw_ml` (returns raw `kappa`, `gamma`), `compute_lnmu_stats`,
`get_simulator_config`. Cosmology args: `Om`, `sigma8`, `h` (defaults Om=0.315, σ8=0.811);
`Mmin` is exposed; `zs` is the source redshift; `Nhalos=100` sets the strong/weak κ split.
**Grid kwargs (2026-07-11, for the Mmin/Nz convergence studies):** all samplers +
`compute_lnmu_stats` take trailing `NM=100, Nz=100` (mass/z grid sizes); the helpers
`get_kappa_threshold`/`get_expected_halo_count`/`get_sigma_background` take trailing
`Mmin=1e7, NM=100, Nz=100` (previously hard-coded); `get_simulator_config` dict now also
reports `Mmin/NM/Nz`. Defaults verified bit-identical to the unpatched module (Linux
sandbox side-by-side build: all samplers, subhalo-on model 3, explicit-legacy-kwargs
paths); Mac `make build` + all 11 `tests/test_cosmology_params.py` (incl.
`test_backward_compat_bitwise`) passed 2026-07-12. `compute_lnmu_stats` still does
NOT expose `Mmin`.

### 1+6d parameterization (2026-07-07) — Θ = {h, z_eq, Ω_M, Ω_B, A_s, n_s}
All entry points take trailing kwargs `As=-1.0, OmegaB=0.0493, zeq=3402.0, ns=0.965`.
**A_s-mode:** `As > 0` normalizes P(k) directly via the analytic Bunn–White mapping
(`deltaH8 = (2/5)·gfid·√As·(c·k_p/H0)^((1−ns)/2)/Ω_M`, k_p = 0.05 Mpc⁻¹, gfid = the
CPT growth constant in `Dg`; see `cosmology.h::initialize_normalization`) and the
sigma8 positional is IGNORED; `As <= 0` keeps the σ₈ normalization (default path is
bit-identical to pre-change, seed-for-seed). `get_simulator_config(h,OmegaM,sigma8,As,
OmegaB,zeq,ns)` returns `deltaH8`, `sigma8_derived`, `As_derived`, `OmegaR`,
`amplitude_mode`. **Gotchas:** the code's σ₈ uses a smooth-k window (+4.3% vs tophat),
so at Planck A_s=2.101e-9 the derived σ₈ is 0.860 (not 0.811) — expected, not a bug
(`tests/test_cosmology_params.py`, `plots/As_mode_overlay.png`). `zeq` sets
`OmegaR = Ω_M/(1+z_eq)` (freeing z_eq = freeing radiation). σ₈↔A_s round trip is
exact to 1 ULP.

**ML plumbing (ready, not retrained):** `ml/params.py` = single source of truth
(FIDUCIAL, PRIOR_6D, WIDE_6D, `CONTEXT_KEYS = (z, h, Om, lnAs10, Ob, ns, zeq_k)`,
lnAs10 = ln(1e10·A_s), zeq_k = zeq/1000). `generate_dataset.py` samples the 6d wide
box (LHS, As-mode, HDF5 schema 2.0 + per-config `sigma8_derived`); `ml/data.py`
auto-detects schema → X is (N,7) or legacy (N,4). In the **-ar worktree**: train_ar/
train_smooth infer context dim from data (old checkpoints default 4);
`ml/autoresearch/features.py` = shared edge/tail feature maps (legacy 10/14 exact,
6d 16/20); smooth_model dispatches on len(theta) — 6d needs refit `*_6d.json` caches
+ a context=7 body. Retrain order unchanged (see §NSF recipe); regenerate
`param_space_ref`, `groundtruth.npz`, `lowz_aug.npz`, `tail_counts_extra.npz` first.
NOTE: the -ar worktree's `build/` module predates the As kwargs — rebuild there
(or merge branches) before running its gen/validate scripts.

## Building the C++ module
```bash
make build          # -> scripts/build/build.sh  (CMake + pybind11, target gwlensing)
make test           # python/testing_api.py
make pytest         # tests/
```
Rebuild against Python 3.12 (the `test` env interpreter) so the `.so` matches.

## Cosmology constants (C++ internal units: kpc, M⊙, Gyr)
Defaults: Om=0.315, σ8=0.811, h=0.674, ns=0.965, Ωb=0.0493. `rho_c0 = 277.394 h²` M⊙/kpc³;
`c/H(z) = 306.535/Hz(z)` kpc comoving. Grid: M 1e7–1e17 (100, log), z 0.01–10.01 (100, log).
Concentration = `cons14` (Dutton-Macciò 2014, active; `cons16` is commented out). HMF first
crossing = ellipsoidal `pFC` (p=0.3, q=0.8). NFW κ: `kappa(x)=2 kappa0 Fg(x)`, `kappa0=rs·rhos/Σ_c`.
**Note:** `NFWlist[jz][jM]` is 3 elements `{rs, rhos, c}` — the ∂rs/∂M, ∂ρs/∂M derivatives
are commented out in `cosmology.cpp`, despite the spec/header suggesting otherwise.

## Current work — subhalo substructure gate (§4)
Standalone Python (run with system python3.13 OR test env; no C++ needed for these):
- `scripts/subhalo_gate/subhalo_demo.py` — one host + evolved JvdB14 subhalo population (demo plot).
- `scripts/subhalo_gate/subhalo_screen.py` — Campbell-cumulant screen: clump Δ⟨κ²⟩, Δ⟨κ³⟩ vs the model's
  own moments. Slow (~3 min, per-bin quads). Writes `plots/screen_rows.npy`.
- `scripts/subhalo_gate/subhalo_gate.py` — vectorized/spline version; M_min-degeneracy scan, <2 s.
  Validated against `subhalo_screen.py` at the fiducial M_min.
- `scripts/figures/plot_screen.py` — renders `plots/subhalo_screen.png`.
- `scripts/subhalo_gate/ace_gap.py` — baseline 3: ACE vs Vaskonen convergence moments + cross-validation
  of the screen against C++ raw-κ. **Needs the `test` env** (gwlensing + ACE).
- `scripts/figures/plot_gate_verdict.py` → `plots/gate_verdict.png` (summary figure).

**Verdict (gate complete; numbers refreshed 2026-07-02 after the w̃_f fix):** the screen's
unclustered-Poisson clump component is +3.2–3.4% on ⟨κ²⟩, +1.45% on ⟨κ³⟩ (zs=1–5); gate
constants in `subhalo_gate.py` updated to `dK2c, dK3c = 2.307e-5, 1.077e-6`. **However the
full MC substructure effect is larger** — +14.9% on ⟨κ²⟩, +4.5% on ⟨κ³⟩ at zs=5 (brute,
`validate_split_vs_brute.py`) — because within-host clustering + host–clump covariance
dominate over the Poisson term (see docs/subhalo/subhalo_combining.md "Validation"). Still below
the ACE−Vaskonen gap (50–86%), so the gate conclusion stands, but "few %" undersold it.
Host moments: screen vs C++ raw-κ agree to ~0.1%.

**Fix history 2026-07-02** (module rebuilt, 75 tests pass): w̃_f `exp(-0.25)` fix KEPT;
spin-2 double-angle shear REVERTED (single-angle kept by user/supervisor decision — do
NOT re-apply without being asked); max(0, r−r200) floor REVERTED (wrong idea, made
`subhalo_factor` inert; the "94%/11×" split validation was retracted). Full details:
`docs/claude_md_archive.md`.

## NSF emulator — production model (final 2026-07-03, commit 0785dd0)
The smooth production magnification PDF lives in the **`gw-wl-emulator-ar` worktree**
(branch `autoresearch/jun16`, pushed to origin): `ml/autoresearch/smooth_model.py` →
`load_smooth_fn()` = flow body `models/flow_smooth_lowz.pt` + conditional POT tail
(Poisson-regressed exceedance survivals S2/S3/S8(ctx), 3-segment power law, μ⁻²
asymptote, blend at μ_c=1.7) + empty-beam low-μ cutoff (EDGE_W=0.01, 2026-07-11)
+ **flux calibration** (2026-07-11: lnμ shift δ=ln⟨1/μ⟩_model−ln F_trim(ctx) to the
sim's support-restricted flux, `fit_flux_target.py` → `cache/flux_target_fit.json`;
mean panel KL 0.0091→0.0035, z=1 shoulder ~fixed; do NOT target raw ⟨1/μ⟩=1 — sim's
raw mean is corrupted by κ>1 rays). Do NOT use `flow_ar.pt`.
Full experiment log: `ml/autoresearch/REPORT.md` (exp18–24 + 2026-07-11 rounds) + `results.tsv`.

**Validated (REPORT.md 2026-07-03, edge/flux update 2026-07-11):** median KL 0.0073
(= ACE-Lensing's 0.007), post-calibration fine-binned panel KL median 0.0034; PIT
KS ≤2.1% post-cal; posterior recovery pre-cal: h −0.46±0.20 σ_post, Ωm +0.29±0.17
(16×1000-event mocks; post-cal rerun: h −0.20±0.22 = zero-consistent, Ωm +0.34±0.21
unchanged — width-driven, not mean). Edge study 2026-07-11: fitted edge is
oracle-level; Dyer–Roeder is NOT the sim edge (σ8-dep., e^-100 empty-beam prob.);
tail exponent settled μ⁻² image-plane (μ⁻³ = source plane), Poisson race +1400–6800
nats over μ⁻³. **Full findings + gotchas: `docs/edge_tail_flux_note.md`.**

**Retraining recipe (when new/subhalo-corrected training data lands):**
1. regenerate datasets + `cache/lowz_aug.npz` (`gen_lowz_data.py`) — not in git (114MB);
2. `train_smooth.py` (~7 min MPS) → new body;
3. refit `prepare_fix.py` (edge) + `gen_tail_counts.py` + `fit_tail_amplitude.py` (tail)
   + `fit_flux_target.py` (flux calibration, 2026-07-11 — regenerate `cache/flux_grid.npz` too);
4. acceptance gates: `param_space_check.py` (needs fresh `param_space_ref`),
   `validate_kl.py`, `validate_pit.py`, `validate_posterior.py` (control must be ~0σ).

## The `halos` fork — production subhalo port (2026-07-01)
The clean port of the subhalo model into the original Vaskonen code lives in
**`~/Desktop/halos`** (clone of `github.com/vianvask/halos` — the upstream author's repo;
branch `subhalo-production-integration`, commit `133eb00`). Design doc:
`docs/subhalo/subhalo_combining.md` here. Differences vs the prototype in `cpp/` here: the port
adds the r-dependent resolved/unresolved split (option B: reduced smooth host
`(1−f_s,res(r))·M` + Poisson clumps above the dynamic floor `m_res(r)` from the monotone
`r_thr` table), and drops brute mode / per-clump `gslope` removal.

Build (verified 2026-07-01, needs GSL from homebrew):
```bash
cd ~/Desktop/halos
clang++ -std=c++17 -O2 -I/opt/homebrew/include -L/opt/homebrew/lib \
  main_lensing.cpp basics.cpp cosmology.cpp lensing.cpp subhalo.cpp \
  -lgsl -lgslcblas -o lensing
```
No CLI flag for substructure: set `L.subhalo = true` (+ `subhalo_factor`) in
`main_lensing.cpp` by hand. RNG stream is unchanged when off (verified branch isolation).

**HARD RULE (2026-07-08): halos is under the user's exclusive control — Claude NEVER
commits, pushes, builds, or runs anything there, and edits files there ONLY when the
user explicitly asks, under their supervision.**

**Status (2026-07-02): what halos still needs:**
1. **wf bug (the one real fix to port):** halos `subhalo.cpp` uses `af =
   0.815*exp(-1.0)/0.5^0.707` — misreads Giocoli+2007 α_f = 0.815·e^{−2f³}/f^0.707
   (f=½ ⇒ `exp(-0.25)`). Gives w̃_f = 0.893 instead of ≈1.19 (eq. 10 of astro-ph/0611221)
   ⇒ f_s overestimated ~22–26% (measured). Fixed in the emulator; NOT yet in halos.
2. Shear convention: single-angle KEPT everywhere by user/supervisor decision — halos is
   already as desired; do not change.
3. Resolution floor: r-keyed (halos already matches). Set the halos default
   `subhalo_factor` to the converged value from the emulator scan once fixed.
4. halos-only cleanup: `subhalo_m_floor`/`m_floor` knob and `Nsub` table are dead there.
5. Remaining model caveats (both repos, deferred): field-halo c(m,z), untruncated NFW
   clumps; single-angle γ (~0.5% on ⟨γ²⟩, revisit only if shear becomes an observable).
6. **sigmakappaW fix (2026-07-08, supervisor-approved, APPLIED here):** ×2 log-annulus
   measure + Campbell no-subtraction; net σ²_W ≈ 2.10× original, ≤1% on Var(κ_total).
   halos patch applied at user request, left UNCOMMITTED for user review. Details:
   `docs/sigmakappaw_measure_note.md`, `docs/claude_md_archive.md`.
7. **eps_floor parametrized + floor-consistent injection (2026-07-08, emulator only):**
   `sigmakappaW(..., eps_floor=0.001)`; internal caller holds the ABSOLUTE floor at
   0.001·κ_thr_default so `custom_kappathr` sweeps don't drag it; default path verified
   bit-identical (seed 12345); variance partition validated at zs=0.2–10. Analytic
   σ_full(z_s) theory (z^{3/2} law, σ_∞=0.154): `docs/sigma_full_analytic_note.md`.
   Full details + plot/script list: `docs/claude_md_archive.md`.
   **Gotcha:** `paper_prod/scripts/plot_sigma_k_vs_kappa_threshold.py` plots
   `data/variance_sweep_data_z1.npz`, which is the SUBHALO-factor sweep (written by
   `scripts/figures/plot_variance_vs_factor.py`) — its x-axis label "κ_threshold" is
   wrong; regenerate from the new sweep before using in the paper.
9. **Threshold rule — REVERTED to fixed-⟨N⟩=100 default (2026-07-10, user decision;
   supervisor being informed):** the DEFAULT explicit-halo threshold is the legacy
   `<N>=Nhalos` rule again (`kappathr_flat = -1` everywhere: `SamplingParams`,
   `LensingConfig`, all py::args, and the `get_simulator_config` dict). Chosen for
   **predictability** (bounded, constant per-LOS cost — 100 explicit halos at every z_s)
   and **continuity with Vaskonen's convention**. The `kappathr_flat` kwarg / flat rule
   still exists (`> 0` selects a z_s-independent threshold); `custom_kappathr` still
   overrides both. Rebuilt + `test_backward_compat_bitwise` and all
   `tests/test_cosmology_params.py` (11) pass; default path is now bit-identical to the
   pre-6d reference again.
   **The evidence behind the decision** (`scripts/figures/compare_kappathr_pdf_grid.py`,
   `kappathr_convergence_decision.py`; JSD = Jensen–Shannon div of P(lnμ) vs a
   κ_thr→0 "truth" run): κ_thr is a *numerical* split (explicit halo vs Gaussian bg),
   truth = κ_thr→0. ⟨N⟩ ∝ 1/κ_thr diverges (no plateau; measured slope 1.05 at z_s=1)
   while the PDF (JSD) plateaus below κ_thr≈1e-4 — so cost runs away for zero accuracy
   gain past the knee. Convergence (JSD→truth): flat 1e-4 is z_s-uniform (3.3e-4 at z=1,
   3.4e-4 at z=10); fixed-⟨N⟩ is tied at z=1 (3.5e-4) but drifts to 1.2e-3 at z=10 (its
   κ_thr inflates 1.28e-4→1.37e-3), i.e. coarsest where the signal is strongest; flat
   1e-3 = 2.5e-3 at z=1 (only ⟨N⟩≈11 explicit) but 8.6e-4 at z=10. **Known accepted cost
   of this choice:** fixed-⟨N⟩ carries a z-correlated high-z error (~1.2e-3 JSD at z=10,
   ~3.5× flat 1e-4) — still well below the emulator's own KL 7.3e-3 and 0.17% of the ln2 JSD max, so
   immaterial at O(1000) events; if scaling to ≳4000 events, enforce ⟨1/μ⟩=1 (already
   flagged). Trade note for later: on a z-spanning dataset flat 1e-3 is ~2× cheaper total
   than fixed-⟨N⟩ (0.4/11/146 halos at z=0.2/1/10 vs a flat 100) and better-converged at
   high z — revisit if compute becomes the bottleneck. Plots: `plots/kappathr_pdf_grid_1em04_{linlin,loglin,loglog}.png`,
   `plots/kappathr_decision.png`.
   Gotcha (unchanged): BIAS layer is κ_thr-coupled (σ_b uses tube radius rmax(κ_thr)) —
   full-model σ_κ blows up for κ_thr ≳ 3e-3. Sub-threshold FILAMENTS have no weak
   compensation (−2% at z_s=1, pre-existing). Raw σ_κ differences are rare-tail seed noise
   (±20% per 5e4 draws — measure σ on ensembles!). **ML retraining note: training data
   generated between 2026-07-09 and 2026-07-10 used the flat-1e-3 default — regenerate or
   flag it; the default is fixed-⟨N⟩ again now.** ml/params context unchanged.
   **Subhalo-ON JSD re-test (2026-07-10, `scripts/subhalo_gate/kappathr_subhalo_jsd.py`
   → `data/results/kappathr_subhalo_jsd/report.md`, `plots/kappathr_subhalo_jsd.png`):**
   the halo-only decision re-checked with subhalo_model=3 (factor 1e-5), truth = flat
   3e-5 split into two independent seed halves (proper finite-sample JSD floor — the
   earlier halo-only z=1 JSDs of 3.3–3.5e-4 were AT the 80k/120-bin floor (B−1)/4N, so
   z=1 was "converged", not "3.5e-4"). Result: conclusion UNCHANGED at the JSD level —
   floor-subtracted excess, subhalo on: z=1 fixed-⟨N⟩ 0 / flat 1e-4 2e-5 / flat 1e-3
   1.9e-3; z=10 fixed-⟨N⟩ 7.6e-4 / flat 1e-4 4e-5 / flat 1e-3 3.5e-4 (subhalo-off arm,
   same protocol: 1.1e-3 / 5e-5 / 7.1e-4 ⇒ subhalos do NOT widen the global gap).
   BUT the z=10 high-μ tail penalty becomes significant with subhalos on: fixed-⟨N⟩
   q99.9 = 1.41× truth (95% CI [1.09,1.75]) vs flat 1e-4 at 1.04 [0.80,1.28]; q99.99
   unmeasurable at N=1.2e5. If the tail matters at z_s≳5 (strong-lensing-ish events),
   prefer flat 1e-4 there or hybrid max(κ_N=100, 1e-4); q99/⟨1/μ⟩/σ shifts stay ≲1%/
   0.5%/4%. Cost note: model-3 truth runs ≈1.45e3 s/1e4 at z=10 (κ_thr=3e-5).
   Subhalo (`subhalo_model=3`) re-validated vs brute under the fixed-⟨N⟩ default
   (the clump anchor κ_thr,clump = subhalo_factor·κ_thr,host is z_s-dependent again):
   paired Var(κ)−Var(κ_nosub), 4 seeds, N=1e4, factor 1e-5 → z_s=1 Δ=−1.7%±8.5%
   (the stress case, κ_thr,host=1.28e-4 ≪ old flat 1e-3), z_s=5 Δ=−3.2%±2.8%; both
   within the ±5% acceptance band, so model 3 still tracks brute. Runner now
   parametrized: `scripts/subhalo_gate/subhalo_factor_brute_multiseed.py
   --subhalo-model 3 --kappathr-flat -1` (brute arm forced to model 1; model-3+brute
   throws). Its default output now encodes both knobs
   (`…_m{model}_kthr{legacy|value}.npz`, 2026-07-10 code review): the old ambiguous
   `…_z1_N10000.npz` was split into `…_m1_kthrlegacy.npz` (model-1 calibration,
   restored from git 4b95454) and `…_m3_kthrlegacy.npz` (the model-3 rerun).
10. **Wsub unresolved-subhalo term — derivation + C++ implementation DONE
   (2026-07-09, `subhalo_model=3`, new default):** per-host split κ_halo = reduced host (FULL f_s,b) + resolved clumps
   + μ_unres(y) + N(0, σ²_unres(y)) is EXACT in mean/variance at ANY subhalo_factor
   (Poisson restriction theorem; verified to 6 digits, factors 1e-5…1). Key finding:
   the zero-mean Gaussian ALONE recovers almost nothing (0.844→0.851 at factor 1e-3) —
   the deficit is dominated by the mean-profile mismatch, so the deterministic
   μ_unres(y) table is REQUIRED and the host reduction must switch f_s,res(y) → f_s,b.
   subhalo_factor then becomes a pure performance/Gaussianity knob (dropped-c₃ share
   0.2%/0.7%/3% at 1e-3/1e-2/1e-1); m_floor absorbable too (sub-1e7 variance 0.4%).
   IMPLEMENTED in C++: `Subhalo::buildWsubBin`/`wsubTerm` (mu/sigma tables per bin,
   48-pt log-y grid, ~2 s precompute) + `add_host` model-3 branch (host reduced by
   full `fsb`, `muW(y)+sW(y)*N(0,1)` added after addClumps); `subhalo_brute`
   rejected for model 3. Defaults flipped to model 3 everywhere (LensingConfig,
   SamplingParams, all py::args). Validated at z_s=1 (clipped-core excess, multi-
   seed): model 3 flat at brute (±5% estimator noise) over factor 1e-5…1 while
   model 1 collapses to −0.17 at factor 1; mean bookkeeping exact; 8–16 s vs brute
   158 s per 3e4 realizations. Raw Var estimates are tail-noise-dominated (one κ~9
   realization = 4e-4 shift) — compare clipped cores/ensembles only.
   **PDF-level acceptance DONE (2026-07-12, `scripts/convergence/subhalo_factor_jsd.py`
   → `data/results/subhalo_factor_jsd/report.md`, `plots/subhalo_factor_jsd.png`):**
   brute truth (model-1 arm, 2×120k seed halves), model-3 candidates factor
   1e-5/1e-3/1e-2/1e-1 + model-1@1e-2 contrast, 240k each, z_s∈{1,5}, fixed-⟨N⟩
   rule, JSD protocol identical to the κ_thr/Mmin/Nz studies (seed namespace 6e8).
   **factor=1e-2 PASSES**: JSD excess ≤3e-5 at both z (at the floor 1.1e-4;
   indistinguishable from factor 1e-5), q99.9/q99.99 CIs overlap truth; even 1e-1
   passes globally (mild σ/⟨1/μ⟩ drift at z=5). The no-Wsub contrast (model 1 @
   1e-2) shows real excess 1.7e-4 at z=5 ⇒ the test resolves the Wsub term's
   PDF-level contribution. **Cost: factor 1e-2 ≈ 10× faster than 1e-5** (fixed-⟨N⟩,
   s/1e4: z=1 45.9→4.7, z=5 37.6→5.9 vs subhalo-off 3.6) — near subhalo-off cost.
   Known factor-INDEPENDENT residual: ALL split arms (incl. 1e-5 and model 1) sit
   ~2.3% below brute q99(μ) at z=5 (CIs disjoint; both truth halves agree) — a
   split-vs-brute property, not a factor effect; global JSD unaffected.
   **DEFAULT FLIPPED to subhalo_factor=1e-2 (2026-07-12, user decision):**
   `lensing.h::LensingConfig`, `lnmu_wrapper.h::SamplingParams`, all 5 py::args,
   `get_simulator_config` dict, `main_subhalo_profile.cpp` CLI. Rebuilt;
   test_cosmology_params 11/11 incl. bitwise (default path has subhalo OFF —
   unaffected); verified live (config dict reports 0.01; default ≡ explicit 1e-2
   seed-for-seed, ≠ 1e-5). Subhalo-ON runs made before 2026-07-12 used 1e-5 —
   pass `subhalo_factor=1e-5` explicitly to reproduce them. ML training data
   regeneration still pending. NOTE (unrelated, pre-existing): `tests/
   test_phase3b_local_density.py` fails because `write_local_density_outputs`
   writes `docs/…` while the test asserts `docs/phase3/…` (stale docs reorg). See
   `docs/subhalo/wsub_gaussian_term_derivation.md` §7,
   `playground/analytic/wsub_partition_proof.py`. Post-reorg stale imports
   (`scripts.subhalo_factor_proxy_check` → `scripts.subhalo_gate.…`) fixed only in
   `playground/analytic/subhalo_factor_analytic_deficit.py` +
   `playground/dgate/subhalo_factor_dgate_area_scan.py`; other playground subdirs still stale.
11. **Cost + parallelism of subhalo_model=3 (benchmarked 2026-07-10, flat κ_thr=1e-4,
   10-core M-series):** model 3 is the dominant MC cost — s/1e4 realizations (subhalo off →
   model 3): z_s=0.2 2.5→26 (10×), z_s=1 3.6→92 (26×), z_s=10 7.7→502 (66×). The ratio grows
   with z_s because flat 1e-4 grows the explicit-halo count with z_s and each host then needs
   the Wsub term. **`subhalo_threads`/`subhalo_parallel_threshold` do NOT give realization-level
   speedup** — they parallelize the clump loop WITHIN one host (`subhalo.cpp:458`, fires only
   when a single host's `Nc ≥ parallel_threshold`, default 2e5). At production-scale
   subhalo_factor (1e-5 then; even fewer clumps at the 1e-2 default since 2026-07-12,
   which cuts model-3 cost to ~1.3–1.6× subhalo-off — see #10) hosts have few
   clumps, so it buys ≤1.09× and forcing it on
   (`thr=1`) makes model 3 2–3× SLOWER (thread-spawn per host encounter). It only helps at
   tiny `subhalo_factor` (huge Nc, non-production): factor=1e-8 → 2.76×. It also reseeds RNG
   substreams (`subhalo.cpp:463`) → NOT bitwise-reproducible when it fires. **Correct lever =
   process-level parallelism over seeds** (each realization independent, deterministic per
   shard): ~**4.4× at 8 processes** for model 3 (z_s=1: 108→645 real/s). Sub-linear (memory-
   bandwidth-bound, heavy NFW/invRad/Wsub table lookups), and **8 procs beats 10** on a 10-core
   box (oversubscription). Small batches under-amortize the ~2 s/proc precompute (8k real →
   only 2.8×). Bench scripts: `tmp/bench_kappathr.py`, `tmp/bench_threads.py`, `tmp/bench_mp.py`.

12. **Mmin/Nz grid-convergence studies (2026-07-12) — both defaults KEPT:**
   protocol = κ_thr-study JSD acceptance (two-seed-half truth floor), 240k/config,
   z_s∈{0.2,1,5,10}, driver `scripts/convergence/convergence_scan.py` (axes mmin/
   mmin_pd/nz, shard-cached; uses PLAIN SUBPROCESS shards — mp.Pool turns worker
   segfaults into silent infinite hangs). **Nz=100 converged** (excess ≤1.2e-4 =
   floor; Nz=25 costs 3.9e-3 at z_s=0.2 — low-z is the stress case; quadrature arm:
   κ_thr/σ_W each ~2-4% low at Nz=100 but the shift cancels in P(lnμ)). **Mmin=1e7
   NOT fully converged but immaterial**: excess 0.9e-4–1.0e-3 (worst z=1), knee at
   1e5–1e6, missing 3–8% of σ²_W — 7–70× under the emulator KL 7.3e-3; subhalos do
   NOT amplify it (model-3 spot-check at floor). **NM confound:** M-grid resolution
   alone contributes few-e-4 JSD at z_s≥5 (NM=200 control + per-decade-matched axis);
   treat (Mmin, NM) as a pair. **Far-tail pattern:** z=10 q99.9 tracks the κ_thr_eff
   split point across ALL knob studies (~25-30% low at the default vs refined truths)
   — not Mmin/Nz physics; flat-1e-4 recommendation for z≳5 tails unchanged.
   **BUG FOUND+FIXED: grid-edge OOB in `interpolateNFWMass` (subhalo.cpp:31)** —
   Wsub ψ-grid top point a few ULP under Mlist[NM-1] slipped past the edge guard →
   jm=NM → one-past-end NFWlist read → null deref (presented as "model 3 +
   Mmin<m_floor segfaults", but it's float-rounding luck, lldb-confirmed). Fixed by
   clamping jm ≤ NM−1; default path bit-identical (11/11 tests incl. bitwise after
   rebuild). Also: mp.Pool turns such worker segfaults into SILENT INFINITE HANGS
   (task lost, idle respawn) — prefer subprocess shards for MC drivers.
   ACE tie-in (support-clipped moments; raw moments are single-ray garbage at low
   Mmin): ACE has NO halo mass floor (N-body particles projected, m_p≈1.8e9 M⊙,
   Türker+2025 Table 1 verified) yet Vaskonen ⟨κ²⟩ sits 36–42% ABOVE ACE at z=1
   for ALL Mmin 1e4–1e9 — wrong sign for a missing-low-mass explanation; Mmin
   CANNOT close the ACE gap (only z=5 ⟨κ³⟩ partially closes, −46%→−27%).
   Full findings: `docs/convergence_mmin_nz_note.md`; one-pager
   `docs/convergence_onepager.md`; figures `plots/{mmin,nz}_convergence.png`,
   `plots/ace_gap_vs_mmin.png`.
   **⚠ BATCH-MEAN κ COMPENSATION BUG FOUND (2026-07-13, fix decision
   pending):** a permutation-null calibration of the JSD floors
   (`scripts/convergence/floor_permutation_null.py`) flagged 9 whole shards
   (body-shifted −5.9…−63σ); Mac verification
   (`verify_flagged_shards.py`) showed 7/9 REPRODUCE deterministically —
   root cause: `cpp/lensing.cpp::sample_lnmu` (646–663) anchors ⟨κ⟩=0 with
   the EMPIRICAL batch mean, so one κ≫1 monster ray shifts the whole batch
   by −2κ/n (verified: predicted-vs-observed shift matches ~10% in all 9;
   z5 truthB s4 had a κ≈1014 ray ⇒ −0.135). NOT stale cache, NOT corruption:
   realizations within one sampler call are weakly coupled O(κ_max/n) — this
   affects every ensemble ever drawn via `sample_lnmu`, materially only when
   a batch catches a monster (refined-grid truths at 15–30k/batch; also the
   old "mmin_pd sparse-tail floor" caveat and the ⟨1/μ⟩=5.3 truth-half
   anomaly are THIS). Fix options: analytic mean; robust mean excluding κ>1
   (recommended); larger batches. Do NOT quote the old mmin_pd
   floors/excesses; the shard-excluded corrected numbers (worst Mmin excess
   ~4.1e-4 pd z=1; NM z=10 confound 8.2e-4) are provisional until the
   compensation is fixed + affected configs regenerated. Full analysis chain:
   `data/results/floor_permutation_null/report.md`.
   **Block-aware recalibration + exposure (same day,
   `scripts/convergence/shard_screen.py` → `data/results/shard_screen/
   report.md`, `plots/batch_anchor_diagonal.png`):** under batch coupling the
   exchangeable unit is the SHARD — sample-level permutation p-values were
   anti-conservative (exact C(16,8) block enumeration puts the original truth
   splits at pctl 4–90 = typical; the earlier "pctl 98–100" was the wrong
   test). Exposure screen of all 5 study caches: **κ_thr + subhalo_factor
   decision studies CLEAN** (no anchor shards, block ≈ sample floors — both
   default decisions stand); nz essentially clean; mmin plain mostly clean
   EXCEPT z10 nm200ctl (−0.035 shift ⇒ the z≥5 NM-confound number is the most
   exposed published value) and mild hits at z5 m1e7/m1e5; mmin_pd z1/z5 have
   LOW RESOLUTION (block floors 8.3e-4/2.4e-3) — order-of-magnitude only.
   Regen list post-fix (~10 shards, cheap): mmin_pd truths, mmin z10
   nm200ctl + z5 m1e5/m1e7, nz z0.2 truthB + z1 nz50, mmin z1_sub truthB.
   Future floors: use shard-level (block) resampling; record κ_max per shard.

13. **BIAS layer couples a physical scale to the numerical grid — strong tail
   is grid-defined, fix pending (2026-07-13/14):** Var(κ) convergence in Nz is
   SUPPORT-DEPENDENT — |κ|<1 body converges (O(1/Nz), matches the analytic
   count quadrature 0.967@Nz=100), but f(κ>1) grows unsaturated with Nz
   (z=10: 0.45%→1.30% over Nz 25→1600; q99.99 2.05→8.8; raw-κ sweeps
   `scripts/convergence/vark_vs_nz{,_frozen,_nobias}.py`, namespaces
   8.0/8.1/8.2e8). Mechanism CONVICTED by three arms (frozen κ_thr: growth
   unchanged; analytic counts: converge; bias=0: growth VANISHES): the bias
   modulation amplitude σ_b = σ(M_b) uses the tube-SEGMENT mass with segment
   length = shell width and radius = rmax(κ_thr) (`lensing.cpp:123`), drawn
   INDEPENDENTLY per (jz,jM) cell — so Nz (and κ_thr via rmax: the known
   "σ_b blows up ≳3e-3" gotcha is the SAME bug transversely; and NM via
   per-jM independence) silently change the clustering model; no continuum
   limit. The reference model's own validity condition ("Δz chosen so M_b ≫
   typical lens masses") is exited by refinement. Do NOT quote bias-tail
   quantities (f(κ>1), q≳99.9 at z_s≳5, wide-support ⟨κ²⟩) as converged; do
   NOT "fix" by raising Nz (makes it worse). P(lnμ)/JSD verdicts and the
   emulator are UNAFFECTED (body-weighted). Recommended fix (supervisor
   decision pending): explicit peak-background-split field — one δ_env(χ)
   low-pass filtered at fixed comoving R_bg (~heaviest-halo Lagrangian
   radius), realized on a fixed coarse grid, bias-scaled per (M,z); behind a
   `bias_model` flag (legacy 0 default). Prediction/validation gate: also
   kills the κ_thr≳3e-3 blow-up. **M_b/M validity map DONE (2026-07-16,
   `scripts/convergence/mb_validity_map.py` → `data/results/mb_validity/report.md`,
   `plots/mb_validity_map.png`; no C++ change — reuses the validated Python port):**
   the paper's "M_b ≫ typical lens masses" claim FAILS at high z_s even at the
   default grid — bias-variance-weighted median M_b/M = 14/8.3/3.1/2.2 at
   z_s=0.2/1/5/10 (fixed-⟨N⟩ rule; ~zero weight above M_b/M=100 anywhere), and
   INVERTS under refinement (Nz=400: 83–100% of the weight has M_b<M at z_s≥5).
   The physical-vs-comoving rmax² wart in Mb (×(1+z)² if fixed) softens z_s=1
   but not the high-z_s verdict. Full note:
   `docs/nz_bias_convergence_note.md`; evidence chain
   `data/results/vark_nz/mechanism_note.md`; figures
   `plots/nz_tail_mechanism.png`, `plots/vark_total_vs_nz.png`.
   **Python prototype of the fix DONE (2026-07-14,
   `scripts/convergence/bias_field_prototype.py` →
   `data/results/bias_field_prototype/report.md`, `plots/bias_field_prototype.png`,
   seed namespace 9.0e8):** design refined to have NO free R_bg — environment =
   1D pencil-projected linear-P(k) field along the LOS; per-cell amplitude = σ of
   the field in the cell's own counting cylinder (R=(1+z)·rmax comoving, L=Δχ)
   with a parameter-free peak-background-split floor at the halo Lagrangian
   radius R_L(M) (separable window; this is what kills the transverse blow-up
   path); cross-cell corr from P_1D; same mean-1 lognormal λ. The Python port of
   the C++ tables validates EXACTLY vs the gwlensing helpers (κ_thr/⟨N⟩/σ_W to
   ≤5e-9). Results (bias+halo-layer MC, 2e4 real, 1 seed/config): z=10 f(κ>1)
   old arm 2.5→4.3e-3 over Nz 25→800 (reproduces the measured full-model growth
   shape) vs new 1.5→2.9e-3 much flatter (residual slope ~1.5σ; per-cell σ_new
   plateaus only once Δχ≲R_L i.e. Nz≳1000 — needs seeds to call it converged);
   κ_thr sweep: old w-p99 σ_b grows 3.3→5.1 over 1e-4→1e-2 vs new SATURATING
   1.6→1.8 (blow-up killed); analytic linearized clustering std flat in Nz for
   new (0.029–0.030 at z=10, Nz 25–400) vs falling for old (0.028→0.011 —
   correlated body variance leaking into ever-wilder iid lognormal tails).
   **Honest cost — NOT body-preserving:** at the default grid the new model's
   clustering variance is BIGGER (z=1 std 0.0084 vs 0.0024 ⇒ ~6% vs 0.5% of
   total Var(κ)) because cross-shell correlations — cancelled by construction in
   the old iid layer — dominate, even though per-cell σ (w-mean) is 2–4× SMALLER
   (cylinder ≪ sphere-of-M_b). Also found: the old layer's lognormal moments
   E[λ²] DIVERGE (cells with σ_b~30–150 at barN~1e-20), so analytic comparisons
   must be linearized (or MC), and any unweighted max/moment statistic is junk —
   use barN·κ̄-weighted quantiles.
   **Design converged (2026-07-15, whiteboard session + lit study; full note:
   `docs/bias_field_design_note.md`):** one correlated Gaussian field δ_1D(χ) per
   realization from a mode sum, σ_n² = (L/2)·P_1D(k_n) per Re/Im component (KP91
   pencil projection of P_lin with smooth transverse window; = Agrawal+17
   lognormal-mock recipe; pad L ≳ 2χ(z_max) against periodicity); all (M,z) cells
   ride it via b(M,z) (which carries D(z)); modulation exp(bδ−½b²σ²) (model 1,
   supervisor's write-up) or the conditional first-crossing ratio
   T·pFC(δ_c(z)−δ₀, S(M)−S_env)/pFC(δ_c(z),S(M))/C (model 2 candidate — positive,
   mass-budget-saturating, nonlinear b_n self-consistent, enforces R_⊥>R_L(M)
   structurally). Window floor R_⊥ ≥ R_L(M) = halo Lagrangian radius (PBS/
   separate-universe validity; heavy bins M ≳ few×1e14 force the per-mass floor).
   Plan: `bias_model` flag 0=legacy(bitwise default)/1/2. **CODE FINDING:
   `halobias` uses q=0.75 but `pFC` uses q=0.8 — bias is not the PBS response of
   the code's own HMF (few % at high ν); test = consistency relation
   ∫M n̄ b dM/ρ̄_m = 1; filament bias should derive from pFCfil.** Also settled:
   Poisson-given-field is the CORRECT limit for tube sampling (thinning
   (rmax/R_⊥)² ~ 1e-4 kills mass-conservation anti-correlations; do NOT subtract
   drawn halos from the field). turboGL precedent: its split scale k_L is
   CALIBRATED (exp(3.9−4.6z) Mpc⁻¹, λ_L≈13 Mpc at z=1 ≈ our R_L(M_max)); tails
   need N-body in this whole method class ⇒ "don't quote f(κ>1)" rule survives
   the fix. Validation plan (ξ_hh = b₁b₂ξ_1D acceptance test, consistency
   relation, JSD(old,new), R_⊥ scan) in the note §7.
   **C++ WIRING DONE (2026-07-16, ⚠ Mac `make build` + bitwise gate PENDING):**
   `bias_model` (0=legacy iid default / 1=correlated 1D field) + `bias_Rperp`
   (comoving kpc) plumbed through `LensingConfig`, `SamplingParams`, all 5
   py::args, `get_simulator_config`; `cosmology.h` gained public `Pk0(k)`
   (growth-free P(k), sigmalist normalization). `lensing.cpp::BiasField1D`:
   KP91 disk-window P_1D table → EXACT mode-sum shell covariance (L=1.05χ(z_s),
   N_max=L/R_⊥ floored at 4, capped 5e6; trig-recursion, resync every 4096
   modes; log-k quadrature was tried and REJECTED — can't resolve the ~1e4 sinc
   oscillations, 2.5% σ error) → Cholesky; per-realization per-shell field
   drawn up front (float, Nreal×n_shells — 158 MB at Nreal=400k, keep shards
   small), λ=exp(bDgδ̄−½(bDg)²σ̄²) replaces the iid draw; same λ multiplies
   filaments. Model-1 RNG consumption sits at ONE fixed point; model-0 stream
   untouched (bitwise gate must confirm after build). Pre-build validation
   PASSED: `playground/bias_field/validate_field_covariance.py` (numpy verbatim
   replica vs independent notebook-grade reference: |ΔCov| ~1e-6 over
   R_⊥=8.4kpc–840Mpc at z_s=1,5; Cholesky 1e-13; ⟨λ⟩=1). After Mac build run:
   `$PY playground/bias_field/validate_field_covariance.py --module` (smoke) +
   `make pytest` (incl. bitwise). Mac gates ALL PASSED 2026-07-16 (build,
   11/11 incl. bitwise, layer-1 |ΔCov|~3e-6, module smoke). Post-review fix:
   `bias=False` + `bias_model=1` now skips the field build/draws entirely
   (was dead weight; nobias arms across bias_model values are NOT
   stream-identical by construction). Known deviation kept for now: L=1.05χ
   pad vs design note's ≳2χ — measured wrap leakage ≤1.6% on last-shell σ,
   end-to-end corr −1.4e-2 vs −4.9e-3 at pad 3 (revisit later).
   **R_⊥→PDF scan DONE (2026-07-16, `scripts/convergence/rperp_pdf_scan.py
   mc|report`, z_s={1,5}, 14 masses 1e7–1e20 M⊙ via R_⊥=R_L(M) = 39 kpc–844
   Mpc, anchors legacy/nobias, 240k/arm, kappa_anchor=1, subprocess shards,
   seed namespace 9.5e8 → `data/results/rperp_pdf_scan/report.md`,
   `plots/rperp_pdf_scan.png`, PDF-only detail `plots/rperp_pdf_only.png`
   via the `pdfplot` stage):** PDF response is MONOTONE in R_⊥ — smaller
   window ⇒ more power in δ_1D ⇒ wider body + heavier tail; saturates to the
   nobias PDF above R_⊥ ≈ 20–40 Mpc (JSD vs nobias at floor). NO single
   global R_⊥ reproduces legacy at both z_s: legacy ≈ field @ R_⊥≈8 Mpc
   (M=1e14) at z_s=1 (JSD 1.4e-4 = floor) but ≈ field @ R_⊥≲0.2–0.8 Mpc
   (M=1e9–1e11) at z_s=5 — the sweep directly exhibits the legacy layer's
   z-drifting effective scale (same pathology as the M_b/M validity map).
   At the PBS-motivated R_⊥=R_L(M_max)≈18 Mpc (≈turboGL λ_L) the one-field
   clustering widens σ(lnμ) only +2.5%/+1.3% (z=1/5) over nobias vs legacy's
   +10%/+9% ⇒ a global heavy-mass window kills most clustering variance for
   light cells; per-mass R_⊥ (model 2 / R_L(M) floor) is where the real
   answer lies. Dynamic range across the scan: σ(lnμ) up to +62%/+18%
   (z=1/5) over nobias at R_⊥=39–84 kpc; q99.9(μ) 2.9/10.7 vs nobias
   1.75/6.0. Gotcha: raw ⟨1/μ⟩ in small-R_⊥ arms is single-monster-ray junk
   (one lnμ=−20 demagnified ray ⇒ ⟨1/μ⟩=2377 at z=1/m1e9; top-ray-removed
   =1.06) — known κ≫1 artifact, use trimmed means only.
   **DEFAULT bias_Rperp = 8441 kpc = R_L(1e14 M⊙) (2026-07-16, user decision
   from the scan + weighted-scale calc; supervisor sign-off pending):** the
   clustering-variance-weighted (w=(barN·κ̄·b̃)², cell_tables port) R_L
   quantiles are median/90%/99% = 8.3/14/19 Mpc at z_s=0.2, 6.1/11/17 at
   z_s=1, 3.3/7.6/12 at z_s=5, 2.8/7.1/12 at z_s=10 — so 8.4 Mpc = the
   low-z signal scale (where the clustering share of Var(κ) is largest,
   12–25% at z_s≤1, and where the GW events are), PBS-valid at the median
   signal cell; turboGL λ_L≈13 Mpc corroborates at the z≈1 pivot ONLY
   (k_L=exp(3.9−4.6z)/Mpc is z-dep. and its κ_L plays a different role —
   additive split, not an environment window; supports the scale, not the
   fixedness). **Legacy-match is NOT a criterion (user
   ruling 2026-07-16: legacy passing its own consistency test ≠ correct;
   ACE ⟨κ²⟩ gap is independent evidence against it)** — the exact z_s=1
   agreement with legacy in the scan is an after-the-fact cross-check only.
   Honest caveat: on purely internal grounds anything in ~4–12 Mpc is
   defensible (weighted median drifts 8.3→2.8 Mpc over z_s 0.2→10), which
   strengthens the case for the per-mass R_L(M) extension. Known accepted
   costs: PBS marginal for the heaviest ~10% of weight at low z_s; z_s=5
   departs from legacy (σ 0.237 vs 0.249 — legacy invalid there per the
   M_b/M map). Fixed comoving
   number, NOT recomputed per cosmology. Rejected: 18 Mpc strict-PBS
   (clustering nearly inert, +2.5%/+1.3% σ) ; per-mass R_L(M) windows =
   model-2/follow-up (needs cross-mass P_1D^{ab} covariance + joint
   shell×mass Cholesky in C++). Updated in lensing.h, lnmu_wrapper.h, all 5
   py::args + config dict, validate smoke; rebuilt, 11/11 incl. bitwise,
   default ≡ explicit 8441 seed-for-seed. Runs made 2026-07-16 BEFORE the
   flip (incl. the rperp scan arms) used explicit bias_Rperp values —
   unaffected; anything that relied on the 3000 kpc placeholder default must
   pass bias_Rperp=3000 explicitly to reproduce.
   **Clustering-share systematics (2026-07-16, `tmp/rperp_zoom.py` +
   `tmp/rperp_z_and_N.py`, arms cached in the scan mc dir at 240k–960k;
   figs `plots/rperp_zoom_8to18.png`, `plots/rperp_zoom_z05_z10.png`,
   `plots/clustering_vs_N.png`):** within the candidate window band the PDF
   is soft — 8.44 vs 10 Mpc unresolvable at 960k; 8.44→18.2 Mpc = −5% σ at
   z_s=1 (coherent ±3–5% peak/shoulder tilt, ~1/3 of the nobias distance).
   Clipped-core clustering share of Var(lnμ) at R_⊥=8.44 Mpc:
   13.0/9.4/4.8/4.2% at z_s=0.5/1/5/10 (18.2 Mpc: 4.2/4.1/2.2/1.7%) —
   clustering matters most at LOW z_s and the window choice matters most
   there too. **⟨N⟩ mechanism CONFIRMED:** share grows monotonically with
   the explicit-halo count, 9.4→11.3→14.6% (±0.5, clipped, shard-jackknife)
   at ⟨N⟩=100/300/1000, z_s=1 — because ONLY explicit counts are
   field-modulated (κ_W Gaussian is not), so raising ⟨N⟩ converts
   unmodulated background into modulated Poisson. RAW-variance shares are
   tail-noise junk (26±4% at N=300 — non-monotonic artifact; the old "clip
   or jackknife" rule applies to lnμ too). Production implication: do NOT
   raise ⟨N⟩ (cost ∝ N for +2.6 share points per 3×); the efficient
   completion is the joint-framework weak arm (same field modulates the
   clustered part of σ_W + count–weak covariance, 2S_ew≈2S_ww, sized
   12–25% of Var at z_s≤1) at unchanged ⟨N⟩=100 cost.
   **Weak-arm PDF-level MC DONE (2026-07-16,
   `scripts/convergence/bias_field_weak_arm.py` →
   `data/results/bias_field_weak_arm/report.md`,
   `plots/bias_field_weak_arm.png`, seed namespace 9.7e8):** the clustered
   weak background κ_W,clust = Σ_jz s_w[jz]·v_jz (SAME realized field as the
   counts; amplitudes from the validated joint-sizing bookkeeping, windowed
   + pencil brackets) measured in the prototype MC with PAIRED arms (base =
   shipped counts-only bias_model=1 replica; weak terms are deterministic
   given the field, so same draws), 2×100k, z_s∈{0.2,1,5}: windowed arm adds
   **+11/+23/+52% to clipped Var(κ)** (matches analytic S_ww+2S_ew to ~1%
   at z_s≤1 — MC⇄sizing cross-validated), **JSD vs base 1.6–2.0e-2 nats**
   (≈2–3× the emulator's whole KL budget 7.3e-3; floor 2.8e-4; pencil
   bracket 4.6–9.3e-2). Structure: the effect is dominantly the LOW-κ flank
   (underdense-LOS demagnification spread — counts-only saturates at zero
   halos and can't express it); the high tail barely moves. Legacy iid ≈
   counts-only field at z_s=1 (JSD 1.7e-3) ⇒ both current models miss the
   term about equally. **Verdict: material at PDF level — the C++ weak arm
   is warranted (supervisor decision pending).** Implementation flags:
   (a) prototype term is LINEAR in δ̄ — at z_s=0.2 the pencil arm pushes κ
   below the empty-beam floor; the C++ version should modulate the per-shell
   weak MEAN RATE with the same mean-1 lognormal λ form as counts (positive,
   respects empty beam); (b) amplitudes inherit the untruncated-NFW ≲×1.5–2
   inflation (framework flag 1); (c) ACE tension: adds body variance on top
   of the +36–42% ⟨κ²⟩ overshoot (flag 2); (d) it moves the low-μ edge, so
   the emulator's edge/flux calibration must be refit after retraining.

## Conventions
- Plots → `plots/`; throwaway/scratch → `tmp/`.
- Substructure reference papers are in `papers/misc/` (JvdB14, vdB05, Han+16, BMO09, SatGen,
  CUSP). Lift fitting-function numbers from the PDFs, not from memory.

## Multi-CLI delegation (offload token-heavy work; Claude orchestrates)
Claude keeps judgment, physics, edits, and **all git/gh/push**. Delegates never commit.
Available delegates: codex (large implementations/debugging), agy (bulk token-heavy
reads, mechanical cross-checks), gh copilot (shell one-liners). ALWAYS invoke the
`delegate` skill (`.claude/skills/delegate/SKILL.md`) before delegating — it has the
verified command lines (incl. hang gotchas) and the standing rules.
