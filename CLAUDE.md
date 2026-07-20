# CLAUDE.md — gw-wl-emulator

Fast ML emulator for the GW weak-lensing magnification PDF, built on the C++ Monte-Carlo
engine from Vaskonen (2026). See `README.MD` for the project overview and `memory/MEMORY.md`
(auto-loaded) for cross-session context.

**Paper (PRD draft, .tex on Overleaf — not in repo):** before touching paper text,
figures, or answering draft comments, read `paper_prod/paper_memo.md` (2026-07-20) —
maps every draft section/equation to its implementation (file:line), figure scripts,
and the claim-constraining standing rules; `paper_prod/draft_comments_memo.md` holds
the resolved \R{}/\Gala{} comments + known draft↔code mismatches (R_s/window, model-3
description, stale figure paths). Keep both updated like this file.

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
`sample_lensing_raw_ml` (returns raw `kappa`, `gamma`; since 2026-07-16 also
`kappa_weak` = background-arm component per ray, so `kappa - kappa_weak` = explicit
part — additive change, bitwise-clean), `compute_lnmu_stats`,
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
9. **Threshold rule — DEFAULT is legacy fixed-⟨N⟩=100 (reverted 2026-07-10, user
   decision: predictability + Vaskonen continuity):** `kappathr_flat = -1` everywhere;
   `kappathr_flat > 0` selects a flat z_s-independent threshold; `custom_kappathr`
   overrides both. Evidence (JSD vs κ_thr→0 truth): PDF plateaus below κ_thr≈1e-4 while
   ⟨N⟩∝1/κ_thr diverges; fixed-⟨N⟩ carries an accepted z-correlated high-z error
   (~1.2e-3 JSD at z=10, immaterial at O(1000) events). Subhalo-ON retest: global JSD
   verdict unchanged, but fixed-⟨N⟩ z=10 q99.9 = 1.41× truth [1.09,1.75] — if tails
   matter at z_s≳5, use flat 1e-4 or hybrid max(κ_N=100, 1e-4). Live gotchas: BIAS σ_b
   blows up for κ_thr≳3e-3 (see item 13); sub-threshold filaments lack weak compensation
   (−2% at z_s=1); raw σ_κ is rare-tail seed noise — measure on ensembles; **ML data
   generated 2026-07-09→10 used the flat-1e-3 default — regenerate or flag it.**
   Full evidence narrative: `docs/claude_md_archive.md`; `plots/kappathr_decision.png`;
   `data/results/kappathr_subhalo_jsd/report.md`.
10. **Wsub unresolved-subhalo term — `subhalo_model=3` is the default; `subhalo_factor`
   default = 1e-2 (flipped 2026-07-12, user decision; passes PDF-level brute acceptance,
   ~10× cheaper than 1e-5, near subhalo-off cost):** per-host split κ_halo = reduced host
   (FULL f_s,b) + resolved clumps + μ_unres(y) + N(0, σ²_unres(y)); exact in mean/variance
   at any factor; `subhalo_brute` + model 3 throws. **Reproduction: subhalo-ON runs made
   before 2026-07-12 used factor 1e-5 — pass it explicitly.** ML training data regen still
   pending. Known factor-INDEPENDENT residual: all split arms sit ~2.3% below brute q99(μ)
   at z=5 (global JSD unaffected). Compare clipped cores/ensembles only — one κ~9 ray =
   4e-4 raw-Var shift. Pre-existing unrelated failure: `tests/test_phase3b_local_density.py`
   (stale docs path). Derivation: `docs/subhalo/wsub_gaussian_term_derivation.md` §7;
   acceptance: `data/results/subhalo_factor_jsd/report.md`; full narrative:
   `docs/claude_md_archive.md`.
11. **Model-3 cost + parallelism (benchmarked 2026-07-10):** at the 1e-2 factor default,
   model 3 costs ~1.3–1.6× subhalo-off (was 26–66× at 1e-5). `subhalo_threads`/
   `subhalo_parallel_threshold` parallelize WITHIN one host only — useless at production
   factors, 2–3× SLOWER if forced on, and NOT bitwise-reproducible when they fire.
   Correct lever: process-level parallelism over seeds — ~4.4× at 8 processes (8 beats 10
   on a 10-core box; ~2 s/proc precompute needs big batches). Benchmark tables:
   `docs/claude_md_archive.md`; `tmp/bench_{kappathr,threads,mp}.py`.
12. **Mmin/Nz grid convergence (2026-07-12) — both defaults KEPT:** Nz=100 converged
   (excess at floor); Mmin=1e7 not fully converged but immaterial (worst excess ~1e-3 ≪
   emulator KL 7.3e-3; knee 1e5–1e6); treat (Mmin, NM) as a pair — NM alone contributes
   few-e-4 JSD at z_s≥5; **Mmin CANNOT close the ACE gap (wrong sign: Vaskonen ⟨κ²⟩ is
   36–42% ABOVE ACE at z=1 for all Mmin 1e4–1e9; ACE has no mass floor).** Fixed:
   `interpolateNFWMass` grid-edge OOB segfault (jm clamp, bitwise-clean). **⚠ Batch-mean
   κ compensation bug (found 2026-07-13, fix decision pending — robust mean excluding κ>1
   recommended):** `cpp/lensing.cpp::sample_lnmu` (646–663) anchors ⟨κ⟩=0 with the
   EMPIRICAL batch mean, so one κ≫1 monster ray shifts the whole batch by −2κ/n. Do NOT
   quote the old mmin_pd floors/excesses (corrected numbers provisional); κ_thr +
   subhalo_factor decision studies verified CLEAN (both defaults stand). Future floors:
   shard-level (block) resampling; record κ_max per shard. mp.Pool hides worker segfaults
   as SILENT HANGS — use subprocess shards for MC drivers. Full findings:
   `docs/convergence_mmin_nz_note.md`, `docs/convergence_onepager.md`,
   `data/results/{floor_permutation_null,shard_screen}/report.md`,
   `docs/claude_md_archive.md`.
13. **Bias field — legacy iid layer couples a physical scale to the numerical grid
   (strong tail is grid-defined); correlated-field replacement SHIPPED (2026-07-16):**
   `bias_model` 0 = legacy iid (bitwise default) / 1 = correlated 1D pencil-projected
   field (`lensing.cpp::BiasField1D`); **default `bias_Rperp` = 8441 kpc = R_L(1e14 M⊙)**
   (user decision 2026-07-16 from the R_⊥→PDF scan + weighted-scale calc; supervisor
   sign-off pending; on internal grounds ~4–12 Mpc is defensible — per-mass R_L(M)
   windows = model-2 follow-up). Mac gates ALL PASSED (build, 11/11 incl. bitwise,
   layer-1 |ΔCov|~3e-6, module smoke); default ≡ explicit 8441 seed-for-seed.
   `bias_weak` (Cox-split weak arm, supervisor-approved formulation) wired 2026-07-16,
   default OFF, requires bias_model=1 (throws otherwise) — **Mac gates + acceptance
   PASSED (2026-07-16, independent Claude re-review):** rebuilt .so, 11/11 incl.
   bitwise; joint determinism, guard throw, bias=0 no-op, default≡explicit-legacy all
   bit-equal; **split invariance JSD(w1e14 ⟨N⟩=100, wN300 ⟨N⟩=300), 240k each, scan
   protocol: z_s=1 3.1e-4 vs floors 2.4/1.9e-4 = AT FLOOR; z_s=5 5.3e-4 vs 2.5/3.1e-4 =
   excess ~2.5e-4** (σ(lnμ) 0.2506→0.2564; expected approximation-class residual — the
   Cox split is exact in mean+variance but the shuffled band swaps Poisson skewness +
   profile shapes for a Gaussian; 30× under emulator KL, not blocking; gate shards now
   cached in the scan mc dir). **Fiducial magnitude at default R_⊥=8.44 Mpc: clipped
   Var(lnμ) +18–20% (z_s=1), +20–24% (z_s=0.5); JSD(joint, counts-only) 4.8e-3/7.7e-3
   (floors ~3e-4)** — consistent with the prototype windowed sizing. ⚠ Gotchas that
   nearly produced wrong verdicts: `sample_lnmu` positionals are (z, OmegaM, sigma8, h)
   — h-first (σ8=0.315) kills field power and the weak arm reads "at floor"; raw-Var
   weak-arm checks are seed junk (20k raw ratio 0.98 — clip or JSD only). History:
   c5d3d5d had buildWeak but NO sampler block (a556fe3 fixed) — discard any joint runs
   from between. Known nits (non-blocking): weakSV NaN if a shell σ→0 (theoretical);
   lnT/lnV interp errS ≤3e-3; joint stream not field-paired with counts-only at same
   seed — if the draw-and-discard pairing patch is wanted, land it BEFORE the full
   joint sweep (invalidates w* shards; 64 gate shards cheap to regen, counts-only
   cache unaffected).
   NFW-only scope: filaments carry no weak arm (call out in design note). The weak arm
   moves the low-μ edge — refit the emulator's edge/flux calibration after retraining.
   **Standing rules:** do NOT quote bias-tail quantities (f(κ>1), q≳99.9 at z_s≳5,
   wide-support ⟨κ²⟩) as converged — legacy has no continuum limit (do NOT "fix" by
   raising Nz; makes it worse) and absolute far tails are uncertified in this whole
   method class (needs N-body); P(lnμ)/JSD verdicts + the emulator are body-weighted and
   UNAFFECTED. Do NOT raise ⟨N⟩ to chase clustering share (cost ∝ N; the weak arm is the
   efficient completion at unchanged ⟨N⟩=100). Raw ⟨1/μ⟩/max/moment statistics are
   monster-ray junk — use trimmed/clipped ensembles and barN·κ̄-weighted quantiles.
   Legacy-match is NOT a correctness criterion (user ruling 2026-07-16; ACE ⟨κ²⟩ gap is
   independent evidence against legacy). Window choice is soft in the BODY only — at
   lnμ>1 the 8.44 vs 18.2 Mpc arms differ ~40% in tail rate (immaterial at O(1000)
   events). **Reproduction:** runs before the 2026-07-16 default flip that relied on the
   3000 kpc placeholder must pass `bias_Rperp=3000` explicitly; the rperp-scan arms used
   explicit values (unaffected). Known kept deviation: L=1.05χ pad vs design note's ≳2χ
   (wrap leakage ≤1.6% — revisit).
   **Variance-partition figure recreated for the bias era (2026-07-16/17, bias_model=1 +
   bias_weak; grid extended to κ_thr=1e-8 + weak shot/corr decomposition 2026-07-17):**
   `paper_prod/scripts/plot_sigma_partition_vs_kthr_bias.py` →
   `sigma_partition_vs_kthr_bias.{png,pdf}` + `data/sigma_partition_vs_kthr_bias_z1.npz`;
   weak/strong/total MEASURED (per-ray split via the new `kappa_weak` field) on the
   shared κ_tot≤1 core mask, 8-seed ensembles (100k rays/pt on the 33-pt main grid,
   adaptive 4–30k on the 6 deep half-decade points — ⟨N⟩∝1/κ_thr, ~0.2 s/ray at 1e-8;
   `sweep_sigma_partition_components.py mode=deep`): σ_tot flat to ~2% ptp over
   κ_min≤κ_thr≤1e3 (1.3% on the main grid alone; median SEM 0.3%), plateau
   σ_κ=0.0313 at z_s=1 (shot-only legacy plateau 0.0273), crossing κ_thr≈0.03;
   additivity Var_w+Var_s+2Cov=Var_tot exact on the shared mask. Weak decomposition:
   shot = analytic floor-consistent σ_W (clip-insensitive), corr = √(Var_w−shot²) —
   corr DOMINATES the weak arm below κ_thr≈0.1 (e.g. 0.0055 vs 0.0004 at κ_thr=1e-5),
   clipped-corr plateau 0.0155 vs law-level (6σ) 0.0304 — the difference is the
   uncertified field tail. **Figures plot the model domain κ_thr≥κ_min=1.28e-7 ONLY
   (user decision 2026-07-17); sub-floor points (to 1e-8) stay in the npz:** below
   the absolute floor conservation is NOT claimed — the weak arm is empty by
   construction (eps_floor>1 ⇒ σ_W=0, bias_weak tables ≡0), strong == total exactly,
   and the total genuinely grows (law-level Var 1.145× plateau at κ_thr=1e-8, clipped
   σ +2.1%) because the explicit arm re-includes the sub-floor band NO arm ever
   covered; the band's clustering mean grows ∝ln(1/κ) (untruncated-NFW artifact — the
   reason the absolute floor exists), and below κ_thr≈3e-8 the `rmaxfNFW` r≤1e6 kpc
   bisection ceiling truncates the largest cells (identically in production).
   Law-level RAW conservation proven analytically by
   `playground/sigma_partition_bias_vs_kthr.cpp` (includes lensing.cpp → production
   BiasField1D + clamped lnT/lnV tables; shot part to 4.4e-9, total incl. clustering +
   weak↔strong cross term to 4.7e-4 for κ_thr≥κ_min, common-mode field draws).
   **Floor-constraint test (2026-07-17): the below-floor rise is PURELY the domain
   edge, not the split** — the full 1e-8..1e3 sweep rerun with the floor lowered to
   κ_min=1e-11 is FLAT everywhere: law-level varT spread 4.0e-4
   (`playground/sigma_partition_bias_vs_kthr_zs1_kmin1e-11.txt`, probe argv[4]
   override), measured clip1 ptp 2.6% incl. small-sample deep points (8 seeds,
   `partition_components_zs1_floortest_seed*.npz`, sweep `mode=floortest`);
   comparison figure `plots/sigma_partition_floor_test.png`. **Production floor
   lever (no code change): `kappathr_flat=K` + `custom_kappathr=κ_thr` ⇒
   κ_min=1e-3·K** — kappathr_flat feeds ONLY the two floor expressions
   (lensing.cpp:692,772) once custom_kappathr drives the split. ⚠ The plateau is
   floor-DEPENDENT even clipped: σ_t 0.0313→0.0379 (law-level 0.0409→0.0604) for
   κ_min 1.28e-7→1e-11 — ln(1/κ) growth of the faint-band clustering mean ⇒ κ_min
   is a MODEL parameter (effective NFW outer truncation / stand-in for the missing
   2-halo completion of sub-floor matter), NOT a numerical convergence knob; do not
   try to "converge" it downward.
   ⟨N⟩-axis companion (2026-07-17): `plot_sigma_partition_vs_N_bias.py` →
   `sigma_partition_vs_N_bias.{png,pdf}` — same npz, x = ⟨N⟩ = NhfNFW(z_s,κ_thr)
   (`playground/compute_nexp_vs_kthr.py`; ⟨N⟩≈0.013/κ_thr in the low-κ_thr regime,
   =100 at fiducial, collapses super-exp. above κ_thr~1 — points ⟨N⟩<1e-4 dropped);
   plot ends at the domain boundary ⟨N⟩(κ_min)≈1.1e5; weak/strong crossing at
   ⟨N⟩≈0.15; at the production default ⟨N⟩=100 the weak arm still carries
   σ_w=0.0093 (~9% of Var, mostly corr term).
   ⚠ FINDING: the FORMAL clustering variance is 6σ-clamp-dominated — Var(κ) doubles
   between 5σ and 6σ field clamps (probe's clamp scan) and the unclamped log-normal
   pair sum exp(a_p·a_q·C_ij) OVERFLOWS (a = b·Dg ~ 40–50 at M≳1e16: e^{a²σ²} beats
   the HMF e^{-qν²/2}) — the model's raw 2-halo variance has no sane top-mass limit;
   body (≤3σ field) = +18% on Var(κ) at z_s=1, matching the clipped acceptance. Raw
   MC σ_tot confirms junk (0.031–0.072 scatter at 100k rays,
   `playground/sigma_total_vs_kthr_bias_zs1.txt`); a q99.9 trim tilts the sweep (4.3%
   ptp) because it bites a κ_thr-dependent tail slice — κ_tot≤1 is the certified
   estimator. Physics note: with the correlation term the weak curve no longer →0 at
   low κ_thr (clustering variance scales with the band MEAN, not its κ² moment):
   σ_w(κ_thr=1e-5)=0.0055, and strong alone ≠ total (cross term bridges — both
   expected, explain in caption). Components sweep:
   `playground/sweep_sigma_partition_components.py` (8 parallel seed processes).
   Full design/evidence:
   `docs/bias_field_design_note.md`, `docs/nz_bias_convergence_note.md`,
   `data/results/vark_nz/mechanism_note.md`, `data/results/{bias_field_prototype,
   rperp_pdf_scan,bias_field_weak_arm,mb_validity,shard_screen}/report.md`; M_b/M map,
   deep-tail and clustering-share narratives: `docs/claude_md_archive.md`.

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
