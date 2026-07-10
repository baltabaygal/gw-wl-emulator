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
asymptote, blend at μ_c=1.7) + empty-beam low-μ cutoff. Do NOT use `flow_ar.pt`.
Full experiment log: `ml/autoresearch/REPORT.md` (exp18–24) + `results.tsv`.

**Validated (REPORT.md 2026-07-03):** median KL 0.0073 (= ACE-Lensing's 0.007, and
holds to μ=100 where they truncate at 6); PIT KS 3–10%; ⟨1/μ⟩=1 within 1%; posterior
recovery on 16×1000-event mocks: model bias h −0.46±0.20 σ_post, Ωm +0.29±0.17.
Fit for purpose to O(1000) events; for ≳4000 events first enforce ⟨1/μ⟩=1 (lnμ shift).

**Retraining recipe (when new/subhalo-corrected training data lands):**
1. regenerate datasets + `cache/lowz_aug.npz` (`gen_lowz_data.py`) — not in git (114MB);
2. `train_smooth.py` (~7 min MPS) → new body;
3. refit `prepare_fix.py` (edge) + `gen_tail_counts.py` + `fit_tail_amplitude.py` (tail);
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
   throws).
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
   realization = 4e-4 shift) — compare clipped cores/ensembles only. REMAINING:
   PDF-level acceptance (KL+tails vs brute) to pick the production subhalo_factor,
   then regenerate ML training data. See
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
   when a single host's `Nc ≥ parallel_threshold`, default 2e5). At the production
   `subhalo_factor=1e-5` hosts have few clumps, so it buys ≤1.09× and forcing it on
   (`thr=1`) makes model 3 2–3× SLOWER (thread-spawn per host encounter). It only helps at
   tiny `subhalo_factor` (huge Nc, non-production): factor=1e-8 → 2.76×. It also reseeds RNG
   substreams (`subhalo.cpp:463`) → NOT bitwise-reproducible when it fires. **Correct lever =
   process-level parallelism over seeds** (each realization independent, deterministic per
   shard): ~**4.4× at 8 processes** for model 3 (z_s=1: 108→645 real/s). Sub-linear (memory-
   bandwidth-bound, heavy NFW/invRad/Wsub table lookups), and **8 procs beats 10** on a 10-core
   box (oversubscription). Small batches under-amortize the ~2 s/proc precompute (8k real →
   only 2.8×). Bench scripts: `tmp/bench_kappathr.py`, `tmp/bench_threads.py`, `tmp/bench_mp.py`.

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
