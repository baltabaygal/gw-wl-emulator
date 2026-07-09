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

**Fix history 2026-07-02** (module rebuilt, 75 tests pass):
1. **KEPT — w̃_f:** `exp(-1.0)` → `exp(-0.25)` (Giocoli+2007 e^{−2f³}, f=½) in
   `cpp/subhalo.cpp` + 3 screen scripts + doc — old value inflated f_s by ~22–26%.
2. **REVERTED by user/supervisor decision — shear stays single-angle** (original Vaskonen
   convention). The spin-2 double-angle form is exactly right per realization
   (`tmp/shear_convention_check.py`, convention-free Jacobian ground truth) but the
   aggregate effect is ~0.5% on ⟨γ²⟩; consistency with the original code was preferred.
   Do NOT re-apply without being asked.
3. **REVERTED (wrong idea) — max(0, r−r200) floor:** NFW κ diverges on-axis so the
   worst-case criterion degenerates to brute for every ray inside r200, making
   `subhalo_factor` inert (caught by the factor-convergence scan). Floor is keyed to the
   host-center distance r (original design); the r-vs-d bias is absorbed by tuning
   `subhalo_factor` to the convergence plateau: `scripts/subhalo_gate/subhalo_factor_convergence.py`.
   The earlier "94%/11×" split validation was contaminated by this bug — retracted.

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
6. **sigmakappaW fix (2026-07-08, supervisor-approved, APPLIED here):** (a) log-annulus
   element is `2π r² dlnr`, not `π r²` — `PI` → `2.0*PI` in the accumulators; (b) Campbell's
   theorem for Poisson halo counts — `return sqrt(kappa2)`, no `-kappa1^2/Nh` subtraction
   (`Nh`/`kappa1` accumulators are then dead). halos patch applied 2026-07-08 at user
   request, left UNCOMMITTED for user review (user compiles/commits/pushes halos).
   Net σ²_W ≈ 2.10× original; ≤1% on Var(κ_total). Proofs + MC validation:
   `docs/sigmakappaw_measure_note.md`, `playground/sigmakappaw_poisson_vs_fixed.cpp`.
7. **eps_floor parametrized + floor-consistent injection (2026-07-08, emulator only):**
   `sigmakappaW(C, zs, kappathr, eps_floor=0.001)` — the outward stop is κ <
   eps_floor·κ_thr. Converged: default keeps 99.90% of K2, missing variance ∝ ε
   (`playground/k2_vs_floor.cpp`, `plots/k2_vs_floor.png`). The internal caller
   (`lensing.cpp` sample path) now holds the ABSOLUTE floor at 0.001·κ_thr_default
   so `custom_kappathr` sweeps don't drag it (old behavior collapsed σ_W at high
   κ_thr); default path verified bit-identical (seed 12345). Variance partition
   validated: measured total = √(σ²_explicit+σ²_W) flat at the full-Campbell 0.02728
   (zs=1, halo-only) — `playground/sigmaW_vs_kthr.cpp`, `sigma_explicit_vs_kthr.cpp`,
   `sweep_sigma_total_vs_kthr.py`, `plots/sigma_partition_vs_kthr.png`. Confirmed at
   zs=0.2–10 (8 redshifts, `plots/sigma_partition_zs_study.png`): total/plateau mean
   0.991–1.001 everywhere; σ_full grows 0.0036→0.109; handover NOT universal in
   κ_thr/κ_thr_fid (midpoint drifts ~1 decade over the range).
   **Analytic theory of σ_full(z_s)** (2026-07-09): exact moment decomposition
   σ² = M₂ − 2M₃/χ_s + M₄/χ_s² (moments of one source density P(z); NFW kernel
   constant C₂ = 1.4674011); proven σ ∝ z_s^{3/2} (z→0, coeff 0.0443 derived) and
   saturation σ_∞ = 0.154 (z→∞, parabola in 1/χ_s); BPL fit = interpolant only.
   See `docs/sigma_full_analytic_note.md`, `playground/campbell_moments.cpp`,
   `plots/campbell_asymptotics.png`.
   **Gotcha:** `paper_prod/scripts/plot_sigma_k_vs_kappa_threshold.py` plots
   `data/variance_sweep_data_z1.npz`, which is the SUBHALO-factor sweep (written by
   `scripts/figures/plot_variance_vs_factor.py`) — its x-axis label "κ_threshold" is
   wrong; regenerate from the new sweep before using in the paper.
8. **Wsub unresolved-subhalo term — derivation DONE, implementation NOT started
   (2026-07-09):** per-host split κ_halo = reduced host (FULL f_s,b) + resolved clumps
   + μ_unres(y) + N(0, σ²_unres(y)) is EXACT in mean/variance at ANY subhalo_factor
   (Poisson restriction theorem; verified to 6 digits, factors 1e-5…1). Key finding:
   the zero-mean Gaussian ALONE recovers almost nothing (0.844→0.851 at factor 1e-3) —
   the deficit is dominated by the mean-profile mismatch, so the deterministic
   μ_unres(y) table is REQUIRED and the host reduction must switch f_s,res(y) → f_s,b.
   subhalo_factor then becomes a pure performance/Gaussianity knob (dropped-c₃ share
   0.2%/0.7%/3% at 1e-3/1e-2/1e-1); m_floor absorbable too (sub-1e7 variance 0.4%).
   Acceptance must be PDF-level (KL+tails vs brute), NOT σ². See
   `docs/subhalo/wsub_gaussian_term_derivation.md`,
   `playground/analytic/wsub_partition_proof.py`. Post-reorg stale imports
   (`scripts.subhalo_factor_proxy_check` → `scripts.subhalo_gate.…`) fixed only in
   `playground/analytic/subhalo_factor_analytic_deficit.py` +
   `playground/dgate/subhalo_factor_dgate_area_scan.py`; other playground subdirs still stale.

## Conventions
- Plots → `plots/`; throwaway/scratch → `tmp/`.
- Substructure reference papers are in `papers/misc/` (JvdB14, vdB05, Han+16, BMO09, SatGen,
  CUSP). Lift fitting-function numbers from the PDFs, not from memory.

## Multi-CLI delegation (offload token-heavy work; Claude orchestrates)
Claude keeps judgment, physics, edits, and **all git/gh/push**. Delegates never commit.

| Tool | Delegate to it for |
|---|---|
| **codex** (OpenAI, `/opt/homebrew/bin/codex`) | Large implementations, hard debugging, scientific/ML software, `codex exec review` |
| **agy** (Antigravity, `~/.local/bin/agy`) | Bulk token-heavy reads, mechanical cross-checks (citations/units/notation) |
| **gh copilot** | Shell one-liners (`gh copilot -p "…"`); plain `gh` = GitHub CLI, run by Claude |

Verified invocations (2026-07-01):
- codex read-only: `codex exec -s read-only --skip-git-repo-check -C DIR "<prompt>"`
- codex write/run: `codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C DIR -o out.txt "<prompt>"`
  (⚠️ `-s workspace-write` alone HANGS in exec mode — verified 34-min silent hang.)
- agy: `agy -p "<prompt>" --model "Gemini 3.5 Flash (High)" --add-dir DIR --dangerously-skip-permissions --print-timeout 15m`

Standing rules: prompts must be fully self-contained (delegates see nothing from this
conversation; include absolute paths + the env python path above); always monitor with a
progress/timeout guard, never fire-and-forget; capture output to a file; sandbox read-only
unless writes are needed; output is ADVISORY — reproduce numbers yourself, require
file:line citations.
