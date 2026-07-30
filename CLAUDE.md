# CLAUDE.md — gw-wl-emulator

Fast ML emulator for the GW weak-lensing magnification PDF, built on the C++ Monte-Carlo
engine from Vaskonen (2026). See `README.MD` for the project overview and `memory/MEMORY.md`
(auto-loaded) for cross-session context.

**Paper (PRD draft, .tex on Overleaf):** before touching paper text, figures, or
answering draft comments, read **`paper_prod/paper_writer.md` (2026-07-28) FIRST** —
the writing contract: file ownership, the `\B{}` = "differs from production.tex"
convention + its verification script, prose style (synthesis of Vaskonen 2026 and
Baltabay+ 2026 arXiv:2607.01333), the verification contract, and the open items.
Then `paper_prod/paper_memo.md` (2026-07-20) — maps every
draft section/equation to its implementation (file:line), figure scripts, and the
claim-constraining standing rules; `paper_prod/draft_comments_memo.md` holds the
resolved \R{}/\Gala{} comments + known draft↔code mismatches (R_s/window, model-3
description, stale figure paths). Keep all three updated like this file.

**Two .tex files, different edit rights (2026-07-23):**
- `paper_prod/production.tex` — pasted straight from Overleaf. **User-only: Claude may
  read it (to compare/recommend/report) but must NEVER edit it.** The user reviews
  Claude's draft edits, then merges what they like into this file themselves, on their
  own schedule.
- `paper_prod/draft_revised_2026-07-20.tex` — Claude's working copy. Make requested text/
  figure changes here.
Default behavior: read both when asked to compare, edit only the draft, and never write
to `production.tex` even if a diff would be trivial to apply — surface the recommendation
instead and let the user apply it to production.tex themselves.

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
**`subhalo_carve` (2026-07-22, default `true`; all samplers + config dict):**
mass-conserving realized-clump host carve (scheme A) — a subhalo-bearing host is
built at `M − Σ_i m_i − M_u(r)` (reduced by the REALIZED resolved-clump mass + mean
unresolved mass) so the total halo mass is `M` exactly every realization, instead of
the deterministic `(1−f_s,b)M`. Gated to `subhalo_model=3` + the brute reference
(model 1/2 + `subhalo_brute`); `false` reproduces the pre-2026-07-22 reduction
(A/B + bitwise reference). RNG-stream-preserving (host build draws no randoms), so
carve-off is bitwise-identical to the old model 3. Guard clamps `M_host<Mmin` and
counts in `LensingProfile.subhalo_carve_negatives`. Design/verification:
`docs/subhalo/mass_conserving_carve_note.md` §9; `tests/test_subhalo_carve.py`.
**⚠ Mac rebuild + acceptance still pending** (see that note §7).
**`subhalo_model=4` — supervisor's simplified production model (2026-07-23, STAGED
in C++, default still 3):** drop the whole unresolved apparatus — every subhalo
sampled individually down to `psi_min = m_floor/M` (= `M_min/M`, floor `1e7`), host
carved to `M − Σ_i m_i`, **no `M_u`, no `κ_u`/Wsub, no dynamic floor, no
`subhalo_factor`**. Equivalent to `model 1 + subhalo_brute + subhalo_carve` as one
named model (requires `subhalo_carve=true`, throws otherwise; ignores
`subhalo_brute`/`subhalo_factor`). Edits: `lensing.cpp` guard + carve gate,
`subhalo.cpp::addClumps` brute-floor for model 4, `lensing.h` doc; **no default/binding
changes (STAGED — bindings already accept `subhalo_model=4` as an int).** **Mac
`make build` + gates PASSED 2026-07-23:** 29/29 — 11 `test_cosmology_params.py` incl.
`test_backward_compat_bitwise` (default path bitwise-clean post-rebuild), 10
`test_subhalo_carve.py` (the previously-pending Mac carve acceptance), 8 new
`tests/test_subhalo_model4.py` (defining invariant: model 4 ≡ `model1+brute+carve`
bit-for-bit; carve-off throws; brute/factor knobs dead; floor moves substructure;
default still 3). Pre-rebuild `.so` backed up in the session scratchpad.
**Cost + floor validated 2026-07-23** (`scratchpad/floor_sweep.py`,
memory `subhalo_brute_production_floor`); **cost figure CORRECTED 2026-07-24 — the old
"~1.3–3.4× model-3" was an N=300 artifact where the ~4.5 s fixed precompute swamps the
per-ray work.** Marginal PER-RAY cost (two-point fit, N=400 vs 4000, z_s=1, `sample_lnmu`):
model 4 brute ≈ 45–90 ms/ray vs thresholded model 3 ≈ 0.19–0.20 ms/ray = **model 4 is
~300–500× more expensive per ray** (default factor 1e-2 thr=1.3e-6, or factor 1 =
host-thr 1.3e-4 — both ≈0.2 ms/ray; subhalo-off ≈0.06–0.24 ms/ray, run-to-run jitter
±2× on the sub-ms terms, but the brute ratio is robust). The N=300 benchmark reproduces
3.6×, N=3000 gives 47×, asymptotic per-ray is few-hundred×. So the brute cut is NOT
cheap at production N — its whole cost is rendering the ~3.4e4 sub-threshold clumps that
move σ_κ by 0.2% (see `docs/subhalo/subhalo_kappa_threshold_note.tex` +
`playground/analytic/sigma_vs_subkappathr.py`). Floor `1e7/M` converged (flat 1e7→1e8;
bites only ≥1e9). Draft rewritten
(`draft_revised_2026-07-20.tex` §Subhalos: Eq.reducedhost simplified, `kappau_moments`
+ `ε_sub` para removed) + **Fig 4 replaced** (`fig:subhalo-factor` → host/subhalo
convergence+scatter decomposition, `paper_prod/scripts/plot_fig_subhalo_sigma_decomposition.py`
→ `fig_subhalo_sigma_decomposition.{png,pdf}`; single-host Campbell). **Fig 4
cross-checked vs production C++ 2026-07-23:** `playground/subhalo_single_host_probe.cpp`
(compile cmd in its header; runs production `addClumps` model-4 + carve on the grid host
nearest M=1e13, z_l=0.5) → `tmp/subhalo_single_host_probe*.csv`; comparer
`scratchpad/compare_fig4_probe.py`. ⟨N⟩ exact (3659.8 vs 3659.9), host mean <0.3%,
sub-dominates-σ + negative host–sub cov confirmed at 25/25 r-points; σ_sub pointwise
is heavy-tail MC-noisy (~±30% at 20k reals — deep 100k rerun for certification).
**Flip to default + retire model-3 code + emulator
retrain = pending Ville sign-off** (bundle with the bias_model/bias_window/R_⊥ flips).
**⚠ Prefer `subhalo_model=5` over 4 for any new production run — same physics, 217×
cheaper (see next entry). Model 4 is now the REFERENCE, not the candidate default.**

**`subhalo_model=5` — model 4 + per-clump κ threshold (2026-07-27, STAGED, gated):**
solves model 4's cost. **Same population** — every subhalo still exists down to
`psi_min = m_floor/M`, no unresolved/Gaussian stand-in, carve intrinsic (throws without
it; also throws on `subhalo_brute`) — but only clumps whose **κ at the ray** exceeds
`kappa_thr,sub` are RENDERED. This is the same rule the HOST halos already obey
(`r_thr`/κ_thr), applied self-consistently to subhalos, so it is NOT the model-3
weak/unresolved split the supervisor rejected: nothing is replaced by a statistical
stand-in, dropped clumps just keep their mass in the smooth host via the carve.
Knobs `subhalo_kappathr` (absolute) / `subhalo_kappathr_factor` (× host κ_thr,
**default 0.1**), wired through wrapper + all 5 py entry points + config dict.
**Draw-and-reject saves NOTHING** (the test needs m and d, so all ~1e6 clumps would
still be instantiated) — `Subhalo::addClumpsRestricted` samples the RESTRICTED
intensity: retention disc `d ≤ D(m)` with `D` = the `r_thr` reach table (built at
κ_thr,sub for model 5, NOT `subhalo_factor·κ_thr`), Poisson-thinned against a
**global** envelope `Smax = max_R Σ_n`; two exact branches (small-target = uniform in
disc × Σ_n/Smax; big-reach = full profile × `d≤D` test). Envelope costs extra
proposals, never accuracy; being global makes the proposal intensity **y-independent**
⇒ precomputes per (z,M) bin (`buildRestrictedBin`, `precompute(..., build_restricted)`).
**Cost (marginal ms/ray, z_s=1):** model 4 = 45.22, **model 5 f=0.1 = 0.208**,
f=1.0 = 0.119, model 3 = 0.19–0.20, subhalo-off = 0.108 ⇒ **217× cheaper than model 4
and already at model-3 cost.** Cost model verified independently: 45.11 ms / 1.11e6
clumps = 40.6 ns/clump, where the clump count comes from analytic Campbell + engine HMF
and the time from the actual C++.
**Gate vs model 4** (`scripts/convergence/subhalo_model5_gate.py`, 8 subprocess shards ×
1500 rays, `kappa_anchor=1`, f=0.1): JSD = 1.18/2.16/2.80e-3 at z_s=0.5/1/5 vs
same-model seed-split floors 3.9/4.2/5.7e-3 ⇒ **AT FLOOR at all three**; clipped means
agree to ~1e-4. ⚠ **Do not oversell this gate** — the predicted difference (0.08–0.21%
of the substructure part of σ_κ ≈ 0.03% of total) is far below what 12k rays resolve, so
"at floor" is the EXPECTED result and only excludes gross error; the analytic sweep is
what bounds the real error. ⚠ Clipped-σ ratios scatter few-% (0.948–1.026) because the
fixed ±1 clip is ~30σ at z_s=0.5 and trims nothing — use the body JSD, not clipped σ,
at low z_s. Sizing + population weighting (host weights dumped from the production
engine via `playground/host_weight_probe.cpp`, reproduces `NhfNFW` exactly):
**`data/results/subkappathr_population/report.md`**, driver
`playground/analytic/sweep_subkappathr_population.py`, single-host theory
`docs/subhalo/subhalo_kappa_threshold_note.tex`. Population-weighted clump reduction at
f=0.1 = 1.9e4/1.3e4/4.1e3× for σ losses 0.084/0.111/0.210% (z_s=0.5/1/5); f=1.0 costs
0.24/0.41/1.29%, which is why **0.1 is the recommended default** — the cost battle is
already won at 0.1, so the extra decade buys nothing measurable. Clump cost concentrates
in cluster hosts (median 2.4e14 M⊙ at z_s=1, 90% below 1e15).
**Grid kwargs (2026-07-11, for the Mmin/Nz convergence studies):** all samplers +
`compute_lnmu_stats` take trailing `NM=100, Nz=100` (mass/z grid sizes); the helpers
`get_kappa_threshold`/`get_expected_halo_count`/`get_sigma_background` take trailing
`Mmin=1e7, NM=100, Nz=100` (previously hard-coded); `get_simulator_config` dict now also
reports `Mmin/NM/Nz`. Defaults verified bit-identical to the unpatched module (Linux
sandbox side-by-side build: all samplers, subhalo-on model 3, explicit-legacy-kwargs
paths); Mac `make build` + all 11 `tests/test_cosmology_params.py` (incl.
`test_backward_compat_bitwise`) passed 2026-07-12. `compute_lnmu_stats` still does
NOT expose `Mmin`.

### 1+6d parameterization — Θ = {h, z_eq, Ω_M, Ω_B, **σ₈**, n_s} (amplitude switched A_s→σ₈ 2026-07-27)
**AMPLITUDE PARAMETER IS σ₈, not A_s (user decision 2026-07-27).** Matches Vaskonen
(2026) in both the paper (MCMC parameters {Ω_M, h, σ₈}; priors σ₈∈[0.4,1.4],
h∈[0.59,0.76], Ω_M∈[0.15,0.47]) and upstream `halos` (`main_lensing.cpp:29`
`C.sigma8 = 0.811`, `par = {OmegaM, sigma8, h, 1.0}`). **No C++ change was needed** —
σ₈-normalization is already the default path (`As=-1.0`). The A_s-mode below is
retained as an unused escape hatch. NOTE the `As>0` branch in *halos* is OUR commit
`8ba6968` on the local branch, NOT Ville's — upstream is σ₈-only.

⚠ **OPEN — must discuss with Ville and then move to option (b) (he is on holiday as
of 2026-07-27).** The engine normalizes through `sigmaC`, which uses the smooth-k
window `Ws(x)=1/(1+(0.43x)^6)`, NOT a real-space top-hat: at the fiducial the
top-hat σ₈ is **0.7786** vs the code's 0.811 (+4.2% in σ, ~8.5% in P(k);
`tmp/engine_checkpoints.txt`). **Vaskonen's paper §2 explicitly states a real-space
top-hat** ("We compute the latter using the real space top-hat window function …
σ8 = σ_M(R = 8h Mpc)"), so his own text and code disagree. Half of this is
deliberate and right — a smooth-k filter is the standard excursion-set choice for
σ_M(M), since a top-hat gives no Markovian random walk — the questionable part is
only that the *normalization* inverts the same `Ws` at M8.
- **Option (a) = CURRENT**: keep his code convention. Zero code change, bitwise-safe.
- **Option (b) = AGREED TARGET, needs his sign-off**: keep `Ws` for σ_M(M) but anchor
  the normalization to a top-hat σ₈ at 8 Mpc/h. Not cosmetic — raises `deltaH8` 4.2%,
  P(k) ~8.5%, exponentiated by the HMF at cluster masses, and it redefines the number
  in his abstract. Bundle with the bias/subhalo default-flip sign-off.

A_s-mode (legacy escape hatch): all entry points take trailing kwargs
`As=-1.0, OmegaB=0.0493, zeq=3402.0, ns=0.965`.
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

**ML plumbing (σ₈ re-parameterization done 2026-07-27, main repo only; not retrained):**
`ml/params.py` = single source of truth (FIDUCIAL, PRIOR_6D, WIDE_6D,
`CONTEXT_KEYS = (z, h, Om, sigma8, Ob, ns, zeq_k)`, zeq_k = zeq/1000). σ₈ sits at
context index 3, so the 6d layout's first four columns now coincide with the legacy
1+3d one. Prior boxes returned to the legacy/Vaskonen values: ID σ₈ (0.65,1.05),
WIDE (0.40,1.40). `python/generate_dataset.py` samples the 6d wide box (LHS,
σ₈-mode, **HDF5 schema 2.1**, `amplitude_mode="sigma8"`, stores per-config
`As_derived` as the diagnostic); `ml/data.py` auto-detects 2.1 / 2.0 / legacy → X is
(N,7) or legacy (N,4). Schema discriminator is **`OmegaB` presence, not σ₈** (legacy
1+3d files carry σ₈ too). Old schema-2.0 A_s files still load: their stored
`sigma8_derived` is fed in as the amplitude column, lossless to 1 ULP.
⚠ **The 6d fiducial POINT MOVED**: the old A_s-mode fiducial was Planck
A_s=2.101e-9 → σ₈_derived **0.860**; the new fiducial is σ₈=0.811 (→ A_s_derived
1.8695e-9), i.e. −11% in P(k) amplitude. Any pre-2026-07-27 6d reference/control
(`param_space_ref`, `groundtruth.npz`) is at the OLD amplitude — regenerate, do not
compare across the switch. **-ar worktree CONVERTED 2026-07-28** — `ml/params.py` and
`ml/data.py` copied from the main repo (verified strict supersets; main's `data.py`
still reads deprecated schema-2.0 A_s files via the stored `sigma8_derived`), and the
nine autoresearch files converted in place: `param_space_check`, `prepare_ar`,
`gen_tail_counts`, `gen_lowz_data` (θ dicts + simulator calls), `validate_pit`,
`validate_posterior` (simulator calls), `features`, `smooth_model`, `train_ar`
(docstrings). Every simulator call now passes σ₈ as the amplitude POSITIONAL with
`As` left at its −1.0 default. **Panels are preserved**: the old route
`As = FID_As·(s8/0.86)²` was exactly invertible (σ₈_derived ∝ √A_s), verified against
`get_simulator_config` — old-vs-new σ₈_derived agree to 3e-4 at all six panel corners
(residual is only the rounded 0.86 vs the true 0.859744). Import-smoke + an end-to-end
`param_space_check._worker` run pass. ⚠ 443 lines of PRE-EXISTING uncommitted work in
that worktree were preserved, not clobbered (diffstat 443→476, same 15 files).
⚠ **Still NOT done there: the retrain itself.** Caches remain on the old amplitude and
must be regenerated, not reused — `param_space_ref.npz`, `groundtruth.npz`,
`lowz_aug.npz`, `tail_counts_extra.npz`, and any `*_6d.json` feature-fit cache (its
amplitude slot was `lnAs10`, now σ₈). ⚠ The -ar validators do NOT pass
`PRODUCTION_CONFIG` — deliberately left alone, since the shipped emulator was trained
on the older physics and changing the validators' config now would invalidate the
comparison. Fold that into the retrain bundle.
In the **-ar worktree**: train_ar/
train_smooth infer context dim from data (old checkpoints default 4);
`ml/autoresearch/features.py` = shared edge/tail feature maps (legacy 10/14 exact,
6d 16/20); smooth_model dispatches on len(theta) — 6d needs refit `*_6d.json` caches
+ a context=7 body. Retrain order unchanged (see §NSF recipe); regenerate
`param_space_ref`, `groundtruth.npz`, `lowz_aug.npz`, `tail_counts_extra.npz` first.
~~NOTE: the -ar worktree's `build/` module predates the As kwargs — rebuild there.~~
**CORRECTED 2026-07-27: there is nothing to rebuild.** `gw-wl-emulator-ar/build` is a
**symlink** to `gw-wl-emulator/build` (made 2026-06-16), so the two worktrees share
one `.so` and a `make build` in the main repo updates both. Verified: identical inode.

## ⚠⚠ PAPER-DEFAULT FLIP (2026-07-29) — the C++ defaults ARE the paper config now
**User decision: "overall all defaults must be what was described in the paper."**
Nine settings flipped, so a bare no-kwarg call now runs the model the draft describes:
`subhalo` false→**true**, `subhalo_model` 3→**5**, `subhalo_virial` false→**true**,
`bias_model` 0→**1**, `bias_window` 0→**1**, `bias_Rperp` 8441→**20000.0**,
`bias_weak` false→**true**, `fil_bias` false→**true**, `kappa_anchor` 0→**1**.
(`subhalo_carve`, `m_floor`, `subhalo_kappathr_factor`, `kappa_anchor_cut` were
already the paper values.) **R_s = 20 Mpc is FIXED and confirmed by Ville** — this
part is no longer pending. Verified: no-kwarg == `**PRODUCTION_CONFIG` bitwise at
z_s=0.5/1/5.
- **Defaults live in THREE places** — `lensing.h` (struct), `lnmu_wrapper.h`
  (struct), and the `py::arg(...)` list in `python_bindings.cpp` (44 entries; the
  py::arg values SHADOW the structs, so flipping only the structs changes nothing
  through Python). A fourth copy — hardcoded literals inside `get_simulator_config`
  — **was silently reporting the pre-flip values**, which would have written wrong
  provenance into every dataset's `metadata/simulator_defaults`. Fixed properly: it
  now reads a default-constructed `SamplingParams`, so that drift class is gone.
- **`ml.params.LEGACY_CONFIG` + `legacy_config(**over)` / `production_config(**over)`
  (NEW).** ⚠⚠ **The paper defaults are a COUPLED set — a single-flag override can now
  THROW**: `bias_model=0` alone raises (`bias_weak`/`bias_window` are inherited ON and
  both require `bias_model=1`); `subhalo_model=3` alone raises (`subhalo_virial` is
  inherited ON and requires model 4/5). Before the flip these were harmless because
  the partners defaulted off. **Splat the whole `LEGACY_CONFIG` for a reference arm;
  use `legacy_config(bias_model=1)` to vary one setting against it.** ~20 A/B and
  figure scripts under `scripts/` do single-flag overrides and were NOT audited — they
  will throw or silently change physics. Fix them with `legacy_config()` as you hit them.
- **Attribution proof (protocol satisfied):** with the flip in place, an explicit
  `LEGACY_CONFIG` call reproduces the pre-flip reference **bit-for-bit at all 3
  points** ⇒ the flip moved DEFAULTS ONLY, no physics. Reference re-baselined;
  pre-flip vectors kept as `tests/data/reference_lnmu_pre_paper_defaults.npz` and
  **still guarded** by the new `test_legacy_physics_bitwise`.
- **Tests: 16 failed on the flip (all "asserts the OLD default"), all updated; suite
  green.** New: `test_legacy_physics_bitwise`, `test_defaults_are_the_paper_config`.
  `test_parallel_generation.py`'s guard was **INVERTED** — it used to assert the pin
  DIFFERED from the defaults (a no-op guard while they were staged); it now asserts
  they AGREE. ⚠ Several tests passed vacuously after the flip until fixed to reference
  the off-state EXPLICITLY (comparing the new default against itself) — when a default
  moves, re-read every `assert not array_equal` in that file.
- ⚠ **`test_sigma8_as_round_trip_samples` is no longer bitwise on the default path.**
  Legacy: 100% bitwise. Paper defaults: **0%**, max |Δlnμ| = 1.8e-7 — realized subhalo
  and clustered-count draws depend on `deltaH8`, so a 1-ULP amplitude change
  decorrelates the RNG stream. Stream divergence, not amplitude error; the test now
  asserts an ABSOLUTE tolerance (lnμ crosses zero, so rtol is wrong) and keeps the
  bitwise check on the legacy path.
- ⚠ **The suite is now much slower** (~10 min for 6 physics files, was seconds) —
  every test that used to run cheap default physics now runs model-5 subhalos + the
  correlated field + the weak arm.
- **Must regenerate on the new defaults:** any stored subhalo-ON/bias reference, the
  model-4/5 gate, the subkappathr sizing, and any figure whose script relied on the
  old defaults.

## Production simulator config — PINNED IN PYTHON (2026-07-27)
`ml/params.py::PRODUCTION_CONFIG` is the single source of truth for the physics the
paper describes, passed EXPLICITLY on every call so the ML pipeline does not depend on
the staged C++ default flip:
`subhalo=True, subhalo_model=5, subhalo_carve=True, m_floor=1e7,
subhalo_kappathr_factor=0.1, subhalo_virial=True, bias_model=1, bias_window=1,
bias_Rperp=20000.0, bias_weak=True, fil_bias=True, kappa_anchor=1,
kappa_anchor_cut=1.0` (hash `0d50caf91c75`; was `e9150c0370af` before
`subhalo_virial` was folded in 2026-07-28). `subhalo_carve`, `m_floor`, `subhalo_kappathr_factor` and
`kappa_anchor_cut` are already the shipped defaults — pinned for explicitness and
because model 5 THROWS without carve. Threaded through `python/generate_dataset.py`
(splatted into the sampler worker).
**Datasets are now self-describing:** `metadata/physics_config` records what was
PASSED, `metadata/simulator_defaults` what the C++ compiles in. ⚠ The pre-2026-07-27
code wrote only `get_simulator_config()` — which reports the compiled-in DEFAULTS, not
the config used — so any older dataset's `filaments`/`bias`/`ell`/`Nhalos` attrs
describe the defaults and NOT necessarily that run. Gated by
`tests/test_parallel_generation.py` (which also asserts the pin differs from the
defaults, so it cannot silently become a no-op). That test was ALSO stale — it still
asserted schema 2.0 / `amplitude_mode == "As"` after the σ₈ switch; fixed to 2.1.

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
1. regenerate datasets (`python/generate_dataset.py --log_z`, splats PRODUCTION_CONFIG)
   + `cache/lowz_aug.npz` (`gen_lowz_data.py`) — not in git (114MB);
2. `train_smooth.py` (~7 min MPS) → new body (context dim inferred from data: 7 for 1+6d);
3. refit `prepare_fix.py` (edge) + `gen_tail_counts.py` + `fit_tail_amplitude.py` (tail).
   **`fit_flux_target.py` is NO LONGER NEEDED** — see the flux entry below;
4. acceptance gates: `param_space_check.py` (needs fresh `param_space_ref`),
   `validate_kl.py`, `validate_pit.py`, `validate_posterior.py` (control must be ~0σ).

**PIPELINE REWIRED FOR THE RETRAIN (2026-07-29). Three changes, all in -ar:**

**(a) `ml/autoresearch/simcfg.py` (NEW) — one switch for the physics, size and cache
paths of every -ar simulator call.** Until now the -ar scripts called the simulator
with NO physics kwargs, i.e. on the compiled-in C++ defaults (`subhalo_model=3`,
`bias_model=0`, `fil_bias`/`subhalo_virial` off, `kappa_anchor=0`) while
`generate_dataset.py` used PRODUCTION_CONFIG — so training data and the
edge/tail/flux calibrations fitted on top of it sat on **different physics**. That
hold was deliberate (the shipped emulator predates the new physics) and is now
lifted. `SIM_CONFIG` (= PRODUCTION_CONFIG) is splatted at all 7 pipeline call sites
(`gen_lowz_data`, `gen_tail_counts`, `param_space_check`, `validate_pit`,
`validate_posterior`, `prepare_fix`, `prepare_ar`). ⚠ **Never add a bare simulator
call to that package again** — that is how the two physics mixed silently. Escape
hatch `AR_SIM_LEGACY=1` reproduces pre-2026-07-29 results. The historical experiment
scripts (`validate_ar`, `validate_flow`, `validate_alpha2`, `measure_tail`, `l1_check`,
`plot_*`) were deliberately NOT patched — they reproduce exp18–24 figures.
- **Size knob `AR_N_SCALE`** (float, default 1.0) scales every hardcoded sample count
  via `nsamp()`, so the same code path smoke-tests in seconds
  (`AR_N_SCALE=0.02 python -m ml.autoresearch.gen_lowz_data`). A scaled run is
  valid-shaped and **statistically worthless** — never quote a number from one.
- **`cache_path()` tags caches** by physics/size (`lowz_aug.smoke0p02.npz`,
  `*.legacy.npz`); production+full-size keeps the bare historical filename so
  existing caches stay valid. This exists because a smoke run **did** clobber the
  119MB `lowz_aug.npz` and `stats_smooth.json` (the standardization stats the
  SHIPPED body depends on — `train()` rewrites it every run). Both were restored;
  `STATS_PATH` is now tagged too.
- **`AR_DATASET_DIR`** overrides `train_ar.DATASET_DIR` (colon-separated).
  ⚠ `datasets_logz_1k` is GONE from disk — the training set must be regenerated.

**(b) Flux calibration: `FLUX_TARGET = "unit"` — ⟨1/μ⟩ = 1 exactly (user decision
2026-07-29).** `"trim"` restores the legacy regressed `F_trim(ctx)`.
**Why it is now safe:** `"trim"` existed because the OLD `sample_lnmu` anchored
⟨κ⟩=0 on the EMPIRICAL batch mean, so a few κ>1 monster rays dragged the raw flux to
~1.08 and F_trim ran 1.0001–1.021 — targeting 1 then over-shifted by ~2%.
PRODUCTION_CONFIG's `kappa_anchor=1` fixes that at source. **Measured** (20k rays,
fiducial): ⟨1/μ⟩ = 1.000026/1.000093/1.000395/1.001363/1.002476/0.996085 at
z_s=0.3/0.5/1/2/5/10. 8-seed ensembles give 1.000379±0.000014 (z=1) and
1.002081±0.000090 (z=5) ⇒ the residual is **REAL, ~25σ, not MC noise**, but it is
0.04%/0.21% — an order of magnitude below what "trim" corrected, and of order ⟨κ²⟩,
the expected residual of a mean-anchored scheme. So targeting 1 is a deliberate
O(0.04–0.2%) departure from THIS simulator in favour of the exact theorem. The
z_s=10 row sits BELOW 1 and is tail-noise-limited — do not read it as a sign flip.
Verified: composite gives ⟨1/μ⟩=1 to 1e-5 (grid quadrature) vs legacy 1.0002→1.0050.
**Consequence: `fit_flux_target.py`, `cache/flux_target_fit*.json` and the 20MB
`cache/flux_grid.npz` drop out of the recipe** — one fewer fit and one fewer 6d cache.

**(c) Verified end-to-end at small N (2026-07-29), all on production physics:**
`generate_dataset.py --log_z` (40 cfg × 4k, schema 2.1, `physics_config_hash
0d50caf91c75` recorded) → `ml.data` → X (N,7) → `train_smooth.train()` → **context=7
body, finite log-p at a 6d context**. So the 1+6d plumbing works; only the fits and
the real training runs remain. ⚠ Still MISSING for a real retrain: the 6d caches
`edge_alpha_fit_6d.json` (prepare_fix) and `tail_amp_fit_6d.json`
(fit_tail_amplitude) — `smooth_model` dispatches on `len(theta)` and raises a clear
FileNotFoundError until they exist. `groundtruth.npz` (needed by `prepare_ar.evaluate`,
which `train_smooth.main()` calls) is also still at the OLD amplitude AND old physics.

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
   + resolved clumps + μ_unres(y) + N(0, σ²_unres(y)); exact in mean/variance
   at any factor; `subhalo_brute` + model 3 throws. **Host reduction updated 2026-07-22
   by the mass-conserving carve `subhalo_carve` (default on): the host is now reduced by
   the REALIZED `Σm_i + M_u(r)` (total mass = M every realization), not the deterministic
   FULL `f_s,b` mean — see the API section + `docs/subhalo/mass_conserving_carve_note.md`.** **Reproduction: subhalo-ON runs made
   before 2026-07-12 used factor 1e-5 — pass it explicitly.** ML training data regen still
   pending. Known factor-INDEPENDENT residual: all split arms sit ~2.3% below brute q99(μ)
   at z=5 (global JSD unaffected). Compare clipped cores/ensembles only — one κ~9 ray =
   4e-4 raw-Var shift. ~~Pre-existing unrelated failure: `tests/test_phase3b_local_density.py`
   (stale docs path).~~ **FIXED 2026-07-29** — the test expected `docs/phase3/...` but
   `phase3b_diagnostics.DOCS_DIR` is `docs`. The suite is now FULLY green.
   Derivation: `docs/subhalo/wsub_gaussian_term_derivation.md` §7;
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
   **WINDOW/SCALE DECISION (user, 2026-07-20): production = spherical TOP-HAT
   (`bias_window=1`) at R_s = 20 Mpc, NOT R_L(1e14)=8.44 Mpc.** R_L is a
   Lagrangian radius, not a smoothing scale; 20 Mpc is larger/PBS-cleaner and is
   Ville's original draft value. σ_TH(point) 0.972 (8.44) → **0.530 (20 Mpc)** ⇒
   clustering variance ~30%. Matched Gaussian at R_s=20: R_G=9.45 Mpc.
   **20 Mpc MC DONE 2026-07-20** (`data/results/bias_window/rs_dependence.md`,
   `plots/bias_window_rs_scan.png`; arms r05..r80 counts-only / rw05..rw80 joint,
   240k each, R_s = 5/10/20/40/80 Mpc × z_s = 0.5/1/5): at the production
   R_s=20 Mpc the FULL config (bias_weak on) gives Var_clip(lnμ)/no-clustering =
   **1.151 / 1.108 / 1.078** at z_s=0.5/1/5, JSD vs no-clustering
   3.2e-3 / 1.9e-3 / 9.3e-4 against floors 1.3/1.6/2.9e-4 — i.e. **clustering is
   NOT inert at 20 Mpc; it sits 3–24× above the sampling floor.** ⚠ Do not judge
   this from counts-only arms: those give only 1.074/1.043/1.018, so **the
   conditional sub-threshold arm (bias_weak) carries roughly half to two-thirds
   of the clustering effect at this scale** — a counts-only reading falsely
   suggests "inert" (this misled an earlier note here). Going 8.44→20 Mpc drops
   the boost +51%→+15% (z_s=0.5), +34%→+11% (z_s=1), +22%→+8% (z_s=5).
   ⚠ σ²(R_s) scaling is APPROXIMATE, not a law — normalized at R_s=20 the
   measured boost runs 6.5/3.0/1/0.30/0.12 (z_s=0.5) and 5.8/2.7/1/0.41/0.11
   (z_s=1) over R_s=5/10/20/40/80 Mpc vs σ²(R_s) 6.2/2.7/1/0.29/0.066: good to
   ~10% for z_s≲1 at R_s≲20 Mpc, but SHALLOWER than σ² at R_s≥40 (where the
   boost is only 1–5%, near clip noise) and shallower at all R_s for z_s=5
   (4.2/2.4/1/0.51/0.27). Do not quote it as a clean σ²(R_s) law. **Default STAGED, not flipped:** shipped
   default stays `bias_model=0`+`bias_window=0`; production config
   `bias_model=1, bias_window=1, bias_Rperp=20000` passed EXPLICITLY; full flip
   (incl. bias_model 0→1 + the 8.44→20 reversal) bundled for Ville's sign-off.
   Paper+fig updated: `draft_revised_2026-07-20.tex`, `fig_clustering_field.*`
   (headline 20 Mpc + 8.44 comparison). See §5b of the report + `bias_window_sigmaR.py`.

   **Window generalized to a selectable enum (2026-07-20, GATES PASSED — default
   flip PENDING user/supervisor sign-off, bundle with the R_⊥ sign-off):**
   `bias_window` 0 = transverse disk on k_⊥ (legacy, **bitwise default**) /
   1 = spherical top-hat / 2 = Gaussian, the latter two on the FULL modulus
   |k| = √(k∥²+k_⊥²) (`lensing.cpp::biasWindow2` + `BiasField1D::build`);
   requires `bias_model=1` (throws); wired through `lnmu_wrapper` + **all five**
   py entry points + config dict. Gates: pristine-HEAD-worktree build vs patched
   = 32/32 arrays bitwise (default AND explicit `bias_window=0`); 11/11 cosmology
   + 14/14 new `tests/test_bias_window.py`; k_⊥ grid converged for all windows at
   the existing constants (≤4.3e-5, gate 1e-4); replica-vs-reference |ΔCov| ≤9e-6,
   ⟨λ⟩ 0.998–1.001; **NEW production-C++-vs-replica probe** (`bias_window_field_probe.cpp`
   + `check_cpp_vs_replica.py`) agrees to **7e-13** — this is the gate that catches
   a window applied to k_⊥ instead of |k|, which every other gate passes.
   Physics (240k/arm, `kappa_anchor=1`): at the production R_⊥=8.44 Mpc
   **JSD(top-hat, disk) is AT THE SHARD FLOOR at z_s=0.5/1/5** (matched 120k-vs-120k);
   the resolvable difference is Var_clip(lnμ) +3.1/+1.3/+0.8%. **Candidate default R
   UNCHANGED at 8441 kpc** — the clustering-variance-weighted R_L scale is a
   population property and is *identical* for all three windows (2026-07-16 numbers
   reproduced exactly). Gaussian at the variance-matched **R_G = 3972.1 kpc
   (0.4706·R_TH; `bias_Rperp` is used AS GIVEN, no internal rescaling)** is at floor
   vs the top-hat ⇒ once R is fixed by σ(R), window shape is immaterial to P(lnμ).
   Weak arm composes unchanged: split-invariance JSD 3.36e-4 (z1) / 5.56e-4 (z5) vs
   a **new disk control** 3.40e-4 / 5.91e-4 — identical, so the Cox-split residual is
   not window-induced. **Edge unmoved by the window** (q01 shifts ≤7e-4 at z_s≤1 vs
   the weak arm's −1.4e-2) ⇒ no NEW emulator edge/flux refit beyond the weak arm's.
   ⚠ Counterintuitive but real: the top-hat has ~8% LESS per-shell variance yet
   ~6% MORE LOS-**coherent** power (Σ_ij Cov_ij) — κ integrates coherently, so
   Var(lnμ) goes UP; do not reason from trace/σ²_shell alone. Exact identity the
   disk cannot have: σ²_point(δ_1D)=σ²(R) to 1.0000 (0.9997 with the mode cutoff);
   the cutoff removes 4.0% of the DISK field's σ but 0.03% of the top-hat's.
   Full evidence: `data/results/bias_window/{report.md,tables.md}`,
   `plots/bias_window_scan.png`; scripts `scripts/convergence/bias_window_{scan,
   sigmaR,weighted_scale}.py`.

14. **`fil_bias` — filament clustering bias (STAGED, default false; ALL GATES CLOSED
   2026-07-27):** filaments ride the PBS bias of the code's OWN filament barrier
   `cosmology::filbias` `(p,q)=(0,0.7)` instead of borrowing `halobias` `(0.3,0.8)`;
   10–20% lower. Requires `bias_model=1` (inert in the legacy iid layer, not a throw).
   Same field, no new RNG — `lambdaF` reuses the realized `bfvals`, so off is
   bitwise-identical. Mac gates: `make build` + `test_cosmology_params.py` 11/11 (its
   reference predates fil_bias ⇒ that IS the bitwise gate) + new
   `tests/test_fil_bias.py` 10/10 (incl. a dead-flag guard and an all-five-entry-point
   check). A/B at the full PRODUCTION_CONFIG, 480k rays/arm at z_s=0.5/1/5 + 2M/arm at
   z_s=1 (`scripts/convergence/fil_bias_ab.py`): **JSD at floor everywhere, and the
   cross-JSD tracks 1/N** (6.22e-5 → 1.53e-5 for ×4.17 rays) — the signature of a null,
   bounding any true offset to J\* ≲ 1e-5 ≈ 500× under the emulator KL. Clipped σ shifts
   −0.4 to −0.8%, negative at every z_s (predicted direction), but ~2σ and it does NOT
   firm up with statistics — **quote "of order −0.4% at ~2σ, unresolved"; do NOT quote
   −0.78%**. Edge Δq01 +2–4e-4 (35–65× below the weak arm's −1.4e-2), flux Δln⟨1/μ⟩
   ~1e-6–2e-4 ⇒ **no new emulator edge/flux refit** beyond the weak arm's. ⚠ A null was
   the EXPECTED outcome (filaments are subdominant) — this bounds, does not measure.
   The case for `fil_bias=true` is physical correctness: the draft states the (0,0.7)
   barrier, and with the flag off that sentence is false in the code. Evidence:
   `data/results/fil_bias/report.md`, `docs/filament_bias_note.md` §5.
   Gotcha the A/B test caught: `sample_lnmu`/`compute_lnmu_stats` take
   `(z, OmegaM, sigma8, h, Nreal, …)` but the `*_ml` family takes
   `(z, h, OmegaM, sigma8, nsamples, …)` — pass by keyword.

15. **Subhalo radial profile SHAPE FIX (2026-07-28) — the significant physics result;
   NOT bitwise, no flag (the old form was simply wrong).** All three sites used
   `dN/dx ~ x^2 B(x)/(1+cx)^2`. Green+21 define B as a ratio of **volume number
   densities**, so B multiplies rho_NFW and `dN/dx = 4 pi r^2 n_sub ~ x B/(1+cx)^2` —
   the x^2 shell factor cancels ONE power of x against the NFW 1/x cusp. The old form
   was a CORED `n_sub ~ B/(1+cx)^2` (inner slope x^2.25 not x^1.25, outskirt x^-2 not
   x^-3), contradicting both `B->1` and the Han+16 x^1.3 inner bias B is fitted to.
   **Impact:** substructure mass within 0.3 r200 **x3.0** (stable in M and z; f_s and
   psi_res unchanged, so total mass per host is the SAME — it REDISTRIBUTES inward);
   clipped sigma(lnmu) **CONFIRMED at depth, and ATTRIBUTED: the profile fix ALONE is
   +1.539+/-0.194 % (7.9 sigma) at z_s=1, 2M rays/arm.** ⚠ The same-day
   `halobias` q 0.75->0.8 edit in cosmology.cpp confounded the first A/B — combined
   profile+halobias is +1.895+/-0.219 % (8.7 sigma) at z_s=1 and +1.277+/-0.110 %
   (11.7 sigma) at z_s=5; `halobias` alone is +0.351+/-0.243 % (1.4 sigma, NOT resolved).
   Additivity closes to 0.005% (1.890 vs 1.895), confirming the split. **Do NOT attribute
   +1.90% to the profile.** Central value stable under a 4x depth increase and sigma
   scaled as sqrt(N) = the signature of a REAL effect, unlike `subhalo_virial` and
   `fil_bias`, whose central values COLLAPSED under the same test.
   ⚠ **Always `git diff --stat cpp/` before attributing an A/B to a named change** — this
   confound came from assuming the only edit was the one described in a code comment. q99 +2.7-2.9%, edge q01
   -9e-4..-3.1e-3. At 480k the binned JSD sat at ~1.0x floor even where the effect was
   real (a 1-2% width change barely reshapes a 161-bin histogram); at 2M it clears the
   floor (1.4-2.2x) — so **use sigma for width changes, and treat "JSD at floor" as a
   statement about binning and depth, not about the effect.** Must regenerate:
   **Fig 4** (`fig_subhalo_sigma_decomposition` — sigma_sub up at small r, down at
   large r), the **model-4/5 gate** and the **subkappathr population sizing** (clump
   reach D(m) unchanged, but model 5's envelope thins against the projected Sigma_n,
   which moved), and anything with a stored subhalo-ON reference
   (`test_backward_compat_bitwise` is safe — its reference has subhalo=false). ~2% on
   sigma is ~4e-4 in KL (under the emulator's 7.3e-3) but it is SYSTEMATIC, and the
   width carries the sigma_8 information — at sigma(lnmu) ~ sigma_8^2 the profile fix's
   +1.54% is **~0.75% in sigma_8**, a central-value shift not a widening, so state it in
   the paper rather than absorbing it. (`subhalo_virial` and `halobias` add nothing
   resolvable — both at floor.) Evidence:
   `data/results/subhalo_profile_fix/report.md`, `docs/subhalo/subhalo_combining.md`.

16. **`subhalo_virial` — JvdB14 virial convention (2026-07-28, C++ default OFF, but ON
   in PRODUCTION_CONFIG).** JvdB14 sec. 2 defines haloes/subhaloes inside their VIRIAL
   radii, so f_s and psi=m/M are M_vir quantities and the population extends to r_vir;
   Green+21 normalizes the radial bias at r_vir. The engine's M is M_200c (`NFWlistf`:
   200 rho_crit, `Az=E^2`) and `cons14` is DM14's **c200** relation. Only the bias SCALE
   x0 was converted (`etaVirTo200` — verified vs an independent brentq to <=4e-16, and
   two routes to M_vir/M_200 agree exactly; bracket safe since Delta_vir <= 18pi^2 < 200
   always => eta>1). The EXTENT and MASS normalization were not => excess substructure
   inside r_200 of **1.20/1.12/1.08/1.04 at z=0.1/0.5/1/5** (recomputed with the
   CORRECTED profile; the earlier 1.49/1.28/1.18/1.09 used the wrong shape — do NOT
   quote those). The flag sets psi -> m/M_vir, samples the profile to x = eta =
   r_vir/r_200, and carves `M - Sum m_i/(M_vir/M_200)`. Gated to model 4/5 (0-3 reduce
   the host with M_200-referred incomplete-Gamma/Wsub tables; throws). Legacy bitwise
   (xmaxh=1, Mpsih=M => exact x1.0). Mechanism verified DIRECTLY by
   `playground/subhalo_virial_probe.cpp` (production addClumps: M_vir table matches
   mu(eta c)/mu(c) to 6 digits, r_max scales exactly as eta, <Sum m_i> tracks the
   prediction to ~3%) — this is what distinguishes "works but does not matter" from
   "silently inert". **PDF effect: AT THE SAMPLING FLOOR** on the corrected
   profile baseline — z_s=0.5 reads +2.17+/-0.66 % (+3.3 sigma) at 480k rays/arm but
   **+0.045+/-0.328 % (+0.1 sigma) at 2M**, i.e. the 3.3 sigma did NOT survive; z_s=1
   +0.43+/-0.39 %, z_s=5 +0.11+/-0.20 %; JSD at floor throughout. Quote |dsigma/sigma|
   <~0.7% at 2 sigma. ⚠⚠ **8-shard SEM yields 2-3 sigma FALSE POSITIVES** (SEM is a
   chi^2 on 7 dof, ~27% uncertain; several quantities read across 3 z_s) — this bit twice
   in one session (fil_bias -0.78% "2.2 sigma" -> -0.38%; virial +2.17% "3.3 sigma" ->
   +0.045%). **Require a depth-doubling confirmation or >=5 sigma before calling a
   sigma-ratio real.** ⚠ The analytic "excess inside r_200" is a population diagnostic,
   NOT a predictor of the lensing response (clumps beyond r_200 still lens; rays sample
   impact parameters outside r_200). **Decision
   2026-07-28: ON in production** — the draft claims a JvdB14 population, and OFF
   carries the LARGER forced-regeneration risk (population wrong by 4-20% vs the cited
   source, versus only the ~1%-of-M carve sub-choice left open). The carve conversion is
   OUR call, not settled by the sources. **Decision 2026-07-28 (user): KEEP
   `M - Sum m_i/(M_vir/M_200)` and flag it for Ville with the bundle** — the smooth host
   then keeps the same FRACTIONAL mass (1-f_s) in both apertures, where the literal
   `M - Sum m_i` would strip ~10% more than f_s from the M_200 budget. They differ by
   ~1% of M on the host. Ambiguous because the host NFW is untruncated and the clumps
   now reach r_vir, so there is no single well-defined "total" to conserve. Evidence:
   `docs/subhalo/virial_convention_note.md`, `tests/test_subhalo_virial.py`,
   `data/results/subhalo_virial/`.

17. **`halobias` q 0.75 -> 0.8 (2026-07-28, user decision: one (p,q) everywhere; NOT
   bitwise, no flag).** `cosmology::halobias` used q = 0.75 while the HMF barrier
   `cosmology::pFC` uses (p,q) = (0.3, 0.8), so b was NOT the peak-background split of
   the code's own first-crossing barrier — and the draft's clustering section already
   states "(p,q) = (0.3, 0.8)" for field halos, so the code contradicted the text.
   `filbias` already mirrored `pFCfil` at q = 0.7, so this makes the pair consistent.
   b rises ~1-3%, strengthening the clustering modulation lambda. **Effect NOT resolved:
   +0.351+/-0.243 % (1.4 sigma) on clipped sigma(lnmu) at z_s=1, 2M rays/arm, JSD at
   floor** — adopt on PBS consistency, not because it moves P(lnmu). **Broke
   `tests/test_cosmology_params.py::test_backward_compat_bitwise`** — the test doing its
   job on a deliberate physics change, NOT a regression. **RESOLVED 2026-07-28: reference
   RE-BASELINED** (user decision); old vectors kept as
   `tests/data/reference_lnmu_pre_halobias.npz`; 11/11 green.
   ⚠ **Protocol before ANY re-baseline** (documented in
   `tests/capture_reference_lnmu.py`): revert the suspected change ALONE and confirm the
   old reference still passes, so the re-baseline cannot silently absorb other drift.
   Done here — a build with only halobias reverted reproduced the 2026-07-08 reference
   bit-for-bit at all 3 points, proving halobias was the sole cause and that the same-day
   profile fix + `subhalo_virial` code are bitwise-clean on the default path. ⚠ halobias
   is used on the DEFAULT path (`samp.bias = 1` is hardcoded in the ML entry points), so
   this shifts every ray regardless of `subhalo`. Evidence:
   `data/results/subhalo_profile_fix/halobias_only_n250000.json` + report sec. 3b.

18. **Analytic-vs-C++ subhalo profile conventions RECONCILED (2026-07-28) — the §15
   re-derivations are DONE and `subhalo_kappathr_factor=0.1` STANDS (loss is smaller
   than originally certified).** Re-deriving the §15-invalidated items exposed that the
   analytic `playground/analytic/` chain disagreed with `cpp/subhalo.cpp` on **three**
   things, not just the shape §15 named:

   | | analytic (stale) | production C++ |
   |---|---|---|
   | shape | `x^2 B/(1+cx)^2` | `x B/(1+cx)^2` (§15) |
   | bias scale | `x0 = 0.54` read as **r_200** units | `BIAS_X0_RVIR = 0.86` in **r_vir** units → `0.86*eta(c,z)` in r_200 (refit to Klypin+11 Bolshoi pts; landed **undocumented** in commit `6bf0633`, 2026-07-27) |
   | extent / psi | `x <= 1`, `psi = m/M_200` | `x <= eta`, `psi = m/M_vir` under `subhalo_virial` (**ON** in PRODUCTION_CONFIG) |

   Combined, the stale chain carried a **~30% radius-dependent tilt** in `<kappa_sub>(y)`
   vs the engine (1.19 at x=0.15 → 0.92 at x=0.76); fixed it is ~6% away from the
   innermost point. Fixed via explicit `x0`/`xmax`/`shape_exp` args on
   `projected_profile` + `production_profile_params()`, and a `virial=` arg on
   `run_kthr` (psi→M_vir for clump lensing, →M_200 for the carve response).
   **Validated against production C++, not asserted:** `subhalo_single_host_probe.cpp`
   gained a `virial` argv; `scripts/convergence/check_analytic_vs_probe.py` compares.
   Median analytic/engine ratios at production (virial ON): `<N_c>` 1.0153,
   `<kappa_sub>` 1.0138, `sigma_sub` 1.0081; the virial jump reproduces exactly
   (engine ×1.0997, analytic ×1.0998). The residual ~1.5% on `<N_c>` is mass-grid
   quadrature, is radius-INDEPENDENT, and **cancels in the sigma ratios the sweep quotes.**
   **Attribution is clean** — `sweep_subkappathr_population.py --legacy-profile`
   reproduces the 2026-07-27 published table to every digit.
   **Revised sizing (production convention):** sigma loss at f=0.1 **FELL ~30%** to
   0.059 / 0.078 / 0.152 % at z_s=0.5/1/5 (was 0.084/0.111/0.210); clumps/ray rose ~10%
   to 88.5/93.6/108.6 (virial psi scale) = 3.8 us/ray, still ~3% of the 0.108 ms/ray
   fixed overhead. Both the cost and accuracy arguments get BETTER. f=1.0 costs
   0.190/0.333/1.123%, so the f=0.1 recommendation is if anything strengthened.
   **Model-4/5 gate re-run** (`--virial`, fresh outdir, corrected profile): z_s=1 at 96k
   rays/arm gives JSD 2.40e-4 vs floors 4.39/4.38e-4 = **AT FLOOR**, sd ratio
   **+0.58 +/- 0.97 % (0.6 sigma)**.
   ⚠⚠ **Two traps this re-run hit, both now guarded:**
   (a) the gate script CACHES shards by path — the default outdir still held
   2026-07-27 OLD-PROFILE `.npy` files, which would have been silently reused. **Always
   pass a fresh `--outdir` after a physics change.**
   (b) the pooled sd ratio reads +0.55% and `1/sqrt(2N)` suggests 0.23% SEM ⇒ "2.4 sigma".
   That is WRONG — the clipped sd is heavy-tail dominated, so its effective sample size is
   far below the ray count and the true shard-resolved SEM is 0.97%, i.e. **0.6 sigma**.
   The gate now computes and prints the shard SEM and labels it as the number to use.
   **Never read a sigma ratio off `1/sqrt(2N)` in this codebase.**
   **Fig 4 also had an INDEPENDENT normalization bug** (not the profile): it normalized
   the SHMF so `f_s` was the bound fraction over `[psi_min, 1]`, whereas the engine
   (`precompute`'s `gden`) defines JvdB14's `f_s` over the RESOLVED band
   `[psi_res = 1e-4, 1]`. Like-for-like at the engine's grid host that under-populated
   the host **1.169x** (3131 vs 3660 clumps) ⇒ `sigma_sub` low **8.1%**. Fixed +
   regenerated; `paper_prod/scripts/check_fig4_vs_probe.py` is the standing check
   (reproduces the engine's `gnorm` to 6 digits and its `<N_c>` to 3659.8 vs 3659.6
   measured). Three draft captions were stale as a result and were corrected in
   `draft_revised_2026-07-20.tex` — subhalo mean share is **4.6%→34%** across the
   aperture (not "a few per cent"), `sigma_sub > sigma_host` **everywhere** shown (not
   "beyond 0.15 r_200"), and `sigma_tot < sigma_sub` fails inside `0.046 r_200`. The
   figure script now prints these three claim checks on every run — **re-verify them
   whenever the profile or the SHMF normalization changes.**
   Evidence: `data/results/subkappathr_population/report.md` (REVISED block at top).

## Hubble diagram reconstruction (HDR) — downstream of the PDF (2026-07-29)
Everything after the magnification PDF: mock catalogue → likelihood → MCMC →
posteriors. Produces Vaskonen's σ8 forecast (10% ET / 30% LISA / 8% combined) and
his Fig. 5. **Not in the current paper**; design + code map in
**`docs/hubble_diagram_reconstruction.md`**. Read it before touching
`cpp/main_lensing.cpp` or `lensing.cpp::{loglikelihood,Hubble_diagram_fit}`.
Verified facts worth knowing here:
- `loglikelihood` (:1410) / `Hubble_diagram_fit` (:1490) are **NOT bound to
  Python** (`grep -c` on python_bindings.cpp = 0). Recommendation: reimplement the
  likelihood in Python rather than bind — the C++ one calls `Plnmuf` internally
  (:1446), so binding buys the slow stochastic MC, not an emulator-driven run.
- **Plane trap:** `Plnmuf` (definition :1364, conversion+renorm :1396–1405)
  returns the **source-plane** PDF, and both the likelihood AND the catalogue
  (`main_lensing.cpp` :120, :166) use it. Our emulator is image-plane ⇒ apply
  `P_S = μ⁻¹ P_I` *and renormalize* (the renorm is required, not automatic, and is
  θ-dependent). ⚠ `paper_memo.md` misquoted `Plnmuf` as :1163 until 2026-07-29.
- The C++ likelihood is **stochastic** (fresh MC per call, `Nreal=1e4`), so MH runs
  on a noisy logL; z is quantized to **6 nodes** (10 combined). The emulator makes
  it deterministic and continuous in z — a correctness upgrade, not just speed.
- The MCMC fits a **4th parameter the paper never mentions** — a DM-model mass,
  `par[3]`, prior [-0.6,1.4] in log10, inert in CDM but random-walked.
- ⚠ **Unbounded OOB read in the SMBH catalogue arm** (`main_lensing.cpp` :110 vs
  :113): the list is truncated to 10 but 12 accepted events are demanded, and the
  index advances on rejected events too. Verify before reusing stage 2.
- `DLthr` is hardcoded to 3.0e8 kpc (:231) = 300 Gpc > D_L(z=10), so the P_det
  threshold machinery **exists but has never been exercised**.

## Conventions
- Plots → `plots/`; throwaway/scratch → `tmp/`.
- Substructure reference papers are in `papers/misc/` (JvdB14, vdB05, Han+16, BMO09, SatGen,
  CUSP). Lift fitting-function numbers from the PDFs, not from memory.
- **Paper figure log-axis ticks:** decades -1/0/1 print as `0.1`/`1`/`10`; every other decade
  stays `$10^{n}$`. Not automatic — call `format_log_axis_decimal(ax, axis=...)` from
  `paper_prod/plot_style.py` per axis where the tick range includes 0.1/1/10.

## Multi-CLI delegation (offload token-heavy work; Claude orchestrates)
Claude keeps judgment, physics, edits, and **all git/gh/push**. Delegates never commit.
Available delegates: codex (large implementations/debugging), agy (bulk token-heavy
reads, mechanical cross-checks), gh copilot (shell one-liners). ALWAYS invoke the
`delegate` skill (`.claude/skills/delegate/SKILL.md`) before delegating — it has the
verified command lines (incl. hang gotchas) and the standing rules.
