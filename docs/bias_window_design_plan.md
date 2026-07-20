# Bias-field window generalization — plan + implementation design (handoff)

2026-07-20. Prepared for execution by a delegate model (Opus). Orchestrator/user keep
judgment, git, and paper text. Read this whole file before editing anything.

## 0. Context and goal

The clustering layer (`bias_model=1`) builds a 1D density field whose power spectrum
is currently computed with a **transverse disk window** acting on k⊥ only
(`cpp/lensing.cpp::BiasField1D::build`, "disk window via GSL J1"):
P_1D(k∥) = (1/2π) ∫ dk⊥ k⊥ P(√(k∥²+k⊥²)) W̃_d²(k⊥R), W̃_d(x)=2J₁(x)/x.
Along the LOS the only suppression is the sharp mode cutoff k_max = 2π/R (N_max = L/R).

Decision (user, 2026-07-20): generalize the window to a selectable
**`bias_window` enum** and move the headline choice to the **spherical (3D)
top-hat**, with a **Gaussian** variant as a robustness check. Rationale: the
peak-background-split bias wants an isotropically smoothed background field; the
disk window leaves unsuppressed LOS power up to a numerically-defined cutoff; the
top-hat matches the equation already in the paper draft and gives the elegant
identity σ²_point(δ_1D) = σ²(R) (the standard smoothed-field variance).

Status of defaults: `bias_window=0` (disk) stays the **bitwise default** until the
user flips it (after gates + physics numbers + Ville's sign-off, bundled with the
pending R⊥ = 8441 kpc sign-off).

References: `docs/bias_field_design_note.md`, `paper_prod/paper_memo.md` §II.A.1,
`paper_prod/draft_comments_memo.md` R1, CLAUDE.md item 13,
`data/results/rperp_pdf_scan/report.md`.

## 1. Specification

New config field (all defaults preserve current behavior bit-for-bit):

```
int bias_window = 0;   // 0 = transverse disk (legacy, bitwise default)
                       // 1 = spherical top-hat, W(x) = 3(sin x - x cos x)/x^3, x = |k| R
                       // 2 = Gaussian,          W(x) = exp(-x^2/2),            x = |k| R
```

- Windows 1 and 2 act on the FULL modulus |k| = √(k∥²+k⊥²), not on k⊥. The P_1D
  integral structure is unchanged; only the weight moves:
  f = k⊥² P(kk) W̃²(kk·R) with kk = √(k∥²+k⊥²)  (window 0 keeps W̃²(k⊥R)).
- R semantics: for windows 0 and 1, R = `bias_Rperp` as-is (8441 kpc = R_L(1e14)).
  For window 2, R is still the value passed in `bias_Rperp` — the VARIANCE-MATCHED
  Gaussian radius is computed offline (§4.2) and passed explicitly; the code does
  no internal rescaling. Document this in the lensing.h comment.
- Mode cutoff/N_max convention (L/R, floor 4) and L = 1.05 χ(z_s): UNCHANGED for
  all windows.
- `bias_window != 0` requires `bias_model = 1` (same guard pattern as `bias_weak`;
  throw std::invalid_argument otherwise). `bias_weak` composes freely with any
  window (it reads the realized field through the shell covariances — no separate
  window logic).
- `get_simulator_config` must report `bias_window`.

## 2. Implementation steps (in order)

Line numbers drift — grep the quoted anchors; function names are authoritative.

### 2.1 C++

1. `cpp/lensing.h` — add `bias_window` to `LensingConfig` next to `bias_Rperp`
   (anchor: "Comoving transverse window radius"); extend the bias-layer comment
   block (anchor: "Bias (clustering) layer model (2026-07-16") with the enum
   semantics and the Gaussian-R note above.
2. `cpp/lensing.cpp::BiasField1D::build` — anchor comment
   `"---- P_1D table (log-log interpolated; disk window via GSL J1)"`:
   - Thread `bias_window` into `build(...)` (new parameter; find the call site by
     grepping `\.build(` / `BiasField1D` in lensing.cpp and pass
     `L.bias_window` — same threading pattern as Rperp).
   - Window 0: keep the existing precomputed `W2[i]` on the k⊥ grid — byte-for-byte
     identical code path (do NOT refactor it; bitwise default is a hard gate).
   - Windows 1/2: W̃² depends on kk, so it CANNOT be precomputed on the k⊥ grid —
     evaluate inside the k∥ loop: `f = kperp[i]*kperp[i]*C.Pk0(kk)*Wt2(kk*R)`.
     Guard small x: top-hat W→1 for x < 1e-4 (use the series 1 - x²/10 below
     ~1e-2 if you prefer; just keep it smooth), Gaussian is exact everywhere.
   - Integration range check: `kperp_hi = 60/Rw` was tuned for the disk window.
     Gaussian dies by x≈8 (fine). Top-hat W̃² decays as x⁻⁴ with oscillations:
     verify convergence by doubling `nkperp` (2048→4096) and `kperp_hi`
     (60→120/Rw) in a throwaway build and comparing the P_1D table (rel. diff
     < 1e-4 over the tabulated range). If not converged, raise the constants for
     windows 1/2 only. Record the check in the report.
   - Keep the `Rw = max(Rperp, 10.0)` floor and the `nktab=600` table as-is.
3. `cpp/python_bindings.cpp` — add `py::arg("bias_window") = 0` to ALL entry
   points that expose `bias_model/bias_Rperp/bias_weak` (grep `bias_Rperp` —
   currently three blocks: `sample_lnmu`, `sample_lnmu_ml`(+diagnostics),
   `sample_lensing_raw_ml`) + wire `samp.bias_window`; add it to
   `get_simulator_config`'s dict.

### 2.2 Python replicas (must stay verbatim ports — validation gates depend on it)

4. `playground/bias_field/validate_field_covariance.py` — `Wdisk` (anchor line
   ~52) and the two `W2 = Wdisk(kperp * Rperp) ** 2` sites (~86, ~128): add
   `Wth`, `Wgauss` and a `window=` argument to `cpp_field`/P_1D helpers,
   mirroring the C++ kk-argument change exactly (same grids, same trapezoid).
5. `scripts/convergence/bias_field_prototype.py::LOSField` — anchor
   `"Q(kpar, R) = (1/2pi) int kperp P Wdisc^2 dkperp"` (~443–451): same change.
6. `paper_prod/scripts/plot_fig_clustering_field.py` — `P1D_curve` (~44–50):
   add window support (needed later for the paper figure; do not regenerate the
   figure yet — paper text is orchestrator scope).

### 2.3 Tests

7. `tests/test_cosmology_params.py` (or a new `tests/test_bias_window.py`):
   - default ≡ explicit `bias_window=0` bitwise, seed-for-seed (extend the
     existing `test_backward_compat_bitwise` pattern);
   - guard: `bias_window=1` with `bias_model=0` throws;
   - determinism: window=1 and window=2 runs are seed-reproducible;
   - smoke: sampler returns finite lnμ for window 1/2 with `bias_weak` on and off;
   - config dict reports `bias_window`.
   All 11 existing tests must still pass.

## 3. Validation gates (all must pass before any physics runs)

Environment (MANDATORY — system python3 is 3.13 and will not import the module):

```bash
PY=/Users/baltabay/miniforge3/envs/test/bin/python
make build          # CMake + pybind11 -> build/gwlensing.cpython-312-darwin.so
make pytest
```

1. **Bitwise default**: full-suite pass incl. backward-compat test; additionally
   one manual seed-for-seed comparison of `sample_lnmu(...)` default vs explicit
   `bias_window=0, bias_model=1, bias_Rperp=8441` against a pre-change run.
2. **Covariance layer** (per window 1 and 2): run
   `playground/bias_field/validate_field_covariance.py` with the new window —
   pass thresholds: |ΔCov|/diag ≤ ~5e-6 python-vs-C++, ⟨λ⟩ = 1 to < 0.5% per
   shell. (Reference: disk window re-passed at ≤ 4e-6 and 0.2% on 2026-07-20.)
3. **Weak-arm composition** (window=1 + `bias_weak=true`): joint determinism,
   guard throw, bias=0 no-op. Split-invariance JSD protocol as on 2026-07-16
   (two arms at ⟨N⟩=100 vs ⟨N⟩=300, 240k rays each, z_s = 1 and 5; scan-protocol
   shard caching): PASS = excess over permutation floors ≲ 3e-4 (reference
   floors ~2–3e-4; the 2026-07-16 disk run had z_s=5 excess ~2.5e-4 — same
   approximation class is acceptable).

## 4. Physics measurement runs (after gates)

Use 8 parallel seed PROCESSES (subprocess shards — mp.Pool hides worker
segfaults as silent hangs); clip to the κ_tot ≤ 1 core for every variance
number; JSD on P(lnμ) for distribution comparisons. NEVER quote raw
moments/tails (monster-ray junk; see CLAUDE.md standing rules).

### 4.1 Top-hat R-scan
Rerun the `data/results/rperp_pdf_scan/` protocol with `bias_window=1` over the
same R grid (± extend a factor 2 both ways). Outputs: JSD(R_i, R_j) matrix +
clipped Var(lnμ) vs R at z_s = 0.5, 1, 5. Re-derive the clustering-variance-
weighted scale under the top-hat (same weighting calc as the 2026-07-16 R⊥
decision) — this is the candidate new default R; do NOT change any default
yourself, report the number.

### 4.2 Gaussian at matched variance
Compute σ²_w(R) = (1/2π²) ∫ dk k² P(k) W̃²(kR) numerically for both windows
using the same P(k) tables (the prototype `Cosmo` class has them; do NOT use
the code's internal smooth-k σ(M) for this — different window family). Solve
σ²_G(R_G) = σ²_TH(8441 kpc); sanity expectation R_G ≈ 0.4–0.5 R_TH ≈ 3.4–4.2 Mpc.
Run ONE Gaussian point at (window=2, bias_Rperp=R_G): JSD vs the top-hat run at
R_TH = 8441 and clipped Var(lnμ), z_s = 1 and 5. Expected outcome: at floor or
near it (the "window-shape robustness" sentence for the paper).

### 4.3 Fiducial magnitudes (window=1, R = 8441 provisional)
Re-measure vs the disk-window 2026-07-16 reference: clipped Var(lnμ) at
z_s = 0.5 and 1 (disk gave +18–24% over counts-only), JSD(joint, counts-only)
(disk: 4.8e-3 / 7.7e-3 vs floors ~3e-4), and the low-μ edge location (q01/q05 of
lnμ) — if the edge moves visibly, flag "emulator edge/flux recalibration
needed" in the report (do not retrain anything).

## 5. Deliverables

- Code + tests as in §2 (NO commits, NO pushes — leave the working tree for user
  review; list every touched file in the report).
- `data/results/bias_window/report.md`: gate results table, kperp-grid
  convergence check, R-scan table + weighted-scale number, matched-R_G value and
  Gaussian comparison, fiducial magnitude table, edge verdict, open questions.
- One-paragraph summaries appended (dated) to CLAUDE.md item 13 and
  `paper_prod/paper_memo.md` §II.A.1 — mark clearly as "gates passed, default
  flip pending user/supervisor sign-off".
- Do NOT touch: paper .tex files, `paper_prod/draft_revised_2026-07-20.tex`,
  figure regeneration (orchestrator scope after the default flip);
  `~/Desktop/halos` (HARD RULE: user-exclusive).

## 6. Gotchas (each of these has produced a wrong verdict before)

- `sample_lnmu` positionals are (z, OmegaM, sigma8, h) — passing h first sets
  σ8 = 0.315 and kills field power ("at floor" false negatives).
- Raw variance at 20–100k rays is seed junk: one κ~9 ray = 4e-4 Var shift.
  Clip κ_tot ≤ 1 or use JSD; ensembles over seeds, never single runs.
- ACE import from repo root needs `sys.path.insert(0,'ace_lensing')` (namespace
  shadowing); gwlensing needs `sys.path.insert(0,'build')`. Not needed here
  unless cross-checking against ACE.
- Do not "converge" κ_min downward or raise Nz to chase tails — model
  parameters, not knobs (CLAUDE.md item 13).
- Runs before 2026-07-16 used bias_Rperp=3000 placeholder; subhalo-ON runs
  before 2026-07-12 used subhalo_factor=1e-5 — pass explicitly when reproducing.
- Legacy-match is NOT a correctness criterion (user ruling 2026-07-16); the
  bitwise gate applies to the DEFAULT path only.
- plots → `plots/`, scratch → `tmp/`; date every memo edit.

## 7. Open decisions (user/supervisor — do not resolve in code)

1. Default flip window 0→1 and the final R under the top-hat (bundled with the
   pending R⊥ sign-off for Ville).
2. Gaussian: paper-visible robustness check vs internal validation only.
3. Whether the paper's R quote stays "R_L(1e14)" if the §4.1 weighted scale
   lands elsewhere.
