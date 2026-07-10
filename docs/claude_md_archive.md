# CLAUDE.md archive — resolved history moved out of always-loaded context

Moved here 2026-07-10 (by /doctor, user-approved) to cut resident context. Each block is
verbatim from CLAUDE.md at the time of the move; one-line pointers remain there. If any of
this becomes active again, move it back.

## From "Current work — subhalo substructure gate (§4)"

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

## From "The `halos` fork" status list — item 6 (sigmakappaW fix)

6. **sigmakappaW fix (2026-07-08, supervisor-approved, APPLIED here):** (a) log-annulus
   element is `2π r² dlnr`, not `π r²` — `PI` → `2.0*PI` in the accumulators; (b) Campbell's
   theorem for Poisson halo counts — `return sqrt(kappa2)`, no `-kappa1^2/Nh` subtraction
   (`Nh`/`kappa1` accumulators are then dead). halos patch applied 2026-07-08 at user
   request, left UNCOMMITTED for user review (user compiles/commits/pushes halos).
   Net σ²_W ≈ 2.10× original; ≤1% on Var(κ_total). Proofs + MC validation:
   `docs/sigmakappaw_measure_note.md`, `playground/sigmakappaw_poisson_vs_fixed.cpp`.

## From "The `halos` fork" status list — item 7 (eps_floor + σ_full analytic theory)

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
