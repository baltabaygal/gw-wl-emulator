# Mmin / Nz grid-convergence studies — findings note (2026-07-12)

Companion to the three earlier knob studies (κ_thr rule 2026-07-10, subhalo_factor/Wsub
2026-07-09, eps_floor/σ_W measure 2026-07-08). Protocol identical to the κ_thr JSD
study: JSD of P(lnμ) against a refined-grid truth run as two independent seed halves
(floor = half-split JSD rescaled by c·(1/N₁+1/N₂)), disjoint seed blocks per config,
default fixed-⟨N⟩=100 threshold rule everywhere, 240k realizations per candidate,
z_s ∈ {0.2, 1, 5, 10}. Driver: `scripts/convergence/convergence_scan.py` (axes:
`mmin`, `mmin_pd`, `nz`; shard-cached, resumable, `--tag/--scale` pilot mode).
Fast arms: `scripts/convergence/analytic_arms.py` →
`data/results/convergence_analytic/`. Auto tables: `data/results/{mmin,mmin_pd,nz}_convergence/report.md`.
Figures: `plots/mmin_convergence.png`, `plots/nz_convergence.png`, `plots/ace_gap_vs_mmin.png`.

## Bindings prerequisite (step 1, 2026-07-11)

`cpp/python_bindings.cpp`: all samplers + `compute_lnmu_stats` now take trailing
`NM=100, Nz=100`; helpers (`get_kappa_threshold`, `get_expected_halo_count`,
`get_sigma_background`) take `Mmin=1e7, NM=100, Nz=100`; `get_simulator_config`
reports all three. Mac rebuild + all 11 `tests/test_cosmology_params.py` (incl.
`test_backward_compat_bitwise`) pass; default path bit-identical.

## Verdict in one paragraph

**Both defaults stand.** Nz=100 is converged at the JSD level (excess ≲ 1.2e-4,
at the measurement floor). Mmin=1e7 carries a small but real global error
(floor-resolved JSD excess ~1e-4…1e-3 depending on z_s and M-grid matching, worst
case z=1 ≈ 1e-3) — 7–70× below the production emulator's own median KL (7.3e-3),
so immaterial at O(1000) events. The one sensitive corner, as in the κ_thr study,
is the z_s ≳ 5 high-μ tail (see §Tails). No training-data regeneration is implied
by these results.

## Study 1 — Mmin (default 1e7 M⊙; truth 1e4, two halves)

MC grids: plain axis (NM=100 fixed for all configs) and per-decade-matched axis
(`mmin_pd`: NM = 10 pts/decade, i.e. Mmin=1e4→NM=130 … 1e9→NM=80; truth NM=130).

- **JSD excess of the default (Mmin=1e7):** plain axis 4.2e-4 / 5.0e-4 / 2.9e-4 /
  2.9e-4 at z_s=0.2/1/5/10; per-decade-matched 3.8e-4 / 1.0e-3 / ~0 (floor inflated,
  see caveats) / 0.9e-4. Monotone in Mmin on both axes: m1e5 ≈ floor, m1e6 ≈
  0.5–2e-4, m1e8 ≈ 2× default, m1e9 ≈ 1.6–2.5e-3.
- **Convergence knee ≈ 1e5–1e6 M⊙.** Below ~1e6 the cons14 c(M) is extrapolation
  (model territory) — convergence is well-posed within the code's own model but the
  physics below 1e6 should not be over-read.
- **NM confound (the reason for the pd axis):** the NM=200 control at Mmin=1e4
  showed M-grid coarsening ALONE contributes JSD ~5–6e-4 at z_s ≥ 5 (1.4e-5 at
  z=0.2) — same order as the Mmin physics. At z=10, per-decade matching drops the
  default's excess 2.9e-4 → 0.9e-4, i.e. most of the plain-axis z=10 "Mmin effect"
  was M-grid resolution. NM=100 is not innocent for wide mass ranges at high z_s;
  treat (Mmin, NM) as a pair.
- **Analytic arm** (`get_sigma_background(Mmin)`, per-decade contribution by
  finite-differencing σ²_W in lnMmin): default Mmin=1e7 is missing 3.3/3.9/6.0/8.1%
  of σ²_W (z_s=0.2/1/5/10) vs Mmin=1e4; going 1e4→1e3 adds only +1.2–1.6% more.
  Local exponent α(M)=dln(dσ²_W/dlnM)/dlnM ≈ 0.02–0.27 — convergent but shallow
  integral, so the M→0 residual is not negligible in σ²_W terms (few % more);
  it simply doesn't matter for P(lnμ) at current accuracy. NM=100→400 moves σ_W
  by ≤0.05% (grid-coarsening in the σ_W INTEGRAL is negligible; the NM effect
  above enters through the explicit-halo NFW table interpolation instead).
- **Fixed-⟨N⟩ coupling:** κ_thr_eff rises as Mmin falls (z=1: 1.28e-4 → 1.37e-4
  at Mmin=1e4) — included by design (production rule).

## Study 2 — Nz (default 100; truth 400, two halves)

- **Quadrature arm** (no MC): κ_thr(Nz=100) is 2.7–4.0% below its Nz→∞ value;
  σ_W(Nz=100) is −1.7% at frozen κ_thr, ~−3% with the rule's own κ_thr(Nz).
  ~First-order convergence (right-edge shell rule); both saturate by Nz≈400–800
  (root-finder tolerance visible as pairwise-equal values). At Nz=400 residual
  σ_W error ≈ 0.33% → JSD ~1e-5 ≪ floor, justifying truth=400.
- **MC arm:** default Nz=100 excess = 1.2e-4 / 0.6e-4 / 0.5e-4 / 1.1e-4 at
  z_s=0.2/1/5/10 — at the floor everywhere. Nz=50 costs 1.1e-3 at z=0.2 (but only
  ~2e-4 at z≥1); Nz=25 costs 3.9e-3 at z=0.2 — the low-z stress case as predicted
  (z_s=0.2 uses only ~11 of 25 bins). Verdict: **Nz=100 converged; do not lower
  below ~50 if low-z sources matter.**

## Tails (the sensitive corner, echoing the κ_thr study)

q99.9(μ) at z_s=10, 240k samples, 95% order-statistic CIs:

| config | q99.9 | κ_thr_eff |
|:--|--:|--:|
| mmin truth (Mmin=1e4) | 18.8 [16.7, 20.5] | 1.45e-3 |
| nz truth (Nz=400) | 19.0 [16.6, 22.2] | 1.41e-3 |
| joint default | 13.3–13.6 [12.3, 15.8] | 1.37e-3 |

The default undershoots the z=10 q99.9 by ~25–30% vs either refinement (CIs
disjoint or nearly so), and by ~10% at z=1 vs the Mmin truth (CIs disjoint). At
z=0.2 tails are identical. Note both refinements — physically unrelated knobs —
give the SAME heavier tail, and their κ_thr_eff is 3–6% higher than the default's
via the fixed-⟨N⟩ rule; in the κ_thr study the z=10 tail also tracked the split
point (fixed-⟨N⟩ q99.9 = 1.41× the flat-3e-5 truth). So the far-tail sensitivity
is plausibly dominated by the explicit/Gaussian split position, not by Mmin/Nz
physics per se. Consistent with the standing recommendation: if the z_s≳5 far
tail matters, use flat κ_thr=1e-4 or hybrid max(κ_N=100, 1e-4) there.

## Subhalo spot-checks (model 3, factor 1e-5, z_s=1)

- **Mmin:** default {Mmin=1e7} vs truth {Mmin=1e6, m_floor=1e6}: excess
  9.7e-5 — at the floor (1.09e-4). The same 1e6→1e7 step with subhalos OFF
  costs ~3e-4, so subhalos do NOT amplify the Mmin sensitivity.
- **m_floor separated (post-fix rerun):** {Mmin=1e6, m_floor=1e7} vs the same
  truth: excess 4.3e-5 — at the floor. Lowering the subhalo mass-budget floor
  1e7→1e6 changes nothing measurable at z_s=1, consistent with the Wsub
  derivation's "sub-1e7 variance 0.4%" estimate.
- **BUG FOUND AND FIXED (2026-07-12): grid-edge OOB in `interpolateNFWMass`
  (`cpp/subhalo.cpp:31`).** Presented as a deterministic segfault for
  `subhalo_model=3` with `Mmin=1e6` + default `m_floor=1e7`, but the root
  cause is a knife-edge float rounding, not the knob combination per se: the
  Wsub ψ-grid's top point lands a few ULP below `Mlist[NM-1]`, the
  `m >= Mlist[NM-1]` guard misses, `val` rounds to exactly NM−1, `jm = NM` →
  one-past-end read of `NFWlist[jz]` → garbage vector header → null deref
  (lldb-confirmed on a RelWithDebInfo build: crash at subhalo.cpp:31 with
  m = 9.99999999999972e16). Whether the rounding falls on the bad side depends
  on `log(m_floor/M)`, which is why {1e6,1e6} happened to work. Fix: clamp
  `jm` to NM−1 — only activates in previously-OOB territory, default path
  bit-identical (11/11 tests incl. bitwise re-verified after rebuild). All
  five (Mmin, m_floor) combos incl. Mmin>m_floor now run clean. Completed
  runs are unaffected (the OOB bin is the M≈1e17 host, effectively never an
  explicit host; table build is deterministic — it either crashed instantly
  or took the guard).
  Side effect worth recording: inside mp.Pool this segfault presented as a
  silent infinite hang (worker dies, task lost, pool respawns idle workers) —
  `convergence_scan.py` therefore now uses plain subprocess shards
  (2026-07-12), which surface worker crashes as nonzero exit codes.
- **Nz:** sub arm (Nz=100 vs truth 400) — see report table (nz_convergence).

## ACE tie-in — Mmin CANNOT explain the ACE−Vaskonen gap

Framing fixed by the paper (verified against the PDF, Table 1 + §2.1–2.2 of
Türker+2025): ACE training data are DMO N-body sims, box 250 Mpc/h, 4×640³
particles, particle mass 4.1 Ωm×10⁹ M⊙/h ≈ 1.8e9 M⊙, ~30 lens planes (one per
snapshot), 12 kpc/h projection grid, z ∈ [0.2, 6], and **no halo mass floor** —
all particles are projected, so ACE misses no sub-Mmin variance by construction.

Sweep (`scripts/convergence/ace_gap_vs_mmin.py`, 4 seeds × 400k raw-κ samples
per config, full model ON, central moments after clipping κ to ACE's own
support — raw unclipped moments are single-ray garbage at low Mmin, e.g.
⟨κ²⟩ = 0.38±0.63 at Mmin=1e4/z=1 from κ≳100 rays; clipped seed scatter is
±few e-6):

- **⟨κ²⟩: wrong sign for a missing-mass explanation.** Vaskonen sits ABOVE
  ACE at every Mmin (z=1: +36% at Mmin=1e9 → +42% at 1e4, default +39%;
  z=5: +1% → +13%, default +7%), and lowering Mmin adds variance, moving
  AWAY from ACE. Sub-Mmin halos add only ~6% (z=1) over five decades.
- **⟨κ³⟩ is mixed:** z=1 same wrong sign (+27→+37%); z=5 Vaskonen is BELOW
  ACE (−46% at default) and lowering Mmin closes part of it (−27% at 1e4) —
  but nowhere near fully.
- Verdict: the ACE−Vaskonen gap is dominated by model-level differences
  (ACE's N-body+grid smoothing / PCA-XGBoost pipeline vs the halo model),
  not by the halo model's mass floor. Consistent with the §4 gate conclusion.

## Caveats / gotchas carried forward

- ~~Raw σ(lnμ) and untrimmed ⟨1/μ⟩ are corrupted by rare κ>1 demagnified rays
  (mmin_pd z=5 truth half had ⟨1/μ⟩=5.3 from a single lnμ≈−14 ray) — the JSD
  histogram is quantile-clipped and immune, but two mmin_pd groups (z=1, z=5)
  have visibly inflated half-split floors from sparse-tail-bin noise; their
  floor-subtracted excesses are correspondingly less precise.~~
  **SUPERSEDED (2026-07-13, see Addendum below):** the inflated mmin_pd floors
  are NOT sparse-tail-bin noise — they are whole-batch lnμ shifts from the
  batch-mean κ compensation in `sample_lnmu` (one monster ray shifts its whole
  batch by −2κ/n); the ⟨1/μ⟩=5.3 half is the same mechanism. The JSD histogram
  is NOT immune to this (the shift moves the body, not just the tail).
- The `finish()` shard-concatenation bug (parts.pop in a comprehension) was
  fixed in BOTH `convergence_scan.py` and the parent
  `scripts/subhalo_gate/kappathr_subhalo_jsd.py` (2026-07-12); the latter's
  cached study results are unaffected (bug only fired on fresh sampling).
- q99.99 numbers at 240k samples rest on ~24 events — direction-only.
- Seed namespaces: mmin=3e8, nz=4e8, mmin_pd=5e8, ACE sweep=7e8 blocks —
  disjoint from each other and from the κ_thr study.

---

## ADDENDUM (2026-07-13) — batch-mean κ anchor bug; which numbers above are provisional

A permutation-null calibration of the JSD floors
(`data/results/floor_permutation_null/report.md`) uncovered a real engine bug:
`cpp/lensing.cpp::sample_lnmu` (646–663) enforces ⟨κ⟩=0 with the EMPIRICAL
batch mean, so a single κ≫1 ray shifts its entire batch by −2κ/n. 7/9 flagged
shards reproduce deterministically; predicted −2κ_max/n matches the observed
shard shift to ~10% in all cases across two decades
(`plots/batch_anchor_diagonal.png`). Realizations within one sampler call are
therefore weakly coupled O(κ_max/n) — material mainly for refined-grid truth
configs run at 15–30k/batch. Block-aware (shard-level) recalibration and a
cross-study exposure screen: `data/results/shard_screen/report.md`.

**Un-shift repair (2026-07-13, `scripts/convergence/unshift_repair.py` →
`data/results/unshift_repair/report.md`):** rather than excluding flagged
shards (over-corrects), EVERY len//8 shard of every cached array was un-shifted
by its own monster-ray anchor estimate δ = −2·Σ(κ≥3)/n (rays dropped, batch
shifted back — first-order emulation of the robust-anchor fix on the same RNG
stream); floors, JSDs, excesses and block nulls recomputed on original edges.
Unrepaired recomputation reproduces summary.json exactly (protocol validated);
232 shards carried detectable shifts (most at the −2e-4…−1e-3 level — the
coupling is pervasive but tiny; only the handful of monster shards mattered).

Repair-corrected numbers for THIS note (verdicts unchanged; margins GROW):

- **mmin_pd floors normalize**: z=1 6.75e-4 → 1.94e-4, z=5 2.20e-3 → 2.95e-4
  (block pctls now 3–91 = typical). Default (m1e7) excess: **pd z=1 1.0e-3 →
  2.2e-4**; pd z=5 resolvable and ~0 at default; both pd ladders now clean and
  monotone (z=1: 2.9e-5 / 9.7e-5 / 2.2e-4 / 5.4e-4 / 7.4e-4 for m1e5…m1e9).
- **Worst-case Mmin default excess is now the PLAIN axis z=1: 5.0e-4 →
  3.9e-4** (z=0.2: 4.2e-4 → 3.4e-4; z=5: 2.9e-4 → 2.8e-4; z=10: 2.9e-4 →
  2.5e-4). The "worst case ≈ 1e-3" in the verdict paragraph is superseded.
- **NM confound confirmed LARGER**: mmin z=10 nm200ctl 5.8e-4 → **8.3e-4**
  (~40% above the published value; z=5 nm200ctl 4.9e-4 → 4.5e-4). (Mmin, NM)
  pairing lesson unchanged and slightly stronger at z=10.
- **Nz conclusions untouched**: nz100 excess after repair 6.8e-5 / 5.9e-5 /
  4.1e-5 / 8.4e-5 (z=0.2/1/5/10) — at/below floor; Nz=25/50 low-z costs stand.
- Subhalo spot-checks move within noise (z1_sub m1e7: 9.7e-5 → 6.3e-5).
- The κ_thr and subhalo_factor decision studies screened CLEAN — those default
  decisions rest on uncontaminated data.

Repair caveats: rays with 1<κ<3 are unidentifiable from lnμ (residual shift
≤6/n per such ray); the un-shift is exact only in the linear lnμ≈−2κ bulk
regime. Post-fix regeneration of the ~10 monster shards remains the definitive
confirmation. Fix decision pending (robust mean excluding κ>1 recommended;
analytic mean = the only exactly-independent option).
