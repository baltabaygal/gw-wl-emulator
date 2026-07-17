# Numerical-knob audit of the MC engine — supervisor summary (2026-07-12)

Five internal knobs of the Vaskonen-based MC engine have now been audited with a
common protocol: define a refined-limit "truth", measure the P(lnμ) error of the
production default as a Jensen–Shannon divergence with a finite-sample floor from
two independent truth halves, and compare against the production emulator's own
median KL of 7.3e-3 (the accuracy budget that matters downstream).

| Knob | Default | Truth / refinement | Default's error | Verdict |
|:--|:--|:--|:--|:--|
| κ_thr rule (explicit-halo vs Gaussian split) | fixed-⟨N⟩=100 | flat κ_thr→0 (3e-5) | JSD ≈ floor at z=1; 1.2e-3 at z=10 (z-correlated) | KEEP (predictability + Vaskonen continuity); z≳5 far tail prefers flat 1e-4 |
| subhalo_factor (resolved/unresolved clump split) | **1e-2**, model 3 (flipped from 1e-5 on 2026-07-12) | brute resolution (2×120k halves) | exact in mean/variance (Wsub theorem); PDF-level JSD excess ≤3e-5 = at floor, indistinguishable from 1e-5; ~10× cheaper (near subhalo-off cost) | KEEP new default 1e-2; PDF acceptance DONE 2026-07-12 |
| eps_floor (σ_W radial integration floor) | 1e-3·κ_thr_default | ε→0 (truncation ∝ ε, proved) | ≤1% on Var(κ_total) after the ×2 measure fix | KEEP (fix shipped 2026-07-08) |
| **Mmin (HMF lower limit)** | **1e7 M⊙** | **1e4, two seed halves** | **JSD 1e-4…1e-3 (worst z=1); missing 3–8% of σ²_W** | **KEEP at current emulator accuracy; revisit if KL budget drops below ~1e-3** |
| **Nz (z-grid / LOS shells)** | **100** | **400, two seed halves** | **JSD ≤ 1.2e-4 = at the floor, all z_s** | **KEEP — converged; don't go below ~50 if z_s≈0.2 matters** |

## The two new studies (2026-07-11/12)

**Bindings:** `Mmin/NM/Nz` exposed as trailing kwargs on all samplers and helpers
(previously hard-coded in the helpers); default path bit-identical (11/11 tests).

**Mmin.** The default is NOT fully converged — JSD excess 3–5e-4 (per-decade-
matched M grid: 0.9e-4–1.0e-3), knee at Mmin≈1e5–1e6 — but sits 7–70× under the
emulator KL. The analytic arm (finite-differenced σ²_W(Mmin) from the code
itself) shows the default misses 3.3–8.1% of background variance (z_s=0.2→10),
with a shallow convergence exponent α≈0.02–0.27; below ~1e6 M⊙ the cons14 c(M)
is extrapolated, so deeper truth is model territory anyway. **Side finding:** the
NM=100 mass grid contributes JSD of the same order at z_s≥5 (shown by an NM=200
control and confirmed by the per-decade-matched rerun) — (Mmin, NM) must move
together in any future refinement.

**Nz.** Default converged. Quadrature arm: κ_thr and σ_W are each ~2–4% below
their Nz→∞ values at Nz=100 (first-order in 1/Nz, saturating by Nz≈400–800), but
this common shift cancels in P(lnμ): MC JSD excess at Nz=100 is at the floor for
all z_s. The predicted low-z stress case is real but only bites below the
default: Nz=25 costs 3.9e-3 at z_s=0.2 (only ~11 shells before the source).

**Tails (cross-study pattern).** At z_s=10 the default undershoots q99.9(μ) by
~25–30% vs BOTH refined truths (95% CIs ~disjoint); at z_s≤1 tails agree. Since
two physically unrelated refinements heavier-tail identically, and their
κ_thr_eff is 3–6% above the default's (fixed-⟨N⟩ coupling), the far tail appears
to track the explicit/Gaussian split position — the same corner flagged in the
κ_thr study. Standing recommendation unchanged: flat κ_thr=1e-4 or
max(κ_⟨N⟩=100, 1e-4) for z_s≳5 tail-sensitive use.

**Subhalo spot-checks (model 3, factor 1e-5, z_s=1):** all at the measurement
floor — Mmin default vs {Mmin=1e6, m_floor=1e6} truth: excess 9.7e-5; the
m_floor knob alone: 4.3e-5; Nz=100 vs 400: 4.6e-5. Subhalos amplify none of
these sensitivities. **Bonus find: a real
C++ bug** — a grid-edge float knife-edge in `interpolateNFWMass` (Wsub ψ-grid
top point slips past the mass-grid guard → out-of-bounds read → segfault;
surfaced as "model 3 + Mmin<m_floor crashes"). Fixed with a one-line index
clamp; default path bit-identical (bitwise test re-passed). Operational lesson:
mp.Pool converts worker segfaults into silent infinite hangs — the convergence
driver now uses subprocess shards, which expose them as exit codes.

**ACE tie-in:** ACE trains on DMO N-body sims with NO halo mass floor (all
particles projected; particle mass ≈1.8e9 M⊙ — verified against Türker+2025
Table 1), so if the ACE−Vaskonen gap were missing low-mass halos, lowering
Mmin would close it. It does the opposite: with moments clipped to ACE's own
κ support, Vaskonen ⟨κ²⟩ is ABOVE ACE at every Mmin (z=1: +36%…+42%, default
+39%; z=5: +1%…+13%), i.e. **the gap has the wrong sign for a missing-mass
explanation — Mmin cannot close it**; it is model-level (ACE smoothing /
pipeline vs halo model). Only z=5 ⟨κ³⟩ moves toward ACE (−46%→−27% at
Mmin=1e4), still far from closing.

**No default changes from the Mmin/Nz studies; no training-data regeneration
implied.** Details: `docs/convergence_mmin_nz_note.md`; figures
`plots/{mmin,nz}_convergence.png`, `plots/ace_gap_vs_mmin.png`.

## ADDENDUM (2026-07-13) — batch-mean κ anchor bug (fix decision pending)

The floor-calibration audit found a real engine bug: `sample_lnmu` anchors
⟨κ⟩=0 with the empirical batch mean, so one κ≫1 monster ray shifts its whole
batch by −2κ/n (7/9 flagged shards reproduce deterministically; predicted vs
observed shift matches to ~10% across two decades). Consequences:

- Realizations within one sampler call are weakly coupled O(κ_max/n);
  material only when a batch catches a monster (refined-grid truths at
  15–30k/batch).
- Exposure screen (`data/results/shard_screen/report.md`): κ_thr and
  subhalo_factor decisions CLEAN; nz essentially clean. A post-hoc un-shift
  repair (every shard un-shifted by its own monster-ray estimate,
  `data/results/unshift_repair/report.md`) firms the corrections: worst Mmin
  default excess **3.9e-4** (plain z=1; pd z=1 drops 1.0e-3 → 2.2e-4, its
  floor was contaminated), z≥5 NM-confound **8.3e-4** (~40% larger than
  quoted), Nz numbers unchanged. **All verdicts unchanged — margins grow.**
- Proposed fix: robust mean-κ anchor excluding κ>1 rays (recommended;
  analytic mean is the only exactly-independent alternative); then regenerate
  ~10 affected shards. Future floors: shard-level (block) resampling; record
  κ_max per shard.
