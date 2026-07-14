# Oral-exam prep — numerical-knob audit of the MC engine (2026-07-13)

Whiteboard-first prep for the supervisor meeting. Per topic: the claim, how to
explain it on the board, the numbers to quote, plots to have open, and the
questions to expect. Sources: `docs/convergence_mmin_nz_note.md`,
`docs/convergence_onepager.md`, `docs/sigmakappaw_measure_note.md`,
`data/results/{kappathr_subhalo_jsd,subhalo_factor_jsd}/report.md`, CLAUDE.md.

---

## 0. The common protocol (say this ONCE, up front — it carries all five studies)

**The question every study answers:** each knob is a *numerical* approximation
inside the MC engine, not physics. So the test is always: define a refined-limit
**truth**, measure how far the production **default** is from it in the observable
that matters — P(lnμ) — and compare that error to the accuracy budget.

**Whiteboard:**

1. Metric: JSD(P,Q) = ½KL(P‖M) + ½KL(Q‖M), M = (P+Q)/2. Symmetric, bounded by
   ln 2 ≈ 0.693 nats, no zero-bin blowups (unlike KL). Computed on histograms of
   lnμ, 120 shared bins, quantile-clipped range (immune to single rare κ>1 rays).
2. **Finite-sample floor:** two runs of the *same* config at N samples have
   JSD > 0. Estimate it by splitting the truth run into two independent seed
   halves → JSD_half, then rescale to the actual pair sizes by c·(1/N₁+1/N₂)
   (floor scales ∝ 1/N; for B bins the expectation is ~(B−1)/4N per comparison).
   Report **excess = max(JSD − floor, 0)**. *This matters:* early κ_thr numbers
   (3.3–3.5e-4 at z=1) turned out to be AT the floor — i.e. "converged", not "an
   error of 3.5e-4". Floors must be quoted with every number.
3. **Budget:** the production emulator's own median KL vs the sim is **7.3e-3**.
   Any knob error ≪ that is invisible downstream at O(1000) events.
4. Scale of runs: 240k realizations per config, z_s ∈ {0.2, 1, 5, 10} (κ_thr and
   factor studies: z ∈ {1, 5 or 10}), disjoint seed namespaces per study.

**The one-line summary of everything:** *five knobs audited with one protocol;
all five defaults stand; the only sensitive corner anywhere is the z_s ≳ 5
high-μ tail, and it traces back to a single cause — the explicit/Gaussian split
position.*

| Knob | Default | Truth | Default's excess | Verdict |
|:--|:--|:--|:--|:--|
| κ_thr rule | fixed-⟨N⟩=100 | flat 3e-5, →0 limit | ~0 at z=1; 1.1e-3 at z=10 | KEEP |
| subhalo_factor | **1e-2** (flipped 2026-07-12) | brute clumps | ≤3e-5 (at floor) | KEEP (new) |
| eps_floor | 0.001 (absolute, ×κ_thr_default) | ε→0 | truncation ∝ ε; 0.1% of K2 | KEEP (post-fix) |
| Mmin | 1e7 M⊙ | 1e4 | 1e-4…1e-3 (worst z=1) | KEEP at current budget |
| Nz | 100 | 400 | ≤1.2e-4 (at floor) | KEEP — converged |

⚠️ **NEW FINDING TO PRESENT (2026-07-13): the batch-mean κ compensation bug.**
The full detective story, in presentation order (this is your strongest
material — a statistical audit that ended in a source-code diagnosis):

1. A sample-level permutation calibration of the JSD floors flagged the
   original truth splits as extreme — later understood to be the WRONG test
   (see 6), but it pointed at the right place.
2. Per-shard scans localized it: whole shards body-shifted negative, up to
   −63σ.
3. Deterministic regeneration on the Mac: 7/9 REPRODUCE with the current
   build — not stale cache, not corruption.
4. Data forensics: every anomalous shard = (sibling distribution) + (uniform
   lnμ shift) + (one monster ray). Predicted shift −2κ_monster/n from the
   shard's own most extreme ray matches observed in ALL cases across TWO
   DECADES (z5 truthB: κ≈1014 ray ⇒ predicted −0.135, observed −0.129);
   `plots/batch_anchor_diagonal.png` is the one-figure proof.
5. Source located: `cpp/lensing.cpp::sample_lnmu` (646–663) anchors ⟨κ⟩=0 by
   subtracting the EMPIRICAL batch mean κ — one κ≫1 ray (outside
   weak-lensing validity anyway) shifts the entire batch.
6. Statistical self-correction: batch coupling makes the SHARD the
   exchangeable unit, so the initial sample-level permutation p-values were
   anti-conservative. Exact block enumeration (all C(16,8) shard splits) puts
   the original splits at pctl 4–90 — typical. The diagnosis rests on (3)+(4),
   not on the p-values.

Exposure verdict (`data/results/shard_screen/report.md`): the κ_thr and
subhalo_factor decision studies are CLEAN (no detectable anchor shards; block
floor ≈ sample floor) — both default decisions stand on clean data. Nz is
essentially clean. The mmin plain study is mostly clean but the z≥5 NM
confound number (nm200ctl z=10, shift −0.035) is the single most exposed
published value; a shard of the PRODUCTION-DEFAULT config (mmin z5 m1e7,
−0.004) is mildly touched, showing the mechanism isn't confined to refined
configs. The mmin_pd group has LOW RESOLUTION (block-null floors 8.3e-4 /
2.4e-3 at z=1/5) — quote its numbers as order-of-magnitude until the anchor
is fixed and ~10 shards regenerated.

**UN-SHIFT REPAIR (same day, `scripts/convergence/unshift_repair.py` →
`data/results/unshift_repair/report.md`) — quote THESE numbers:** every shard
un-shifted by its own monster estimate δ=−2Σ(κ≥3)/n (first-order emulation of
the robust fix on the same RNG stream; unrepaired recomputation reproduces
summary.json exactly, so the protocol is validated). mmin_pd floors normalize
(z=1: 6.75e-4→1.94e-4; z=5: 2.20e-3→2.95e-4; block pctls typical) — the pd
group is no longer order-of-magnitude. Corrected numbers: **worst Mmin default
excess 3.9e-4 (plain axis z=1)**; pd z=1 default 1.0e-3 → **2.2e-4** (its old
worst-case status was floor contamination, and the pd ladder is now cleanly
monotone); **NM confound z=10 = 8.3e-4** (confirmed ~40% larger — now the
largest single number in the Mmin/Nz studies); Nz numbers untouched. Verdicts
unchanged, margins GROW: worst knob error ~19× under the 7.3e-3 budget.
Post-fix regen of the ~10 monster shards = confirmation only.

Implications to state: realizations within one sampler call are weakly
coupled O(κ_max/n); this quietly explains BOTH old caveats (the "sparse-tail
inflated floors" and the ⟨1/μ⟩=5.3 truth half). Proposed fix (supervisor
decision): robust mean-κ anchor excluding κ>1 rays (minimal, physically
motivated); alternatives: analytic mean, or larger batches. Be precise on the
board: the per-batch lnμ shift is −2κ_monster/n (first order in 1/n), so larger
batches suppress the shift ∝ 1/n and the JSD-level damage ∝ (κ/n)² per affected
batch; only the analytic mean makes realizations exactly independent — any
empirical anchor (incl. the robust one) retains O(κ_max,kept/n) coupling, the
robust cut just bounds the kept κ_max at 1.
Validation plan post-fix: shard means CLT-consistent, truth halves typical in
the block null, insensitivity to κ_cut ∈ {0.5, 1, 2}.
Chain of evidence: `data/results/floor_permutation_null/report.md`.
Downstream exposure to own proactively: the production flux calibration
(`fit_flux_target.py`) fits the sim's trimmed ⟨1/μ⟩ from `sample_lnmu` draws
(`cache/flux_grid.npz`) — the same anchor bias mode applies if any flux-grid
batch caught a monster ray. Likely immaterial (trimmed target, large batches),
but unchecked; cheap audit = κ_max (lnμ_min) per flux-grid config, require
2κ_max/n ≪ trim tolerance.

⚠️ **Staleness warning:** `docs/convergence_onepager.md` still lists
subhalo_factor default = 1e-5 and "PDF-level retune still open". That's stale:
the PDF-level acceptance ran 2026-07-12, factor 1e-2 passed, and the default was
flipped to 1e-2 the same day. If the supervisor has read the onepager, correct
this proactively.

---

## 1. κ_thr — the explicit-halo / Gaussian-background split

### Claim
The default threshold rule is (back to) **fixed-⟨N⟩ = 100**: at each z_s, κ_thr
is set so the expected number of explicit halos along the LOS is 100. Chosen for
predictability (constant, bounded per-LOS cost) and continuity with Vaskonen's
convention. Its known cost — a z-correlated JSD excess ~1.2e-3 at z_s=10 — is
0.16× the emulator's KL budget and 0.17% of the ln 2 maximum: immaterial.

### Whiteboard explanation
Draw the LOS: κ_total = Σ(explicit halos with κ_peak > κ_thr, sampled
individually) + Gaussian background N(μ_W, σ²_W) for everything below threshold
(+ bias/filament layers). κ_thr is purely a *numerical* split between the two
representations — truth is the κ_thr → 0 limit where everything is explicit.

Two candidate rules: fixed-⟨N⟩ (κ_thr floats with z_s: 1.28e-4 at z=1 →
1.37e-3 at z=10) vs flat κ_thr (z-independent).

The cost/accuracy tension, on the board as two curves vs κ_thr:
- Cost: ⟨N⟩ ∝ 1/κ_thr (measured slope 1.05 at z=1) — **diverges, no plateau**.
- Accuracy: JSD vs truth **plateaus below κ_thr ≈ 1e-4** (the knee).
So running below ~1e-4 is pure cost for zero accuracy — the question is only
where each rule sits relative to the knee at each z_s.

### Numbers to quote (subhalo-ON arm, model 3; truth = flat 3e-5, 2×120k halves)
Floor-subtracted JSD excess:

| rule | z=1 | z=10 |
|:--|--:|--:|
| fixed-⟨N⟩ (default) | 0 (at floor) | 7.6e-4 |
| flat 1e-4 | 2e-5 | 4e-5 |
| flat 1e-3 | 1.9e-3 | 3.5e-4 |

Subhalo-off arm: 5.0e-5 / 1.1e-3 (fixedN), so subhalos do NOT widen the gap.
Mechanism of the z=10 drift: the fixed-⟨N⟩ rule inflates κ_thr with z_s
(1.28e-4 → 1.37e-3), i.e. the split is coarsest exactly where the lensing signal
is strongest. Flat 1e-4 is z-uniform. Flat 1e-3 is the mirror image: bad at z=1
(only ⟨N⟩≈11 explicit halos), decent at z=10.

**Tail caveat (the honest weakness):** z=10, subhalo-on, q99.9(μ):
fixed-⟨N⟩ = 1.41× truth (95% CI [1.09, 1.75]) vs flat 1e-4 = 1.04 [0.80, 1.28].
Standing recommendation: if the z_s ≳ 5 far tail matters (strong-lensing-ish
events), use flat 1e-4 or hybrid max(κ_⟨N⟩=100, 1e-4) there. Global σ, ⟨1/μ⟩,
q99 shifts stay ≲1%/0.5%/4%.

### Anticipated questions
- **"Why keep a rule with a known z-correlated error instead of flat 1e-4?"**
  Predictability: 100 explicit halos at every z_s = bounded, constant cost. Flat
  1e-4 costs ⟨N⟩=1849 at z=10 (~18× more halo work) for an accuracy gain that is
  invisible under the emulator's 7.3e-3 KL. Plus continuity with Vaskonen. The
  error is documented and bounded; if we ever scale to ≳4000 events, enforce
  ⟨1/μ⟩=1 (already flagged) or switch the high-z tail to flat 1e-4.
- **"Isn't a z-correlated error dangerous for cosmology posteriors?"** In
  principle yes — that's why it's flagged. In practice 1.2e-3 JSD vs the 7.3e-3
  emulator KL means the emulator's own error dominates by ~6×; posterior-recovery
  mocks (16×1000 events) are zero-consistent in h post-calibration.
- **"How do you know your truth (3e-5) is converged?"** The JSD plateau: nothing
  moves below the ~1e-4 knee, and truth sits 3× below the knee; the two truth
  halves agree at the predicted 1/N floor.
- **"Didn't you change this default twice?"** Yes — flat 1e-3 was default
  2026-07-09→10, then reverted deliberately. **Training data generated in that
  window used flat 1e-3 and must be regenerated or flagged.** Own this before
  being asked.
- **Gotchas if probed deeper:** the BIAS layer is κ_thr-coupled (σ_b uses tube
  radius rmax(κ_thr)) — the full model breaks for κ_thr ≳ 3e-3, so the flat rule
  can't be pushed high anyway. Sub-threshold filaments lack weak compensation
  (−2% at z=1, pre-existing, rule-independent). Single-seed σ_κ comparisons are
  tail-noise (±20% per 5e4 draws) — always ensembles.

### Plots
`plots/kappathr_decision.png` (the cost-vs-accuracy decision figure),
`plots/kappathr_pdf_grid_1em04_loglin.png` (PDF overlays),
`plots/kappathr_subhalo_jsd.png` (subhalo-on re-test),
table: `data/results/kappathr_subhalo_jsd/report.md`.

---

## 2. subhalo_factor — the resolved/unresolved clump split (Wsub, model 3)

### Claim
Subhalo substructure is handled by an exact-in-mean-and-variance split
(`subhalo_model=3`): per host, κ = reduced smooth host + explicit clumps above
κ_thr,clump = subhalo_factor · κ_thr,host + a deterministic unresolved mean
profile μ_unres(y) + Gaussian N(0, σ²_unres(y)). This makes subhalo_factor a
**pure performance knob**. PDF-level acceptance (2026-07-12): factor 1e-2 is
indistinguishable from brute truth AND from factor 1e-5, at ~10× less cost —
so the default was flipped 1e-5 → **1e-2**.

### Whiteboard explanation
1. Brute = every clump above m_floor sampled explicitly — the truth, but slow
   (158 s vs 8–16 s per 3e4 realizations at variance level).
2. **Poisson restriction theorem** (the core argument): a Poisson clump
   population restricted to κ > threshold stays Poisson; the removed
   (unresolved) part contributes exactly its mean profile and its Campbell
   variance. So splitting at ANY subhalo_factor is exact in mean and variance —
   verified numerically to 6 digits over factors 1e-5…1.
3. **The key design finding:** a zero-mean Gaussian alone recovers almost
   nothing (variance deficit 0.844 → 0.851 at factor 1e-3). The deficit is
   dominated by the *mean-profile mismatch* — so the deterministic μ_unres(y)
   table is REQUIRED, and the host reduction must use the full f_s,b (not the
   resolved-only fraction). That's what distinguishes model 3 from model 1
   (which collapses to −0.17 excess variance at factor 1).
4. What the factor still controls: which moments beyond variance are dropped
   into the Gaussian — dropped-skewness (c₃) share is 0.2% / 0.7% / 3% at
   factor 1e-3 / 1e-2 / 1e-1. Hence "performance/Gaussianity knob".

### Numbers to quote (PDF acceptance, `data/results/subhalo_factor_jsd/report.md`)
Brute truth (2×120k seed halves), 240k per candidate, z_s ∈ {1, 5}, fixed-⟨N⟩:

- factor 1e-2: JSD excess **3.0e-5 (z=1), 2.5e-5 (z=5)** — at the floor
  (~1.1e-4), same as factor 1e-5 (2.8e-5, 3.1e-5). q99.9/q99.99 CIs overlap
  truth. Even factor 1e-1 passes globally (mild ⟨1/μ⟩ drift at z=5: 1.0118 vs
  truth 1.0041).
- **Negative control that proves the test has teeth:** model 1 @ 1e-2 (same
  clump split, NO Wsub term) shows real excess 1.7e-4 at z=5 — the test resolves
  the Wsub term's PDF-level contribution.
- Cost: s/1e4 realizations, factor 1e-5 → 1e-2: z=1 45.9 → 4.7, z=5 37.6 → 5.9
  (subhalo-off baseline 3.6) — near subhalo-off cost, ~10× speedup.
- Known factor-INDEPENDENT residual: ALL split arms (incl. 1e-5 and model 1) sit
  ~2.3% below brute q99(μ) at z=5, CIs disjoint — a split-vs-brute property, not
  a factor effect; global JSD unaffected.

### Anticipated questions
- **"If the split is exact in mean and variance, why test the PDF at all?"**
  Because the theorem says nothing about higher moments/shape — the unresolved
  part is Gaussianized. The PDF test is precisely the check that the dropped
  non-Gaussianity is invisible; the model-1 contrast shows the test could detect
  a real defect if there were one.
- **"Why 1e-2 and not 1e-1, if 1e-1 also passes?"** Margin: 1e-1 shows a real
  ⟨1/μ⟩/σ drift at z=5 and 3% dropped-c₃; 1e-2 is statistically identical to
  1e-5 with almost all of the speedup already banked.
- **"What about the 2.3% q99 deficit?"** Factor-independent, present even at
  1e-5 — a property of the split construction vs brute, both truth halves agree.
  Flagged, not factor-tunable; global JSD unaffected.
- **"Is the theorem verified or assumed?"** Verified numerically to 6 digits
  (mean/variance bookkeeping, factors 1e-5…1), plus the multi-seed clipped-core
  variance validation at z=1 (model 3 flat at brute ±5% while model 1 collapses).
- **Reproducibility trap:** subhalo-ON runs made before 2026-07-12 used
  factor 1e-5 — pass `subhalo_factor=1e-5` explicitly to reproduce. ML training
  data regeneration with subhalos is still pending.
- **If asked about wf:** the emulator's Giocoli α_f uses exp(−0.25) (f=½ in
  α_f = 0.815·e^{−2f³}/f^0.707); halos still has the exp(−1) misread ⇒ f_s
  overestimated ~22–26% there. Fixed here, not yet ported (halos is
  user-controlled).

### Plots
`plots/subhalo_factor_jsd.png` (acceptance), `plots/wsub_partition_variance_proof.png`
(exactness), `plots/sigma_k_vs_subhalo_factor_paired.png` (variance-level, model 3
flat vs model 1 collapse), table: `data/results/subhalo_factor_jsd/report.md`.

---

## 3. eps_floor — the σ_W radial integration floor (+ the σ_W measure fix)

This topic is really two intertwined 2026-07-08 items: (a) the sigmakappaW
**measure bug fix** (×2 + Campbell), (b) **eps_floor** parametrized with a
floor-consistent internal caller. Present (a) first — it's the stronger story.

### Claim (a): σ²_W was low by ×2 (log-annulus measure) plus ~4–7% (wrong
variance formula); both fixed, supervisor-approved, net σ²_W ≈ 2.10× original —
and it changes nothing scientific because σ²_W is ≤0.4% of Var(κ_total).

### Whiteboard explanation (a)
σ_W is the Gaussian background's width: the variance of the summed κ of all
sub-threshold halos, computed as a radial integral outward from rmax(κ_thr).

1. **The measure bug:** the annulus area element in log radius is
   d(πr²)/dlnr = **2**πr², but the loop used πr². Cleanest proof — the code
   against itself: the same code counts halos two ways, a disk formula
   (calibrates κ_thr via ⟨N⟩=100) and the annulus loop; summing the loop over
   [R₁,R₂] gives π(R₂²−R₁²)/**2** — half the disk difference. Measured ratio
   0.495 at every z_s (0.5 minus ~1% left-Riemann error).
2. **The subtraction bug:** the code returned √(κ₂ − κ₁²/Nh) — the fixed-N iid
   formula. For a Poisson count, **Campbell's theorem** applies:
   Var = ∫ n(κ) κ² dn, NO subtraction (count fluctuations restore the ⟨κ⟩²
   piece). Another ~4–7% low.
3. **Brute-force MC referee:** sample the weak-band halos explicitly with the
   production code's own conventions (Poisson counts from the disk formula,
   positions uniform in area, same NFW kernels): Var_MC / σ²_W(code) =
   **1.98–2.14** across z_s = 0.5–5; Var_MC / Campbell-2π = 0.94–1.02. Case
   closed.
4. **Why nothing published moves:** σ²_W share of Var(κ_total) is 0.03% (z=0.5)
   → 0.40% (z=5); the ×2.10 fix shifts total Var(κ) by ≤ +0.5%.

### Claim (b): eps_floor
The radial loop stops outward where κ drops below eps_floor·κ_thr. Default
eps_floor = 1e-3: keeps **99.90% of K2**, and the truncation error is **∝ ε**
(proved numerically, `playground/k2_vs_floor.cpp`) — so convergence is linear
and controlled; ε→0 truth is trivially extrapolable.

**The subtle fix worth telling:** the internal caller now holds the ABSOLUTE
floor at 0.001·κ_thr_default. Previously the floor was relative to the active
κ_thr, so `custom_kappathr` sweeps dragged the integration stop with the
threshold and σ_W collapsed at high κ_thr — an artifact of the knob coupling,
not physics. Default path verified bit-identical after the change (seed 12345).

### End-to-end validation (the payoff slide/plot)
Variance partition: measured total σ = √(σ²_explicit + σ²_W) stays flat at the
full-Campbell value 0.02728 as κ_thr sweeps (z=1, halo-only), and the
total/plateau ratio is 0.991–1.001 across z_s = 0.2–10. I.e. the explicit and
Gaussian parts trade off exactly as they should — the split conserves variance.
Bonus theory: σ_full(z_s) ∝ z^{3/2} at low z (coeff 0.0443, derived), saturating
at σ_∞ = 0.154 (`docs/sigma_full_analytic_note.md`).

### Anticipated questions
- **"If σ_W is 0.1% of the variance, why did you spend time on it?"**
  Correctness compounds: the same integral pattern was about to be reused for
  the subhalo Wsub term (σ²_unres) — the fix was found before it propagated.
  Also the variance-partition validation only closes with the fixed σ_W.
- **"How sure are you it's exactly ×2 and not a convention difference?"** The
  code contradicts itself: its own disk count vs its own annulus loop, same
  domain, same prefactor, ratio 0.495. No external formula involved. Plus the
  MC referee at 2.04–2.14.
- **"Why no κ₁²/Nh subtraction?"** That's the fixed-N iid variance. The count
  is Poisson; Campbell's theorem for a Poisson point process gives ∫n κ² with
  no mean subtraction. The MC confirms (subtracting would re-introduce the
  4–7% deficit).
- **"Is halos fixed too?"** Patch applied there at user request 2026-07-08 but
  left uncommitted for the user's review (halos is under user control).
- **"Why ε=1e-3 and not smaller?"** Error ∝ ε and already 0.1% of K2 — an
  integral-level 1e-3 relative error on a term that is itself ≤0.4% of
  Var(κ_total). Going lower buys nothing measurable.

### Plots
`plots/sigmakappaw_variance_proof.png` (MC vs code vs Campbell),
`plots/k2_vs_floor.png` (truncation ∝ ε), `plots/sigma_partition_vs_kthr.png` +
`plots/sigma_partition_zs_study.png` (variance partition flat),
`plots/sigma_full_vs_zs.png` (z^{3/2} law). Note: `docs/sigmakappaw_measure_note.md`
has every formula if pressed.

---

## 4. Mmin — the HMF lower mass limit (default 1e7 M⊙)

### Claim
The default is **NOT fully converged but immaterial**: JSD excess 1e-4…1e-3
(worst case z=1 ≈ 1.0e-3 on the per-decade-matched axis), knee at Mmin ≈
1e5–1e6, missing 3–8% of σ²_W — all 7–70× below the emulator's KL 7.3e-3.
KEEP; revisit only if the emulator's own KL budget drops below ~1e-3.

### Whiteboard explanation
Mmin truncates the halo mass function integral — it affects BOTH the explicit
halo population and σ²_W. Truth = Mmin 1e4 (three decades deeper), 2×120k seed
halves. Two MC axes, and this is the methodological point to make:

**The NM confound (why one axis wasn't enough):** widening the mass range at
fixed NM=100 coarsens the M-grid, and an NM=200 control at Mmin=1e4 showed grid
coarsening ALONE contributes JSD ~5–6e-4 at z_s ≥ 5 — same order as the Mmin
physics. So a second, per-decade-matched axis (NM = 10 pts/decade) was run. At
z=10 it drops the default's apparent excess 2.9e-4 → 0.9e-4: most of the
plain-axis high-z "Mmin effect" was M-grid resolution. **Lesson: (Mmin, NM)
must move together in any refinement.** (The NM effect enters via the
explicit-halo NFW table interpolation, NOT the σ_W integral — NM 100→400 moves
σ_W by ≤0.05%.)

**Analytic arm** (no MC noise): finite-difference σ²_W in lnMmin using the
code's own `get_sigma_background`. Default misses 3.3 / 3.9 / 6.0 / 8.1% of
σ²_W at z_s = 0.2/1/5/10 vs Mmin=1e4; the next three decades (1e4→1e3) add only
+1.2–1.6% more. Local exponent α ≈ 0.02–0.27 — convergent but shallow, so the
M→0 residual is a few % of σ²_W forever; it just doesn't matter for P(lnμ) at
current accuracy.

### Numbers to quote
- Default's JSD excess, plain axis: 4.2e-4 / 5.0e-4 / 2.9e-4 / 2.9e-4
  (z=0.2/1/5/10); per-decade-matched: 3.8e-4 / 1.0e-3 / ~0 (floor inflated) /
  0.9e-4. Monotone in Mmin; m1e5 ≈ floor, m1e9 ≈ 1.6–2.5e-3.
- Physics honesty: below ~1e6 M⊙ the cons14 c(M) relation is extrapolation —
  convergence is well-posed within the code's own model, but don't over-read
  the physics below 1e6.
- Fixed-⟨N⟩ coupling is included by design: κ_thr_eff rises as Mmin falls
  (1.28e-4 → 1.37e-4 at z=1) — that's the production rule doing its job.
- Subhalo spot-checks (model 3, z=1): Mmin step with subhalos ON is AT the
  floor (excess 9.7e-5 vs floor 1.09e-4) while the same step subhalo-OFF costs
  ~3e-4 → subhalos do NOT amplify Mmin sensitivity. m_floor 1e7→1e6 alone:
  4.3e-5, nothing — consistent with the Wsub "sub-1e7 variance 0.4%" estimate.

### The ACE tie-in (likely the supervisor's favorite question)
**"Could Mmin explain the ACE−Vaskonen gap?" — No, wrong sign.** ACE trains on
DMO N-body sims with NO halo mass floor (all particles projected, m_p ≈ 1.8e9
M⊙ — verified against Türker+2025 Table 1). If the gap were missing low-mass
halos, lowering Mmin would close it. Instead (support-clipped moments, 4 seeds
× 400k): Vaskonen ⟨κ²⟩ sits ABOVE ACE at every Mmin — z=1: +36% (Mmin=1e9) →
+42% (1e4), default +39% — and lowering Mmin adds variance, moving AWAY from
ACE. Sub-Mmin halos add only ~6% over five decades. Only z=5 ⟨κ³⟩ moves toward
ACE (−46% → −27%), far from closing. Conclusion: the gap is model-level (ACE's
N-body + grid smoothing + PCA-XGBoost pipeline vs the halo model), consistent
with the §4 gate verdict.
Methodological caveat to volunteer: moments are computed after clipping κ to
ACE's own support — raw unclipped moments at low Mmin are single-ray garbage
(⟨κ²⟩ = 0.38 ± 0.63 at Mmin=1e4/z=1 from κ ≳ 100 rays).

### Anticipated questions
- **"If it's not converged, why not just lower the default?"** Cost/benefit:
  the error is 7–70× under the emulator KL; lowering Mmin inflates the halo
  tables and σ_W work for invisible gain, and below 1e6 the concentration
  model is extrapolated anyway — refined "truth" becomes model territory.
- **"Is the 1e-3 worst case (z=1, pd axis) solid?"** Less precise than the
  rest: two mmin_pd groups (z=1, z=5) have inflated half-split floors from
  sparse tail bins (one lnμ ≈ −14 ray gave a truth half ⟨1/μ⟩ = 5.3) — the JSD
  itself is quantile-clipped and immune, but the floor subtraction there is
  noisier. Order of magnitude stands.
- **"What would make you revisit?"** Emulator KL budget below ~1e-3, or a
  tail-sensitive application at z≳5 (but that corner is κ_thr-dominated — §6).
- **The bug story (tell it, it's a credibility win):** the study surfaced a
  real C++ bug — a grid-edge float knife-edge in `interpolateNFWMass`
  (Wsub ψ-grid top point a few ULP below Mlist[NM−1] slips past the guard →
  one-past-end read → segfault, lldb-confirmed). One-line index clamp; default
  path bit-identical (11/11 tests incl. bitwise). Operational lesson:
  mp.Pool swallows worker segfaults as silent infinite hangs — the driver now
  uses plain subprocess shards, which surface crashes as exit codes.

### Plots
`plots/mmin_convergence.png` (regenerated 2026-07-13: left panel now extends to
the truth endpoint at 1e4 — open marker = truth's own floor; middle panel =
floor-subtracted excess with 68% bootstrap bands, dropping to 0 at the truth —
the convergence money-shot), `plots/ace_gap_vs_mmin.png`, tables:
`data/results/{mmin,mmin_pd}_convergence/report.md`. Note the dashed (pd-axis)
open endpoints sit high — that's the inflated pd floors caveat, not physics.

---

## 5. Nz — the redshift grid (default 100)

### Claim
**Nz=100 is converged.** MC JSD excess is at the floor for all z_s (1.2e-4 /
0.6e-4 / 0.5e-4 / 1.1e-4 at z=0.2/1/5/10 vs truth Nz=400). Don't go below ~50
if low-z sources matter.

### Whiteboard explanation
Nz sets the log z-grid (0.01–10.01) used for the LOS shell integrals. Two arms:

1. **Quadrature arm (no MC, fast, explains the mechanism):** κ_thr(Nz=100) is
   2.7–4.0% below its Nz→∞ value and σ_W is −1.7% (frozen κ_thr) to −3% (with
   the rule's own κ_thr(Nz)). First-order convergence in 1/Nz (right-edge shell
   rule), saturating by Nz ≈ 400–800 (root-finder tolerance shows up as
   pairwise-equal values). **The punchline: these are common shifts of the
   split point, and they cancel in P(lnμ)** — which is why…
2. **MC arm:** …the actual PDF error at Nz=100 is at the measurement floor
   everywhere. The knob moves internal quantities by percents but the
   observable by nothing.

**The predicted-and-confirmed stress case:** low z_s. At z_s=0.2 only ~11 of 25
bins lie before the source, so coarse grids bite there first: Nz=25 costs
3.9e-3 at z=0.2 (a real, above-budget error!) but only ~2e-4 at z≥1; Nz=50
costs 1.1e-3 at z=0.2. Hence the "don't go below 50" rider.

**Truth justification (expect this question):** at Nz=400 the residual σ_W
error is ≈0.33% → JSD ~1e-5, an order below the floor — so truth=400 is safely
converged for a floor-limited comparison.

### Anticipated questions
- **"κ_thr is 4% off at the default and you call it converged?"** Converged in
  the observable. The 2.7–4% κ_thr shift and the ~3% σ_W shift are the SAME
  split-point displacement seen by both representations — explicit and
  Gaussian parts move together and P(lnμ) is invariant (variance partition,
  §3). The MC arm proves it: excess at floor at all four z_s.
- **"Why is low z_s the stress case when lensing is weakest there?"** Not
  signal strength — grid support: at z_s=0.2 the integrand has ~11 shells to
  live on at Nz=25. It's a quadrature resolution problem, predicted before the
  run and confirmed (3.9e-3).
- **"Subhalo interaction?"** Spot-check at floor (4.6e-5, model 3, z=1).
- **"Why first-order convergence?"** Right-edge (rectangle) shell rule — O(1/Nz)
  by construction. Saturation past ~400 is the root-finder tolerance, not
  higher-order behavior.

### Plots
`plots/nz_convergence.png` (regenerated 2026-07-13: same treatment — truth
endpoint at Nz=400 + bootstrap-excess middle panel; excess is at/near 0 from
Nz=100 up, which IS the convergence claim), table:
`data/results/nz_convergence/report.md`.

---

## 6. The cross-study tail pattern (have this ready — it unifies everything)

At z_s=10, the default undershoots q99.9(μ) by ~25–30% vs BOTH the Mmin truth
(18.8 [16.7, 20.5]) and the Nz truth (19.0 [16.6, 22.2]) — default 13.3–13.6
[12.3, 15.8], CIs ~disjoint. Two *physically unrelated* refinements produce the
SAME heavier tail; both raise κ_thr_eff by 3–6% via the fixed-⟨N⟩ coupling; and
in the κ_thr study the z=10 tail likewise tracked the split point (fixed-⟨N⟩
q99.9 = 1.41× truth). **Inference: the far-tail sensitivity across ALL knob
studies is one phenomenon — the explicit/Gaussian split position — not Mmin or
Nz physics.** At z_s ≤ 1 tails agree everywhere; q99.99 at 240k rests on ~24
events (direction-only). Standing recommendation, one sentence: *for z_s ≳ 5
tail-sensitive work, flat κ_thr = 1e-4 or hybrid max(κ_⟨N⟩=100, 1e-4); for
everything else the defaults are converged or immaterial.*

---

## 7. Rapid-fire facts (memorize)

- Emulator budget: median KL **7.3e-3**; post-calibration panel KL 3.4e-3.
- JSD max = ln 2 ≈ 0.693; 120 bins; floor ∝ (B−1)/4N, half-split-calibrated.
- 240k realizations/config; seed namespaces disjoint (κ_thr 6e8, mmin 3e8,
  nz 4e8, mmin_pd 5e8, ACE 7e8).
- κ_thr_eff (fixed-⟨N⟩=100): 1.28e-4 (z=1) → 1.37e-3 (z=10); ⟨N⟩ ∝ 1/κ_thr.
- subhalo_factor default 1e-2 (2026-07-12); model 3 cost ≈ 1.3–1.6× subhalo-off.
- σ²_W fix: ×2 measure + Campbell no-subtraction = ×2.10; ≤1% on Var(κ_total).
- eps_floor 1e-3 keeps 99.90% of K2; error ∝ ε.
- Mmin=1e7 misses 3.3–8.1% of σ²_W; knee 1e5–1e6; cons14 extrapolated < 1e6.
- Nz=100: κ_thr −2.7–4.0%, σ_W −1.7–3% — cancels in P(lnμ); Nz=25 → 3.9e-3 at z=0.2.
- ACE: no mass floor, m_p ≈ 1.8e9 M⊙; Vaskonen ⟨κ²⟩ +39% above ACE at default (z=1).
- Open items to own proactively: training data from the flat-1e-3 window
  (07-09→10) needs regen/flag; subhalo-ON runs pre-07-12 used factor 1e-5;
  onepager's subhalo_factor row is stale; halos still lacks the wf fix.

