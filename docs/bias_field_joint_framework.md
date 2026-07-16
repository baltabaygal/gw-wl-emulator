# Joint-field framework: clustered weak background from the excursion-set decomposition (2026-07-15)

Companion to `docs/bias_field_design_note.md` (the correlated-field bias-layer redesign;
this doc is its §9 in long form) and `docs/nz_bias_convergence_note.md` (the problem).
Sizing implementation: `scripts/convergence/bias_field_joint.py` →
`data/results/bias_field_joint/{sizing.npz, report.md}`.

## 1. The idea, in one paragraph

Excursion-set theory generates halos and environment from ONE Gaussian realization:
halos are first crossings, and the "background" is not matter that never collapsed —
in the M_min→0 limit every trajectory crosses, so the background is exactly the matter
locked in halos BELOW the resolution cut. Consequences: (i) the count response to the
environment is the conditional first-crossing ratio (= design-note model 2 — the
one-point projection of this picture, not an alternative to it); (ii) the environment
field is itself mass, so the weak background κ_W must contain a clustered component
driven by the SAME field that modulates the explicit-halo counts; (iii) the amplitude
of that component is fixed by mass conservation (sum rule), not by a new parameter;
(iv) the effective transverse smoothing of the background is not a chosen filter but
the bias-and-mass-weighted profile of the removed (resolved) population. The defining
consistency condition of the joint model is **split invariance**: M_min and κ_thr are
bookkeeping choices, and observables must not move when the resolved/unresolved
boundary moves (the sum rule enforces this exactly for the clustered channel).

## 2. Decomposition and sum rule

Per line of sight, one z=0 environment field δ̄(χ) (pencil-projected, §2 of the design
note), segment-averaged per z-shell. The convergence splits as

    κ = κ_explicit(counts modulated by δ̄)  +  κ_W,shot  +  κ_W,clust
    κ_W,clust = Σ_shells a_w[jz] · δ̄_jz            (deterministic given the field)

κ_W,shot keeps the current Campbell σ_W (Poisson shot noise of sub-threshold
encounters — genuinely independent, the 1-halo piece). The clustered amplitude comes
from mass bookkeeping. The mean convergence rate of ALL matter in shell jz is exact
(each halo's total projected mass is M/Σ_c, independent of profile):

    dκ̄_tot[jz] = Δχ_jz · ρ_M0 (1+z_jz)² / Σ_c(z_s, z_jz)

Bias-weighting it with the halo-locked mass integral I_b(z) = ∫ M n(M,z) b(M,z) dM / ρ̄
and subtracting the part already carried by the modulated explicit counts
(Web[jz] = Σ_cells barN·k̄·b, the bias-weighted mean explicit rate) leaves

    a_e[jz] = D(z_jz) · Web[jz]                        (count-modulation arm, existing)
    a_w[jz] = D(z_jz) · (dκ̄_tot[jz] · I_b(z_jz) − Web[jz])   (clustered weak arm, NEW)

a_w ≥ 0 is structural (the explicit clipped mass is a subset of all halo mass). The
sum rule ∫ M n b dM = ρ̄ (mass-weighted mean bias = 1) is what makes this bookkeeping
close: it fixes the background bias given the resolved population, guarantees the
large-scale power is *split*, not duplicated, between the count channel and the
convergence channel, and makes the split (κ_thr, and M_min for the clustered channel)
a no-op by construction. The code's current (pFC q=0.8, halobias q=0.75) mismatch
violates it at a measurable level (design note §5.1) — quantified in §5 below.

Variance budget with the segment covariance C of δ̄ (pencil P_1D, design note §2.2):

    S_ee = a_eᵀ C a_e     (clustering of explicit counts — the prototype's number)
    S_ww = a_wᵀ C a_w     (clustered weak background — the new object)
    S_ew = a_eᵀ C a_w     (count–background covariance; ρ_ew = S_ew/√(S_ee·S_ww) ≈ 1
                           expected, since both arms ride the same field with similar kernels)

against the Poisson denominators Var_e = Σ barN·k̄², Var_W = σ_W².

## 3. Relation to turboGL (Alfradique+24, arXiv:2405.00147 — extraction verified from the PDF 2026-07-15)

Their eq. (10), p.3: σ²_κL = ∫dr ρ²_M0 G² ∫₀^{k_L} k dk/2π P_L(k, z(r)) — the same
Limber object, with three differences that define our contribution:
1. **Amplitude:** they use the full-matter P_L with b=1 and a CALIBRATED cutoff
   k_L = exp(3.9 − 4.6z) Mpc⁻¹ (fit to PINOCCHIO_halos); we derive amplitude (sum
   rule) and window (unresolved-population-weighted) with no fit.
2. **Coupling:** they convolve κ_L ⊗ κ_H assuming independence (p.4 §2.1.5); the
   count–environment covariance is not quantified anywhere in the paper (checked).
   Our S_ew is exactly that term.
3. **Bookkeeping:** turboGL has NO count modulation, so their σ²_κL absorbs the
   whole clustered variance — the correct mapping is σ²_κL ↔ S_ee + 2S_ew + S_ww
   (S_tot), not S_ww alone. This is the cross-check anchor for the sizing numbers.
Their Fig. 6 (p.8): M ≈ 10¹³ h⁻¹M⊙ dominates the clustering variance (supports the
heavy-bin-dominated window; design-note §3); variance at z_s=3 ≈ 5× z_s=1.

## 4. The derived window (replaces the R_⊥ decision)

The residual field after removing resolved halos is the unresolved-halo continuum, so
its power is not "P_lin low-passed at a chosen R_⊥" but

    P_bg(k) = [ ∫_{unres} dm n(m) b(m) m ũ(k|m) / ρ̄_bg ]² P_lin(k)  + (shot noise → σ_W)

— the effective window is the bias-and-mass-weighted profile of what was removed
(NFW ũ(k|m) → 1 for kR_L(m) ≪ 1, → 0 beyond the profile scale). The design note's
per-mass R_L(M) floor is the sharp-filter approximation of this; the R_⊥ sensitivity
scan (§3/§7.10 there) is replaced by a derivation. For the sizing run we bracket:
pencil P_1D (no window, upper) is the primary number; the profile-weighted refinement
matters only if the headline ratios land near a decision boundary.

## 5. Sizing results (2026-07-15 run; fixed-⟨N⟩=100 rule, Nz=100; full tables in
`data/results/bias_field_joint/report.md`)

Two implementation notes that became findings:
- **The sum-rule subtraction route for a_w FAILS — untruncated-NFW overshoot.** The
  κ-weighted explicit capture fraction f_exp = ΣWe/Σdκ̄_tot is **1.50 at z_s=0.2 and
  1.03 at z_s=1** (>1 is impossible for truncated profiles): with the fixed-⟨N⟩ rule,
  κ_thr at low z_s is so small that rmax ≫ r200 and each explicit halo's projected
  mass within rmax exceeds its M200 (log-divergent NFW wings). Same inflation on the
  weak side: Σm1 = 0.097 at z_s=1 vs total matter budget 0.064 (×1.5). So a_w is
  computed DIRECTLY (bias-weighted first moment of the sub-threshold annuli, same
  loop as sigmakappaW, self-checked to <1e-9 against σ_W²; + sub-Mmin continuum),
  and the amplitudes inherit ≲×1.5–2 inflation from the untruncated profiles — a
  PRE-EXISTING model caveat (CLAUDE.md §caveats) newly quantified here.
- **The transverse window is first-order for the weak arm** (not a refinement): the
  sub-threshold annuli extend to ~Mpc, so the raw pencil P_1D (R=0.5 kpc, all aliased
  3D power) is a hard upper bound. Bracket: pencil (upper) vs windowed (disc at the
  κ-weighted RMS beam radius per cell, R_L(M) floor — errs low since the κ-rate is
  log-uniform in r and RMS sits near the outer edge).

Budget (windowed / pencil), against Var_tot = ΣbarN·k̄² + σ_W²:

| z_s | S_ww/Var_w,shot | 2S_ew/Var_tot | (S_ww+2S_ew)/Var_tot | S_ee/Var_tot | ρ_ew |
|--:|:--|:--|:--|:--|--:|
| 0.2 | 84 / 968 | 9.5% / 65% | **12.5% / 99%** | 7.8% / 32% | 0.999 |
| 1 | 34 / 185 | 17% / 62% | **25% / 106%** | 9.5% / 22% | 0.995 |
| 5 | 18 / 47 | 27% / 60% | **50% / 122%** | 8.1% / 15% | 0.973 |
| 10 | 16 / 34 | 30% / 59% | **62% / 130%** | 7.5% / 13% | 0.949 |

(z_s ≥ 5 ratios are indicative only: the analytic Var_tot is unclipped/tail-dominated
there. The z_s ≤ 1 numbers are the clean ones; vs the measured clipped body variance
at z_s=1 the new terms are 17%.)

Sum-rule closure (§2 check): I_b(Mlo→1e-3) + b(ν→0)·(residual mass) ≈ **0.99 (q=0.75)
vs 1.01 (q=0.8)** at z=1 — the sum rule closes to ~1% for both; the q-mismatch costs
~2%, confirming the design-note §5.1 estimate (few %, not the dominant issue).
turboGL anchor: windowed S_tot(z_s=1) = 2.6e-4 sits within the expected ×2–3
(untruncated inflation + bias weighting) of their calibrated σ²_κL range — orders
agree.

## 6. Acceptance additions (extends design-note §7)

12. σ²_W split measurement: clustered fraction of the weak background (S_ww vs σ²_W).
13. Cov(κ_W, N_h) against the prediction a_e-weighted b(M)·ξ_1D (measurable in the
    MC prototype arms; ρ_ew from §2).
14. **Split-invariance test:** move κ_thr (and the clustered channel's M_min) with the
    joint model ON — P(lnμ) must be invariant within the JSD floor; the existing
    κ_thr/Mmin harness is the apparatus. This converts the known Mmin non-convergence
    (3–8% of σ²_W, `docs/convergence_mmin_nz_note.md`) from a documented defect into
    the model's defining consistency check (clustered channel exactly invariant by
    the sum rule; shot channel invariant down to the physical floor).

## 7. Verdict (2026-07-15)

**The gate ((S_ww+2S_ew)/Var_tot ≳ few %) is passed decisively at every z_s** —
12–25% at z_s ≤ 1 even in the conservative windowed bracket, and the count–background
covariance 2S_ew (the term turboGL drops and the count-only prototype cannot see) is
~2× S_ww itself, with ρ_ew ≈ 0.95–1 as predicted. The clustered weak term is 16–84×
the weak SHOT variance — σ_W alone badly misrepresents the weak background's
fluctuation content once a correlated field exists in the model. Recommendation:
carry the clustered-κ_W term into the C++ `bias_model` design (it is deterministic
given the realized field — one amplitude table a_w[jz] per z_s, negligible runtime).

**Flags for the supervisor decision (not blockers, but material):**
1. The amplitudes inherit the untruncated-NFW inflation (≲×2 on S; even deflated the
   gate passes). The same overshoot makes f_exp > 1 at low z_s — worth fixing or at
   least bounding in the C++ implementation (truncate the a_w bookkeeping at r200?).
2. Tension with ACE: the sim's ⟨κ²⟩ already sits 36–42% ABOVE ACE at z=1
   (`docs/convergence_mmin_nz_note.md`), and this term ADDS body variance. Jointly:
   the current model both under-represents clustering correlations AND overshoots the
   N-body-calibrated variance — which sharpens, not closes, the ACE-gap question
   (candidate common cause: untruncated profiles inflating Var everywhere).
3. These are variance fractions, not PDF/KL statements. Cheap next step before any
   C++ work: add the weak arm to the MC prototype (`bias_field_prototype.py` third
   arm, a_w[jz]·δ_jz added to each realization's κ) and measure the P(lnμ)-level /
   JSD effect with the existing harness. That is the number the emulator actually
   cares about.
