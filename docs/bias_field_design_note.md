# Bias-layer redesign: correlated LOS environment field — design + study note (2026-07-15)

Design session with supervisor (whiteboard + LaTeX draft) + literature study. Companion to
`docs/nz_bias_convergence_note.md` (the problem), `data/results/bias_field_prototype/report.md`
(2026-07-14 Python prototype), `data/results/vark_nz/mechanism_note.md` (evidence chain).

**One-sentence summary:** replace the per-cell iid lognormal bias modulation with ONE
correlated Gaussian field δ_1D(χ) along the LOS (mode sum from the KP91-projected linear
P(k), transverse window floored at the halo Lagrangian radius), shared by all (M,z) cells
via b(M,z); optionally upgrade the modulation law from exp(bδ) to the conditional
first-crossing ratio of the code's own pFC.

## 1. Problem recap (details in nz_bias_convergence_note.md)

`lensing.cpp::deltaNhfNFW` modulates each (jz,jM) cell's Poisson mean by an INDEPENDENT
lognormal with σ_b = D_g(z)·b·σ(M_b), M_b = tube segment (length Δz-shell, radius
r_max(κ_thr)). Physical scale owned by numerical knobs ⇒ no continuum limit in Nz
(σ(M_b)→∞), the κ_thr ≳ 3e-3 σ_κ blow-up (same disease transversely), unphysical
independence across jM, grid-defined f(κ>1). Cells have no memory of neighbors.

## 2. The design (supervisor's formulation, verified this session)

### 2.1 1D projected spectrum (Kaiser & Peacock 1991, ApJ 379, 482, eqs. 3.2/3.8–3.10)

    P_1D(k∥) = (1/2π) ∫_{k∥}^∞ dk k P(k) W̃²(R_⊥ √(k²−k∥²))
             = (1/2π) ∫_0^∞ dk_⊥ k_⊥ P(√(k∥²+k_⊥²)) W̃²(k_⊥ R_⊥)

- Low-k 1D power is dominated by ALIASED small-scale 3D power (k_⊥ ~ 1/R_⊥); a 1D pencil
  can NOT be built by sampling P_3D in 1D.
- R_⊥ → 0 diverges for our spectrum (KP91 §3.1): the transverse window is what
  regularizes σ_1D², and it suppresses P_1D(k∥ ≳ 1/R_⊥) too — no separate longitudinal
  smoothing needed. Use the smooth window (disk tophat W̃ = 2J₁(x)/x), NOT a hard k-cut
  (fine for a variance, ξ-ringing when realizing a field).

### 2.2 Mode draw (verified against Agrawal+17 eq. 3.3; conventions cross-checked)

Periodic comoving path length L (pad L ≳ 2χ(z_s,max) — the c.c. sum makes the field
periodic; unpadded, observer and source share an environment):

    δ_1D(x) = (1/L) Σ_{n≥1} δ̃_n e^{i k_n x} + c.c.,   k_n = 2πn/L
    Re δ̃_n, Im δ̃_n  ~  N(0, σ_n²) independent,   σ_n² = (L/2) P_1D(k_n)
    σ_1D² = (1/π)∫_0^∞ P_1D dk∥ = (4/L²) Σ_{n≥1} σ_n²     [use the DISCRETE sum in the
                                                            mean-1 counterterm — exact at
                                                            finite N, Agrawal+17 trick]

Equivalent convention: a_n = δ̃_n/L, ⟨|a_n|²⟩ = P_1D(k_n)/L, Re/Im var P_1D/(2L).
z-grid is log-spaced ⇒ nonuniform χ; evaluate the cosine sum directly at cell centers
(no FFT; ~512 modes × ~100 cells, negligible cost).

### 2.3 Growth and modulation (model 1, the supervisor's write-up)

    n(x∥, M, z) = n̄(M,z) exp[ b(M,z) δ_1D(x∥) − ½ b(M,z)² σ_1D² ]

b(M,z) carries halo bias AND D(z) (z=0 field, lightcone approx; unequal-time correlator
= b(z₁)b(z₂)ξ_1D(Δχ) — standard in lognormal mock lightcones). Fixes the jM disease for
free: all masses at one z ride the SAME field. This is a log-Gaussian Cox process: pair
correlation of the counts = e^{b₁b₂ξ_1D} − 1 ≈ b₁b₂ξ_lin — "halos near halos" is a
theorem of the construction, not an imposed pair condition.

Known caveats to declare: (i) bias-in-exponent ⇒ effective bias of n exceeds b when
b²σ_1D² ~ 1 (Agrawal+17 eq. 3.5) — validation plot, not a bug; (ii) lognormal target vs
Gaussian spectrum distinction (their eqs. 3.1–3.2, ξ_G = ln(1+ξ)) matters at σ_b ~ O(1) —
decide which field the "target" spectrum refers to; (iii) NOT body-preserving: prototype
measured cross-shell correlations ~6% of Var(κ) at z=1 vs old 0.5% — JSD(old,new) at the
default grid is an acceptance measurement, not assumed null.

## 3. The scale R_⊥ — who owns it

Some smoothing scale is unavoidable (unsmoothed point variance diverges); the design
question is only who sets it. The bound:

**R_⊥ ≥ R_L(M) = (3M/4πρ̄_m)^{1/3}** — the halo's own Lagrangian radius. Justifications,
strongest first:

1. **Constructive (the "halo-removal" argument):** the HMF (pFC = first-crossing of the
   excursion-set walk) has already CONSUMED all fluctuation power at k ≳ 1/R_L(M) — that
   power is what makes the halos. Removing each generated halo from the Lagrangian field
   = flattening its patch to the patch mean ⇒ the residual "background without halos" is
   the field low-passed at the local R_L. Large scales survive intact because they are
   carried BY the halos (consistency relation ∫dM M n̄ b / ρ̄_m = 1 — mass-weighted mean
   bias is unity ⇒ modulating counts by the low-passed field double-counts nothing).
   Peak-patch (Bond & Myers 96; WebSky) is the literal production implementation.
2. **Separate-universe:** b(M,z) is defined as the k→0 response of the HMF; scale-
   independent bias is only licensed for kR_L ≪ 1 (beyond: known k²R_L² scale-dependent
   corrections). The floor is the domain of validity of writing b(M,z)·δ at all.
3. **Excursion-set:** conditional MF expansion needs σ(R_env) ≪ σ(M); at R_L they are
   equal (σ(R_L(M)) = σ(M) defines ν).

R_L spans ~0.08 Mpc (1e7 M⊙) → 1.8 (1e12) → 8 (1e14) → 18 (1e15) → ~84 Mpc (1e17), so a
single global R_⊥ vs a per-mass floor max(R_⊥, R_L(M)) is a real decision. Heavy bins
(M ≳ few×1e14 for R_⊥ ~ 10 Mpc) FORCE the per-mass floor (see ΔS > 0 in §4). Light bins
under a global R_⊥ lose legitimate environment power in [R_L(M), R_⊥] — likely immaterial
(bias-weighted σ_b is dominated by heavy bins; cf. Alfradique+24 Fig. 6: M ~ 1e13
dominates the clustering variance). Quantify with the b·n̄·κ-weighted σ(R) per-bin scan
(extension of the open M_b/M validity map). R_⊥ sensitivity scan = acceptance item; if
observables move materially across the plausible range, that is a REPORTED model
uncertainty, not a tuning knob.

## 4. Modulation law: model 2 candidate (conditional first-crossing)

Replace exp(bδ) with the excursion-set object it linearizes. Walk variables: field at
z=0, barrier δ_c(z) = δ_c0/D(z); S_env = Var(δ_1D per cell, incl. window floor):

    λ(M,z,χ) = (1/C) · T(δ₀) · pFC(δ_c(z) − δ₀, S(M) − S_env) / pFC(δ_c(z), S(M))
    T(δ₀)    = exp(D δ₀ − ½ D² S_env)          [Eulerian transport, the "1+" in b_E]
    C(M,z)   = ⟨T · ratio⟩_{δ₀}                 [16-pt Gauss–Hermite, tabulated (jz,jM)]

- Linear limit: λ ≈ 1 + D·b_E·δ₀ with b_E from pFC's OWN (p=0.3, q=0.8) — model 1
  recovered, and the q-inconsistency (§5) becomes unwritable.
- Buys: positivity without lognormal; saturation vs mass budget (∫M n(M|δ)dM tracks
  ρ̄(1+δ_E) — the whiteboard "N < m_tot/M" as a theorem; no divergent E[λ²]); nonlinear
  biases b₂, b₃… = PBS coefficients of the code's own HMF, free and self-consistent;
  ΔS = S(M) − S_env > 0 is STRUCTURAL (R_⊥ > R_L(M) enforced by domain, not by argument).
- Approximation: pFC with shifted barrier as the conditional crossing is exact for
  constant barrier + sharp-k (tower property, then C ≡ 1); for the ellipsoidal fit it is
  the standard rescaled-variable ansatz (merger trees) — hence C renormalization pins
  ⟨λ⟩ = 1 exactly regardless.
- Filaments: same machinery with pFCfil (p=0, q=0.7) and ITS barrier — filament bias must
  derive from pFCfil, not the halo formula.

Implementation ladder: `bias_model` flag — 0 legacy iid lognormal (bitwise default),
1 shared-field lognormal, 2 conditional-pFC. Model 2 = model 1's field machinery with the
cell modulation function swapped (one function + two small tables). κ_thr and Nz then set
NO variance anywhere; r_max keeps only its counting job.

## 5. Code findings (this session, verified by reading cpp/)

1. **Bias/HMF inconsistency:** `cosmology.cpp::pFC` uses (p=0.3, q=0.8) but
   `cosmology.cpp::halobias` (line ~462) is SMT01 with (p=0.3, **q=0.75**) — the bias is
   not the PBS response of the code's own mass function. Few-% at the high-ν bins that
   dominate σ_b. Testable: ∫dM M n̄ b / ρ̄_m = 1 holds for a PBS-consistent (pFC, b) pair;
   fails at few % with the current mismatch. Fix = derive b from pFC (automatic in
   model 2).
2. **Poisson-given-field is the CORRECT limit for tube sampling** (not an approximation
   to apologize for): mass-conservation anti-correlations among draws scale with the
   sampled mass fraction; the tube thins the environment cell by (r_max/R_⊥)² ~ 1e-4.
   Per-draw subtraction of drawn halos from the field would be WRONG bookkeeping (the
   cell-wide depletion — including the ~1e4× off-tube halos we never instantiate — is
   already in the conditional mean). Validation line: ΣM_drawn/(ρ̄ V_cell) per cell
   ≲ 1e-3. The subtraction scheme becomes relevant only when counting volume ~ coherence
   volume (counts-in-cells applications), or if R_⊥ → tube radius (forbidden by the floor
   anyway).
3. **Halo exclusion** (sub-Poisson counts at small separations, Baldauf+13) is not
   modeled in either law — declared caveat; the R_L floor at least stops super-Poisson
   claims at sub-halo scales.

## 6. Literature map — same problem, four dodges (none copyable, all instructive)

| Who | Scale owned by | Solution type |
|---|---|---|
| old Vaskonen layer | grid (Δz, r_max(κ_thr)) | — (the disease) |
| KP91 pencil surveys | telescope beam (~10′) | physical, measured |
| turboGL/sGL | halo-model split k_L | additive correlated κ_L ⊗ Poisson, k_L CALIBRATED |
| lognormal mocks | grid Nyquist | proper field + only cutoff-insensitive observables |
| SSC (Takada–Hu) | survey window | one shared δ_b per volume (= our structure) |
| separate universe | k→0 by definition | b valid iff kR_L ≪ 1 → our floor |
| Fleury+15 stochastic lensing | ∫P(k) explicit | white noise w/ derived amplitude + stated validity |

Key turboGL details (Alfradique+24, arXiv:2405.00147, studied in full): κ = κ_H (pure
mean-subtracted Poisson, NO count modulation — trivially grid-convergent) ⊗ κ_L (one
zero-mean lognormal, σ²_κL = ∫dr ρ²G² ∫_0^{k_L} kdk/2π P_L = Limber, i.e. our
P_1D(k∥=0) with a hard cutoff). k_L is CALIBRATED to PINOCCHIO_halos: k_L =
exp(3.9−4.6z) Mpc⁻¹ (λ_L ≈ 13 Mpc at z=1 — lands where our R_L(M_max) floor does;
2-halo term effectively off by z ≳ 2), openly absorbing missing filaments. Their
independence approximation drops the count–environment covariance our multiplicative
design keeps (selling point). Their conclusions #1/#3: PDF tails need N-body in this
entire method class ⇒ our "don't quote f(κ>1) as physics" rule SURVIVES the fix (the fix
makes the tail grid-independent, not certified-correct); filaments are the next-order
physics, not clustering. Steal their Fig. 6/7 per-mass/per-z σ², μ₃ decomposition for
the M_b validity map.

Lineage: PTHalos (Scoccimarro & Sheth 2002; Manera+12) = 2LPT field + CONDITIONAL MF per
coarse cell — the direct ancestor of this design (we swap 2LPT → Gaussian, 3D → pencil).
2LPT = documented upgrade path if environment skewness ever matters (deconstruction paper:
PINOCCHIO's gain over turboGL was mostly filaments, not clustering). Lognormal mocks
(Agrawal+17, arXiv:1706.09195; FLASK; CoLoRe) = the field-generation machinery. Peak-patch
= the literal halo-removal construction, production-grade. Full-resolution literal version
for us: infeasible by ~6 orders (R_L(M_min)=0.08 Mpc over Gpc pencil × 1e4-1e5
realizations); scaled-down slab (M > 1e12, ~0.5 Mpc cells, few realizations) is a feasible
one-off ground-truth for n(M|δ_env) and ξ_hh if ever challenged.

## 7. Validation / acceptance plan

1. Field: measured ξ_1D(Δχ) and σ_1D² vs analytic (KP91 projection with chosen window).
2. Bookkeeping: ⟨λ⟩ = 1 exact (discrete-sum counterterm / C table).
3. Linear limit: model 2 → model 1 as amplitude → 0; measured linear response = b_E(q=0.8).
4. **ξ_hh acceptance test:** measured explicit-halo clustering along the LOS per mass bin
   = b(M₁)b(M₂)ξ_1D(Δχ) at large Δχ — validates the whole chain in one measurement.
5. Consistency relation: ∫M n̄ b dM/ρ̄_m = 1 (fails now with q=0.75, §5.1).
6. Mass-budget scan (model 2): ∫M n(M|δ)dM / ρ̄(1+δ_E) across δ — quantifies the
   ellipsoidal-conditional error.
7. Existing Nz/κ_thr harness: f(κ>1) flat in Nz; κ_thr sweep — the ≳3e-3 blow-up must
   disappear (prototype: old p99 σ_b 3.3→5.1 vs new saturating 1.6→1.8 ✓).
8. JSD(old default, new default) at Nz=100 — measured, not assumed null (§2.3 iii).
9. Model 1 vs 2 ablation: body expected identical; tail tamer in 2 (saturation vs
   lognormal) — the scientifically interesting number.
10. R_⊥ (and per-mass-floor) sensitivity scan — reported as model uncertainty.
11. Thinning check: ΣM_drawn/(ρ̄V_cell) ≲ 1e-3 per cell (§5.2).

## 8a. §9 — Joint field: clustered weak background (2026-07-15, sizing in progress)

The excursion-set reading of §3.1 taken to its conclusion (full derivation + sizing:
`docs/bias_field_joint_framework.md`): the background is the below-cut halo continuum,
so the SAME realized δ_1D must also source a clustered component of κ_W (amplitude
fixed by the ∫M n b = ρ̄ sum rule, NOT a new parameter), perfectly correlated with the
count modulation — the count–environment covariance turboGL drops (their κ_L ↔ our
S_ee+2S_ew+S_ww). Adds acceptance items: σ²_W clustered/shot split, Cov(κ_W, N_h) vs
prediction, and the split-invariance test (κ_thr/M_min moves must be no-ops — converts
the Mmin non-convergence defect into the model's defining consistency check).
**Sizing DONE same day (`scripts/convergence/bias_field_joint.py`): gate PASSED
decisively — (S_ww+2S_ew)/Var_tot = 12–25% at z_s ≤ 1 (windowed; pencil upper bracket
~100%), 2S_ew ≈ 2×S_ww, ρ_ew ≈ 0.95–1; clustered weak = 16–84× the shot σ_W².** Also
found: sum-rule subtraction bookkeeping fails (f_exp = 1.50 at z_s=0.2 — untruncated-
NFW projected mass within rmax exceeds M200; ≲×2 inflation flag on all amplitudes),
and the q=0.75/0.8 sum-rule closure costs only ~2% (0.99 vs 1.01 at z=1). Next: weak
arm in the prototype MC for a JSD-level number; supervisor flags in framework §7
(ACE-gap tension: this ADDS variance while ⟨κ²⟩ already overshoots ACE).

## 8. Status and next steps

- Supervisor formulation exists (LaTeX draft, §2.1 here); Python prototype (2026-07-14)
  validated the old-arm replica exactly and the new arm's flatness; conventions and
  normalization independently verified this session (KP91 ↔ Agrawal ↔ draft all agree).
- Decisions pending (supervisor): R_⊥ prescription (global vs per-mass floor — per-mass
  forced for heavy bins either way), modulation law (model 1 vs 2), padding L.
- Then: C++ implementation behind `bias_model` (legacy 0 default, bitwise), acceptance
  suite §7, and ML training-data regeneration AFTER the flag lands (regen already pending
  for the subhalo_factor=1e-2 default anyway).

## References

KP91: Kaiser & Peacock 1991, ApJ 379, 482 (papers/ copy uploaded 2026-07-15).
Agrawal+17: arXiv:1706.09195 (lognormal_galaxies). Alfradique+24: arXiv:2405.00147
(deconstruction of one-point lensing methods). Kainulainen & Marra: arXiv:0909.0822,
PRD 83 023009, PRD 84 063004 (sGL/turboGL). Fleury, Larena & Uzan: arXiv:1508.07903.
Scoccimarro & Sheth 2002 (PTHalos); Manera+12 (BOSS mocks); Monaco 2016, arXiv:1605.07752
(approximate-methods review). Lazeyras+16, arXiv:1612.02833 (separate-universe bias).
Bond & Myers 1996; Stein+19 (peak-patch/WebSky). Baldauf+13 (halo exclusion/stochasticity).
Lacey & Cole 1993 (conditional MF); Mo & White 1996 (bias from conditional MF).
