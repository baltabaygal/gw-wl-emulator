# Mass-conserving host carve: design discussion, POC results, and decision

**Date:** 2026-07-22 (updated 2026-07-23).
**Status:** decision reached (scheme A, realized host carve); **draft updated;
C++ IMPLEMENTED** behind the `subhalo_carve` flag (default **on**), gated to
subhalo_model 3/4 + the brute reference. Sandbox-verified (see §9); **Mac
`make build` + `pytest tests/test_subhalo_carve.py` PASSED 2026-07-23** (10/10,
alongside 11/11 cosmology incl. bitwise + 8/8 model-4). Still pending: the
`subhalo_factor_jsd` acceptance rerun (partly mooted by §10 — model 4 retires
`subhalo_factor`; still relevant while model 3 is the shipped default) and the
ML-data regen. Tracked in `paper_prod/paper_memo.md` (§II.A.4 Subhalos) and memory
(`subhalo_mass_conserving_carve.md`).

**§10 (2026-07-23) — superseded in part by `subhalo_model=4`:** the supervisor
(Teams 2026-07-22/23) chose the simplest design — drop `κ_u`/`M_u` entirely, sample
every subhalo to `ψ_min = m_floor/M = 1e7/M`, host carved to `M − Σ_i m_i`. That is
scheme A with an empty unresolved band ("carved brute" of §2, now affordable:
~1.3–3.4× model-3 cost, memory `subhalo_brute_production_floor`). The A/B/C/D
conditioning analysis (§2–4) remains the justification for "host absorbs the shot
noise", but `unresolvedMass`/`M_u` and the Wsub term are on a retirement path once
model 4 becomes the shipped default (staged; Ville sign-off pending).
**Companion docs:** `docs/subhalo/wsub_gaussian_term_derivation.md` (the model-3
exactness theorem this note supersedes in part), `docs/subhalo/subhalo_combining.md`.

---

## 1. The problem (supervisor's objection)

Model 3 (production default) builds each subhalo-bearing halo as

- smooth host at the **ensemble-mean** reduced mass `(1 - f_s,b) M` (lensing.cpp:929),
- explicit resolved clumps `Σ_i m_i` (Poisson draw from the JvdB14 SHMF),
- unresolved term `κ_u = μ_u(r) + σ_u(r) N(0,1)` carrying the mean unresolved mass
  `M_u = (f_s - f_s,res) M`.

Because the host is pinned at the mean while `Σ_i m_i` fluctuates, the **total mass
laid down wiggles per realization: M ± 9%** (measured at M = 1e13, f_s = 0.14; skewed
— a few heavy clumps dominate). The draft (pre-2026-07-22) even defended this
explicitly ("conserved in expectation"). The supervisor's requirement: *a halo of
mass M must contain exactly M, every realization.* The model-1 brute path carves by
the mean too, so the wiggle was present in every arm, including the old brute
reference.

## 2. The candidate schemes

All conserve ⟨total⟩ = M in the mean; they differ in **which component absorbs the
resolved-clump shot noise** (±0.09 M at the fiducial host — heavy-clump dominated).

| | host mass | κ_u | total mass | proposer |
|---|---|---|---|---|
| **(0) current model 3** | fixed `(1-f_s)M` | marginal μ_u + σ_u N | **M ± 9%** | (status quo) |
| **(A) realized carve** | `M − Σm_i − M_u` (leftover) | marginal, unchanged | **M exactly** | Claude |
| **(B) conditional κ_u** | fixed `(1-f_s)M` | conditioned on mass = remainder `f_sM − Σm_i` | M exactly (needs negative band mass 22% of draws) | supervisor + user (chat: "compensate the rest with the kappa_u") |
| **(C) proportional hybrid** | carve, band takes its w_u = f_unr/(1−f_res) ≈ 4% share | conditioned on its share | M exactly | Claude (safe version of B) |
| **(D) full conditional** | carve incl. sampled band mass | conditioned on the band's OWN sampled mass | M exactly | user's idea done right |
| **carved brute** (truth) | `M − Σ(all clumps)` | none — all clumps explicit | M exactly | = supervisor's "drop κ_u, sample down" route (~200× cost) |

Conditioning is closed-form: κ_u and the band mass M_U are two linear (Campbell)
functionals of the same marked Poisson process ⇒ jointly Gaussian ⇒
`κ_u | M_U = m* ~ N(μ_u + (Cov/Var_M)(m* − μ_M), σ_u²(1−ρ²))`. Needs three extra
tabulated integrals (∫λm, ∫λm², ∫λκm) — same machinery as buildWsubBin. Analytic
moments verified against MC to <0.5% (ρ = 0.745 at the fiducial config).

## 3. POC results (single host, real NFW kernel, carved-brute reference)

Setup: M = 1e13 M⊙, z_l = 0.5, z_s = 1, c = 6, JvdB14 SHMF (α, β, ω = −0.82, 50, 4),
f_s target 0.14, fixed resolution split ψ_res = 1e-3 (f_res = 0.101, f_unr = 0.039,
N_res ≈ 12, N_unr ≈ 3.5e3), area-weighted rays 0.05–1 R200, 40k rays × 12 seeds,
paired seeds. Scripts: `playground/analytic/carve_single_host_kappa_pdf.py`,
`playground/analytic/carve_var_seeds.py`;
mass-budget demo `playground/analytic/mass_budget_carve_demo.py`.

**Var(κ) ×1e4, 12 seeds (mean ± sem), and paired gap vs carved brute:**

| scheme | Var(κ) | gap vs carved brute |
|---|---|---|
| current (0) | 1.7952 ± 0.0106 | **+1.48% ± 0.15%** (~10σ) |
| carve A | 1.7761 ± 0.0110 | **+0.40% ± 0.13%** |
| cond B | 1.8237 ± 0.0131 | **+3.09% ± 0.29%** |
| carve C | 1.7764 ± 0.0110 | +0.42% ± 0.13% |
| carve D | 1.7760 ± 0.0110 | +0.39% ± 0.13% |
| brute (fixed host, uncarved) | 1.7881 ± 0.0105 | +1.08% ± 0.08% |
| **carved brute** | 1.7690 ± 0.0108 | — (reference) |

Key readings:

1. **The missing host–substructure anti-correlation is real and measurable:**
   current sits +1.48% above carved brute; `bruteFixed − bruteCarved = +1.08%` is a
   direct measurement of the anti-correlation term alone.
2. **The carve captures ~¾ of it** (+0.40% residual). The residual is the Gaussian
   approximation of κ_u (dropped skewness) + host-mass nonlinearity (Jensen), not a
   mass-bookkeeping error — no scheme closes it.
3. **A ≈ C ≈ D to 0.03%: the κ_u conditioning is numerically inert.** The band's own
   mass scatter is only ±0.003 M, so conditioning on it barely moves anything; in D
   the host-mass fluctuation and the conditioned κ_u response cancel.
4. **B is worse than doing nothing** (+3.09% vs +1.48%), and needs a negative
   remainder mass in **21.6%** of draws. The conditional regression is exact only for
   the band's own ±0.003 M fluctuations; B feeds it the resolved band's ±0.09 M —
   30× outside its range — and welds together two Poisson bands that are independent
   by the restriction theorem. Smooth-host κ response to the same ±0.09 M is ~25×
   gentler than the band's mass→κ regression slope.
5. PDF-level (JSD, 40k): all schemes indistinguishable at the sampling floor — the
   effect is variance-level, body-weighted, small.

**The "wall" analysis (user follow-up: "make Σm_i + (1−f_s)M never exceed M"):**
enforcing `Σm_i ≤ f_s M` by rejection deletes the violating draws, which are
precisely the heavy-clump realizations:

- **96.6% of subhalos above 1e12 M⊙ thrown away**, 37.7% of mean substructure mass
  deleted; realized bound fraction drops 0.14 → ~0.10 (model no longer samples the
  paper's Eq. shmf);
- acceptance depending on the running sum breaks clump independence ⇒ no longer a
  Poisson process ⇒ the Campbell formulas for μ_u, σ_u lose validity;
- the remainder still fluctuates ±0.032 M (10× the band's own scatter) — the spurious
  coupling survives, just bounded.

Every scheme has a wall ("substructure drawn ≤ mass available"): B's wall sits at
0.14 M and is hit **21.4%** of the time; A's sits at 0.96 M and was hit **0 times in
40,000 draws** (needs a ~9σ SHMF excursion). A is B with the wall moved out of the
SHMF's way.

## 4. Physics verdict: accept A

- The physically random quantity at fixed M is the **assembly history** — how the
  fixed total splits between smooth and clumped. f_s (JvdB14) is the *mean* of a
  distribution with real halo-to-halo scatter, not a per-halo law. A pins the total
  (the defined quantity) and lets the split fluctuate; the smooth host is by
  definition the complement of the realized substructure.
- B over-constrains (total = M **and** substructure = f_sM exactly), fabricates a
  resolved↔unresolved anti-correlation that the underlying process does not have,
  and pushes the observable away from brute.
- The assumption-free limit of the supervisor's own proposal (sample everything
  explicitly) is **carved brute**, and it behaves like A (host absorbs), not B.
- Conservation in A is an algebraic identity — `(M − Σm_i − M_u) + Σm_i + M_u = M`
  cancels for any draw; the only edge case is M_host < 0 (never observed; production
  should carry a guard + counter).
- Cost: ~free (one extra NFW interpolation per encounter + a reorder: draw clumps
  before building the host). Compare the supervisor's literal route (drop κ_u,
  lower ε_sub until converged): ×2.4–3.9 at ε=1e-3, ×8–11 at 1e-4, ×~200–700 brute.

**What f_s does and does not do (the supervisor's notation question):** f_s stays
**fixed** — it normalizes the SHMF and never varies. Only the realized draw Σm_i
fluctuates (Poisson count × random masses), as it already does in the current code
and in brute. The draft's intermediate `f̂_s` ("realized bound fraction",
`f̂_s M ≡ Σm_i + M_u`) caused exactly this confusion and should be dropped in favor
of the explicit form `κ_NFW[M − Σm_i − M_u]`.

## 5. Per-encounter subtlety (implementation-relevant)

The resolution floor ψ_lo(r) is **ray-dependent** (dynamic floor keyed to
host-center distance), so f_s,res, M_u, and hence the partition are defined per
encounter, not per halo. Operationally fine — each realization sees one ray per
host encounter — but (i) the carve must happen inside the encounter, after the
dynamic floor is known, and (ii) draft wording should not imply a ray-independent
partition.

## 6. Draft changes made (2026-07-22, `paper_prod/draft_revised_2026-07-20.tex`)

1. Eq.(reducedhost): host term rewritten to the realized carve (currently in the
   `(1−f̂_s)M` notation — **pending simplification to the explicit
   `κ_NFW[M − Σm_i − M_u]` form** per §4); prose states conservation *exactly in
   every realization, not merely in the mean*.
2. New Eq.(kappau_moments): explicit Campbell integrals
   `μ_u(r) = ∫ dN/dlnψ dlnψ ∫ d²R Σ_2D(R) κ_NFW(|r−R|; ψM)` and the same with
   κ_NFW² for σ_u²; definitions of ψ_lo(r), ψ_min = m_floor/M, Σ_2D (Abel projection
   of Eq. radial); note that σ_u² is the intensity-weighted mean square (no mean
   subtracted).
3. κ_u paragraph: "host reduced by the full bound fraction" → "carved via the mean
   unresolved mass M_u in Eq.(reducedhost)".

**Overstated exactness claims — SOFTENED 2026-07-22** (three spots:
Eq.(reducedhost) prose l.218, Fig.(subhalo-factor) caption, Eq.(kappau_moments)
prose l.286): "reproduces mean and variance exactly / for any resolution floor" →
"reproduces the mean exactly and the variance to sub-percent accuracy." These were
inherited from the *fixed-host* model-3 theorem; under the carve the total-mass
conservation is still exact (algebraic identity, kept), but the convergence variance
holds only to the measured +0.40% (Jensen on the nonlinear host-mass→κ map +
unconditioned κ_u vs the carved-brute host↔unresolved correlation). Eq.(reducedhost)
also simplified to the explicit `κ_NFW[M − Σ_i m_i − M_u]` form (f̂_s dropped) and a
per-sightline clause added (ψ_lo(r) is ray-dependent, §5).

## 7. What remains to do

1. ~~**Implement carve A in C++**~~ **DONE 2026-07-22** (§9): `subhalo_carve` flag
   (default on), gated to model 3 + brute; reorder (clumps → M_u(r) → host at
   M − Σm_i − M_u); guard + `LensingProfile.subhalo_carve_negatives`. Conditioning
   (B/C/D) NOT implemented (inert/harmful, per §3).
2. **PENDING (user's machine):** re-run the factor-sweep acceptance
   (`subhalo_factor_jsd`) and check Fig.(subhalo-factor) still holds; PDF-level impact
   expected ≪ emulator KL 7.3e-3 (variance shift ~1.5% per host, diluted over LOS +
   field + weak arms). Also quantify the guard-clamp rate on the production grid
   (`subhalo_carve_negatives` via a profiled run — see §9 note).
3. ~~Simplify the draft equation~~ **DONE 2026-07-22**: explicit
   `κ_NFW[M − Σ_i m_i − M_u]` form, exactness claims softened, per-sightline clause
   added (see §6).
4. **PENDING (user's machine):** ML training data regen — only after the carve is
   Mac-built + acceptance-passed (bundle with the other pending regen flags in
   CLAUDE.md). The emulator's edge/flux calibration should be re-checked (the carve
   moves the low-μ host-convergence tail at the sub-percent level).

## 8. Artifacts

- `playground/analytic/mass_budget_carve_demo.py` → `plots/subhalo_mass_budget.png`
  (mass-budget histograms; joint (κ_u, M_U) Gaussian + closed-form conditional;
  reservoir/negative-mass panel; A/B/C correlation panel).
- `playground/analytic/carve_single_host_kappa_pdf.py` → `plots/single_host_kappa_pdf.png`
  (single-host κ PDF, 4 schemes vs carved brute, residuals).
- `playground/analytic/carve_var_seeds.py` (12-seed paired variance table incl. B and
  both brutes) → `plots/subhalo_carve_var_gap.png` (headline bar chart).
- Wall-enforcement numbers: inline calc (this note §3); regenerate by resampling the
  resolved band and conditioning on Σ ≤ f_s M.

## 9. C++ implementation (2026-07-22) + sandbox verification

**Flag (user decision: flag, default-on):** `LensingConfig.subhalo_carve = true`
(`lensing.h`), plumbed through `SamplingParams` (`lnmu_wrapper.{h,cpp}` ×2) and all
five py entry points + `get_simulator_config` (`python_bindings.cpp`), mirroring the
`bias_window` pattern. `false` reproduces the pre-2026-07-22 deterministic
`(1 − f_s,b)M` reduction (kept as the A/B + bitwise reference).

**Mechanics** (`lensing.cpp::add_host`, guarded early branch; carve-off and models
0 / 2-non-brute fall through to the untouched original code):
- gate `subhalo && subhalo_carve && (model 3 OR (brute AND model 1/2))`;
- reorder: `addClumps(...)` **first** (now returns the realized `Σ m_i` via a new
  `double *mass_out` out-param — `subhalo.{h,cpp}`), then `M_u(r)` from the new
  `Subhalo::unresolvedMass` (= `(f_s,b − f_s,res(r))M`, same incomplete-Γ and r_thr
  floor as the model-1 host), then build the host at `M − Σ m_i − M_u(r)`, then the
  μ_u+σ_u·𝒩 term. `M_u = 0` for brute (all clumps explicit).
- guard: `M_host < Mmin` ⇒ clamp to `Mmin`, count in `LensingProfile.subhalo_carve_negatives`.

**RNG-stream-preserving:** the host build draws no random numbers, so moving it after
the clump draw leaves the stream identical to the deterministic path — carve on/off
consume the same randoms in the same order. The change is therefore auditable: the
clump realizations are identical; only the smooth host shrinks to absorb them.

**Sandbox verification** (`tmp/carve_verify.cpp`, built against the GSL shim
`/tmp/gslshim`; NOT bitwise-comparable to the Mac build but self-consistent):
- **bitwise carve-off** — model-3 carve-off is byte-identical (κ AND κ_nosub) to the
  working tree with the carve branch removed (isolates the flag from the pre-existing,
  uncommitted radial-bias refit 0.54→0.86+η in `precompute`/`buildWsubBin`, which is
  the only other working-tree-vs-HEAD cpp diff). Subhalo-**off** is byte-identical to
  HEAD (production default unregressed).
- **RNG preservation** — κ_nosub bitwise-identical carve on-vs-off (0/40000).
- **mass bookkeeping (direct)** — ⟨Σ m_i⟩ + M_u = f_s,b·M to ~0.1% at a fiducial host
  (M≈8e12, z_l≈0.5) over r = 50/150/400 kpc: Σm_i and M_u correctly partition the
  bound mass.
- **physics direction** — carve reduces Var(κ) (anti-correlation) with a Jensen mean
  shift of +0.07% (coarse 45³ shim grid; the production magnitude is the acceptance
  rerun's job).
- **guard** — fires at a low tail rate (~1e-4 per subhalo host on the coarse grid;
  clamps cleanly, no NaN). Quantify on the production grid before quoting a rate.

**pytest:** `tests/test_subhalo_carve.py` gates the module-visible behavior for the
Mac build (config default on; default == explicit; determinism; RNG-preservation via
κ_nosub; not-inert; mass-conservation-in-the-mean; no-op when subhalo off; model-1
non-brute unaffected; brute reference carved). Var(κ) reduction and the direct
per-host mass check are NOT in pytest (sub-seed-noise / no Python access to internals)
— they live in the sandbox harness + the acceptance rerun.
