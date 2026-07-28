# Subhalo virial convention — r_vir vs r_200 audit + fix (2026-07-28)

Companion to `docs/subhalo/subhalo_combining.md` (the population model) and
`docs/subhalo/mass_conserving_carve_note.md` (the carve). Records an audit of the
radius/mass convention in the subhalo spatial profile and the `subhalo_virial` flag
that fixes what the audit found.

## 1. The question

The subhalo population is calibrated from papers that use the **Bryan-Norman virial**
radius, while the engine works in **r_200c**. Is the conversion done, and done
everywhere it is needed?

## 2. What the sources actually say (verified from the PDFs, not from memory)

- **JvdB14** (`papers/context/JiangVdB_2014_SubhaloMassFunction.pdf`), sec. 2:
  > "Throughout we assume that both host haloes and subhaloes are defined as spheres
  > with an average density, inside their virial radii, given by
  > rho_h = Delta_crit(z) rho_crit(z)"

  So their M_0 is **M_vir**, `psi = m/M_0` is referred to M_vir, `f_s` is the mass
  fraction within R_vir, and the SHMF counts subhaloes inside R_vir.
  Also confirmed there: the halo definition is "inclusive", so `fs,1st` (eq. 26)
  **does** describe the all-orders fraction — the code's `(all orders)` comment on
  `0.3563/Ntau^0.6 - 0.075` is correct, as are alpha = -0.82, (beta, omega) = (50, 4)
  and psi_res = 1e-4.

- **Green+21** (`papers/context/Green_2021_TidalEvolutionSubstructureArtificialDisruption.pdf`), sec. 1:
  > "The halo mass is defined as the mass enclosed within the virial radius, r_vir,
  > inside of which the mean density is equal to Delta_vir(z) times the critical
  > density ... well-described by the fitting formula presented by Bryan & Norman (1998)"

  and their bias function is normalized "to unity at r_vir" (Fig. 8 in `X = R/Rvir`).
  So the radial-bias transition scale x0 = 0.86 is in **r_vir** units.

- **Han+16** (`papers/context/HanColeFrenkJing_2016_SubhaloSpatialDistribution.pdf`), sec. 1:
  > "The virial mass, M200, is defined to be the mass inside the virial radius, R200,
  > inside which the average density equals 200 times the critical density"

  i.e. Han+16 already work in r_200c and need no conversion — worth noting because
  the base profile shape is Han+16-style, so only the Green+21 bias factor and the
  JvdB14 normalization carry a virial convention.

- **The engine**: `cosmology.cpp::NFWlistf` builds
  `r200 = (3M/(4 pi 200 rhoz))^(1/3)` with `rhoz = Az(z) rhoc` and
  `Az(z) = E^2(z)`, so `rhoz = rho_crit(z)` and the grid mass **M is M_200c**.
  `cons14` is Dutton-Maccio 2014's **c200** relation (also 200 rho_crit), so the
  concentration convention matches.

## 3. Audit result

The virial convention enters in **three** places. Only one was converted.

| # | Where | Status before 2026-07-28 |
|---|-------|--------------------------|
| 1 | radial-bias transition scale x0 | **CONVERTED** — `etaVirTo200`, x0(r200) = 0.86 eta |
| 2 | radial EXTENT of the population | not converted — sampled on x in [0,1] = r <= r_200 |
| 3 | MASS normalization (psi = m/M, f_s M) | not converted — referred to M_200 |

**Item 1 is correct.** `etaVirTo200` solves
`Delta_200 c200^3/mu(c200) = Delta_vir c_vir^3/mu(c_vir)` for c_vir and returns
`c_vir/c200 = r_vir/r200`, valid because both radii share the same NFW r_s.
Verified independently: it matches a `brentq` solve of the same equation to
**<= 4e-16**, and two independent routes to M_vir/M_200 — from the profile,
`mu(c_vir)/mu(c200)`, and from the overdensity definitions, `(Delta_vir/200) eta^3` —
agree exactly, which checks the equation itself and not merely the solver. The
bisection bracket `[c200, 3 c200]` can never fail because
`Delta_vir <= 18 pi^2 = 177.65 < 200` for all z, so eta > 1 always.

**Items 2 and 3 were the bug**: a virial-calibrated population applied over an
r_200 aperture. Net excess substructure inside r_200,

    net = (M_200/M_vir) / [N(<r_200)/N(<r_vir)]

(the two errors act in opposite directions; the truncation dominates). Because psi
and x are drawn independently, this factor applies to number and mass alike.

⚠ **CORRECTED 2026-07-28.** The first version of this table was computed with the
OLD (wrong) radial profile `x^2/(1+cx)^2 B`; see `subhalo_combining.md`. With the
corrected `x B/(1+cx)^2` the population is more centrally concentrated, so a larger
share already sits inside r_200 and the excess is SMALLER than first reported:

| z | eta | M_vir/M_200 | frac inside r_200 | **net excess** | (old, wrong) |
|---|-----|-------------|-------------------|----------------|--------------|
| 0.1 | 1.285 | 1.187 | 0.701 | **1.20** | 1.49 |
| 0.5 | 1.172 | 1.123 | 0.795 | **1.12** | 1.28 |
| 1.0 | 1.111 | 1.086 | 0.855 | **1.08** | 1.18 |
| 2.0 | 1.074 | 1.062 | 0.896 | **1.05** | 1.12 |
| 5.0 | 1.058 | 1.046 | 0.920 | **1.04** | 1.09 |

(all rows at M = 1e14; the mass dependence is weak, ~1% across 1e13-1e15.)

## 4. The fix: `subhalo_virial` (default OFF, gated to subhalo_model 4/5)

- `psi` is referred to `M_vir = M mu(c_vir)/mu(c200)` (`Subhalo::Mpsih`);
- the radial profile is sampled over `x in [0, eta]` (`Subhalo::xmaxh`), i.e. out to
  r_vir, in the 3D inverse CDF, the projected profile `p2p`, the envelope `sig2dmax`
  and `sigma2Dclump` (model 5's exactness depends on the last three);
- the carve converts the realized clump mass back to the M_200 scale,
  `M_host_eff = M - Sum_i m_i / (M_vir/M_200)`.

**Legacy is bitwise**: `xmaxh = 1.0` and `Mpsih = M` when the flag is off, and every
new expression is a multiplication by those, which is exact in IEEE.

**Gated to models 4/5** because models 0-3 reduce the host through incomplete-Gamma
`f_s_res` / `Wsub` tables that are still written in M_200 units; mixing conventions
there would be inconsistent. Models 0-3 + virial throws.

### 4.0 DECISION (2026-07-28): ON in production, default still OFF in C++

`subhalo_virial=True` is pinned in `ml.params.PRODUCTION_CONFIG` (hash `0d50caf91c75`).
The shipped C++ default stays `false`, bundled with the other staged flips for Ville.

Why on, decided before dataset regeneration (turning it on mid-generation would strand
every ray produced before the switch):

1. **Correctness, and the draft already claims it.** The paper describes a JvdB14
   substructure population; with the flag off the code applies JvdB14's virial-referred
   f_s and psi, and Green+21's r_vir-normalized bias, over an r_200 aperture. The text
   would misstate the code.
2. **Asymmetric regeneration risk.** OFF leaves the population wrong by 4-20% against the
   cited source — very likely to be corrected later, forcing a regeneration. ON leaves
   only the carve sub-choice open (sec. 4.1), which moves the smooth host by ~1% of M. So
   ON has the smaller surface for a future forced re-run, and a carve revision would be a
   change to the carve, not a revert of the convention.
3. **Cost is at the sampling floor.** On the CORRECTED profile baseline:

   | z_s | 480k rays/arm | 2M rays/arm |
   |-----|---------------|-------------|
   | 0.5 | +2.17 +/- 0.66 % (+3.3 sigma) | **+0.045 +/- 0.328 % (+0.1 sigma)** |
   | 1.0 | +0.43 +/- 0.39 % (+1.1 sigma) | — |
   | 5.0 | +0.11 +/- 0.20 % (+0.5 sigma) | — |

   The z_s=0.5 "+3.3 sigma" **did not survive**: quadrupling the rays collapsed it from
   +2.17% to +0.045%, where a real +2.17% would have returned at ~6.6 sigma. JSD is at
   the floor throughout. Edge q01 shifts -3e-4..-1e-3, so no emulator edge/flux
   implication beyond the weak arm's. **Quote: at floor, |Delta sigma/sigma| <~ 0.7% at
   2 sigma (z_s=0.5).**

   ⚠⚠ **METHOD WARNING — 8-shard SEM produces 2-3 sigma false positives.** This is the
   second time in one session (see `fil_bias`: -0.78% at "2.2 sigma" -> -0.38% at 4x the
   rays). With 8 shards the SEM is a chi^2 estimate on 7 dof, ~27% uncertain in itself,
   so a nominal 3 sigma is closer to 2 sigma effective — and several quantities are
   being read across three redshifts. **Require a depth-doubling confirmation, or
   >= 5 sigma, before calling a sigma-ratio real.**

   ⚠ The FIRST measurement of this A/B was made on the pre-fix profile and reported
   "at floor" (-0.13 +/- 0.49 % at z_s=0.5). That baseline is obsolete; do not quote it.

   ⚠ An earlier reading of the 480k run claimed the analytic sec.-3 prediction had the
   "wrong sign" (sigma rising rather than falling). With the deep run that apparent rise
   is gone, so there is no sign discrepancy to explain — the effect is simply
   unresolved. The sec.-3 "excess inside r_200" factor remains a population diagnostic
   rather than a predictor of the lensing response, since clumps between r_200 and r_vir
   still lens and rays sample impact parameters far outside r_200.

### 4.1 OPEN — the carve conversion needs supervisor sign-off

Clump masses are now drawn against the M_vir budget while the smooth host is
parameterized by its M_200-equivalent grid mass, so carving `M - Sum m_i` raw would
mix apertures. The implemented choice, `M - Sum m_i (M_200/M_vir)`, keeps the smooth
host at the same **fractional** mass (1 - f_s) in both apertures; the residual is
exactly the substructure living in the r_200..r_vir shell, which was never part of
the M_200 budget. This touches the supervisor's exact-mass-conservation requirement
(`mass_conserving_carve_note.md`) and should go to Ville with the default-flip bundle.

### 4.2 Known caveats, NOT introduced by this change

- The smooth host NFW is **untruncated**, so it already carries mass beyond r_200.
  With clumps now populating r_200..r_vir there is a small double count in that
  shell, bounded by ~f_s (M_vir - M_200), i.e. ~1-2% of M.
- Green+21 note that for m/M_0 <~ 1e-2.5 **nearly half of subhaloes lie outside
  r_vir** (splashback; consistent with Bakels+21). So r_vir is itself a truncation,
  not a physical edge. Modelling splashback is out of scope here.
- Separate, upstream, and NOT addressed: the HMF is excursion-set (`pFC`), whose mass
  is conventionally virial, yet `NFWlistf` treats the same M as M_200c. The engine
  therefore uses one symbol `M` with two implied definitions. Pre-existing in
  Vaskonen's code; deserves its own look.

## 5. Evidence

- Acceptance + A/B: `data/results/subhalo_virial/report.md`
- Tests: `tests/test_subhalo_virial.py`
- A/B driver: `scripts/convergence/subhalo_virial_ab.py`
- Model-4-vs-5 consistency under the new convention:
  `scripts/convergence/subhalo_model5_gate.py --virial`
