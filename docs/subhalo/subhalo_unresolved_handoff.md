# Handoff prompt — unresolved-subhalo variance term (cowork)

Copy everything below the line into a fresh Claude cowork session.

---

You are picking up a physics/software task in `/Users/baltabay/Desktop/gw-wl-emulator`
(a C++/pybind weak-lensing magnification-PDF emulator). Read this whole message;
it is self-contained. Do NOT assume prior conversation context.

## Environment (non-negotiable)
- All Python runs in conda env `test`: use `/Users/baltabay/miniforge3/envs/test/bin/python`
  (the `conda` shell function is sandbox-blocked). numpy 2.4, scipy 1.17.
- Build a standalone C++ probe by linking the existing static lib:
  `c++ -std=c++17 -O2 -I cpp -I /opt/homebrew/Cellar/gsl/2.8/include probe.cpp
   build/libgwcore.a -L /opt/homebrew/Cellar/gsl/2.8/lib -lgsl -lgslcblas -lm -o probe`
  (this is how the physics was validated against the production code).
- Rebuild the module with `./scripts/build.sh` (cmake; ~1–2 min).

## Background: the settled result (do not redo)
The `subhalo_factor` question is CLOSED for the current gate — see
`docs/subhalo_factor_closure.md`. Summary:
- Production gate keeps a clump iff `reach(m) >= r` (host-center ray distance);
  `kappa_thr,clump = subhalo_factor * kappa_thr,host` (`cpp/lensing.cpp:389`),
  `kappa_thr,host` set by <N_halos>=100 (~1.28e-4 at z_s=1; all lengths in **kpc**).
- At fixed ray the gate is a **mass floor** `m_res(r)` (`cpp/subhalo.cpp:~240`).
- An analytic Campbell model (`playground/subhalo_factor_analytic_deficit.py`),
  an independent stratified MC (`playground/subhalo_factor_stratified_mc.py`), a
  population aggregation weighted by the exact C++ host counts
  (`playground/subhalo_factor_population_aggregate.py`), and an 8-seed production
  run all agree: population σ²_sub bias is −0.7%/−5.6%/−16% at factor 1e-5/1e-4/1e-3.
  **1e-5 is the only value under 1% bias for all z_s** — justified, not overkill.
- The Python physics (`scripts/subhalo_factor_proxy_check.py`) is validated against
  the C++ to ≤1.6% (kappa, reach, NFW params, SHMF norm).

## The task: add an unresolved-subhalo variance term
Currently a resolved host contributes `kappa = kappa_smooth_host + sum_i kappa_resolved_clump_i`,
and clumps BELOW the dynamic floor `m_res(r)` are given identically ZERO variance
(their mass sits smoothly in the host). That missing variance is the whole reason
the factor must be pushed to 1e-5. Fix it the same way the code already treats
sub-threshold HALOS: an analytic Gaussian "weak" term.

Target per resolved host, encountered at ray impact parameter `r_h`:
```
kappa_host = kappa_smooth_host_0  +  sum_i kappa_resolved_subhalo_i  +  kappa_w_unres
kappa_w_unres ~ Normal(0, sigma_sub^2(M_host, z_l, r_h))
sigma_sub^2 = <kappa^2>_unres, the Campbell 2nd moment of the UNRESOLVED (m < m_res) clumps:

<kappa^n>(M_host, z_l, r_h) = INT dM_s  (dN_sub/dM_s)(M_s, M_host, z_l)
                               INT dr_3d  p_radial(r_3d; M_host)      # anti-biased subhalo profile
                               INT dtheta/(2pi)                       # uniform angle
                               [ m_s unresolved ] * kappa_clump(M_s, d(r_h, r_3d, theta))^n
```
where `d` is the true clump-ray distance (projection of the clump 3D position onto
the lens plane relative to the ray). Because the integral runs over clump position,
it does the true-d average automatically — the r-vs-d proxy never enters the
analytic piece, so the split is EXACT in mean and variance at any floor.

This mirrors `sigmakappaW` in `cpp/lensing.cpp:157` (which integrates halo profiles
below `kappa_thr` and returns an rms drawn per sightline via
`normal_distribution PkappaW` at `cpp/lensing.cpp:372,379`). Do the subhalo version
per-host inside `Subhalo::precompute` (`cpp/subhalo.cpp:105`) as a table
`sigma_sub[jz][jM]` (it also depends on `r_h`, so either tabulate vs `r_h` or fold
the `r_h` dependence into the per-encounter draw in `addClumps`, `cpp/subhalo.cpp:228`).

SHMF + profile constants live in `cpp/subhalo.h:19` (alpha=-0.82, beta=50, omega=4,
psi_max=1) and the anti-biased radial profile in `sample_biased_radii` /
`Subhalo::invRad`. The Python analogs (validated) are in
`scripts/subhalo_factor_proxy_check.py`: `kappa_nfw`, `reach_radius`,
`sample_brute_realization` (positions), SHMF via `gamma_norm`.

## THREE design decisions that must be right (these are the traps)
1. **Mean bookkeeping — do NOT subtract the unresolved mean.** Unlike the LOS weak
   term (whose mean is absorbed into the FRW background), the unresolved subhalo
   mass really lenses. Minimal-correct choice: keep unresolved mass in the smooth
   host (reduce host only by the RESOLVED fraction `f_s,res`, as now) and add ONLY
   the zero-mean Gaussian. The residual approximation (unresolved mean shaped like
   NFW rather than the anti-biased profile) equals the `delta(y)` term already
   computed in `playground/subhalo_factor_analytic_deficit.py` — QUANTIFY it there
   before deciding whether to add an explicit `mu_unres(y)` mean profile instead.
2. **Where the split can sit is set by GAUSSIANITY, not variance.** Variance is
   exact at any floor, so a big factor looks "converged" in σ² but breaks the PDF.
   A single ~4e9 Msun clump near the ray gives kappa ~ 0.05 — no Gaussian term can
   represent that. Find the ceiling from the THIRD cumulant of the unresolved set:
   extend the `C(y)` Campbell integrals in `subhalo_factor_analytic_deficit.py` to
   `<kappa^3>_unres` and locate where skewness of the unresolved remainder becomes
   non-negligible. Realistic gain is ~1–2 decades of factor, NOT 1000x.
3. **Acceptance is at the PDF level.** Validate brute-vs-new-scheme on the
   magnification PDF P(mu) — KL divergence and far-tail exceedances — NOT on σ².

## First, decide whether it's worth building (before any implementation)
- (a) Runtime: from `main_subhalo_profile` (`cpp/main_subhalo_profile.cpp`), what
  fraction of a realization does the clump loop actually cost at factor 1e-5? If
  small, this is correctness/elegance, not speed.
- (b) The sub-1e7 population (below the hard `m_floor`) comes almost free in the
  analytic term and is currently identically zero variance — quantify that gain.
- (c) Removing a tuned magic number.
Report these three before proposing code.

## Deliverables
1. A short feasibility note (runtime fraction, sub-1e7 variance size, skewness-based
   split ceiling) — decide go/no-go with the user.
2. If go: an implementation plan touching `cpp/subhalo.{h,cpp}` and the combine step,
   with the mean-bookkeeping choice from decision 1 made explicit.
3. PDF-level validation battery (brute vs new scheme, KL + tails).

## Rules
- The user is the physicist and orchestrator; surface decisions, don't bury them.
- Reproduce every numeric claim yourself; cite `file:line`. Output of any delegate
  CLI is advisory only.
- Do NOT commit, push, or run git operations unless the user explicitly asks.
- Scope question to raise early: is this emulator-only, or also intended for the
  `halos` fork? `docs/subhalo_combining.md` is downstream of this.
```
