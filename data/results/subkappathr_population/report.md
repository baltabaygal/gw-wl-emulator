# Population-weighted κ_thr,sub sweep — making `subhalo_model=4` affordable

**2026-07-27. Step 1 of the model-4 cost plan. Analytic only, zero MC.**

> ## ⚠ REVISED 2026-07-28 — re-derived on the corrected radial profile
>
> **Verdict: `subhalo_kappathr_factor = 0.1` stands, with a SMALLER σ loss than
> originally certified. Nothing downstream is blocked.**
>
> The 2026-07-27 numbers below were computed with an analytic radial profile that
> disagreed with `cpp/subhalo.cpp` in **three** ways, not one:
>
> | | analytic (old) | production C++ |
> |---|---|---|
> | shape | `dN/dx ~ x² B/(1+cx)²` | `dN/dx ~ x B/(1+cx)²` (fixed 2026-07-28, CLAUDE.md §15) |
> | bias scale | `x0 = 0.54`, read as **r₂₀₀** units | `x0 = BIAS_X0_RVIR(0.86)·η(c,z)` in r₂₀₀ units (refit, commit `6bf0633`) |
> | extent / ψ | `x ≤ 1`, `ψ = m/M₂₀₀` | `x ≤ η`, `ψ = m/M_vir` under `subhalo_virial` (**ON** in `PRODUCTION_CONFIG`) |
>
> Only the shape was flagged in CLAUDE.md §15. The bias-scale and virial mismatches were
> independent and were never part of the profile-fix A/B. Combined, the stale convention
> carried a **~30% radius-dependent tilt** in ⟨κ_sub⟩(y) against the engine.
>
> Fixed in `subhalo_factor_analytic_deficit.py::projected_profile` (now takes explicit
> `x0`/`xmax`/`shape_exp`, with `production_profile_params()` mirroring the C++) and
> `sigma_vs_subkappathr.py::run_kthr` (new `virial=` argument; ψ referred to M_vir for
> clump lensing, to M₂₀₀ for the carve response, matching `Mpsih` / `virialRatio`).
>
> **Validated against production C++, not asserted.** `playground/subhalo_single_host_probe.cpp`
> (gained a `virial` argv) runs the production `precompute` + `addClumps` + carve;
> `scripts/convergence/check_analytic_vs_probe.py` compares. Analytic/engine median ratios:
>
> | | ⟨N_c⟩ | ⟨κ_sub⟩ | σ_sub | κ_total |
> |---|---|---|---|---|
> | virial OFF | 1.0153 | 1.0193 | 1.0434 | 1.0065 |
> | **virial ON (production)** | **1.0153** | **1.0138** | **1.0081** | **1.0229** |
>
> The virial jump is reproduced exactly — engine ⟨N_c⟩ ×1.0997, analytic ×1.0998. The
> residual ~1.5% on ⟨N_c⟩ is mass-grid quadrature, is radius-independent, and **cancels in
> the σ ratios this sweep quotes**.
>
> **Attribution is clean.** Rerunning with `--legacy-profile` reproduces the 2026-07-27
> table below to every published digit (78.47 / 84.9 / 102 clumps/ray, 0.084 / 0.111 /
> 0.210% loss), so the whole difference is the profile convention and not a pipeline change.
>
> ### Revised result — production convention (`subhalo_virial=True`)
>
> | z_s | mult | κ_thr,sub | clumps/ray | reduction | loss % (new) | loss % (old) |
> |---|---|---|---|---|---|---|
> | 0.5 | 0.01 | 3.79e-7 | 549.6 | 3.03e3× | 0.030 | 0.045 |
> | | **0.1** | **3.79e-6** | **88.5** | **1.88e4×** | **0.059** | 0.084 |
> | | 1.0 | 3.79e-5 | 9.62 | 1.73e5× | 0.190 | 0.240 |
> | 1.0 | 0.01 | 1.28e-6 | 583.7 | 2.12e3× | 0.035 | 0.054 |
> | | **0.1** | **1.28e-5** | **93.6** | **1.33e4×** | **0.078** | 0.111 |
> | | 1.0 | 1.28e-4 | 9.91 | 1.25e5× | 0.333 | 0.408 |
> | 5.0 | 0.01 | 9.04e-6 | 672.0 | 671× | 0.041 | 0.068 |
> | | **0.1** | **9.04e-5** | **108.6** | **4.16e3×** | **0.152** | 0.210 |
> | | 1.0 | 9.04e-4 | 10.2 | 4.42e4× | 1.123 | 1.293 |
>
> The full population grew ~10% (1.11e6 → 1.24e6 clumps/ray at z_s=1) — the virial ψ
> scale — and the σ loss at f=0.1 **fell ~30%** (0.111% → 0.078% at z_s=1). Both the
> accuracy and the cost arguments therefore get *better*, not worse: 93.6 clumps/ray ×
> 40.6 ns = 3.8 µs, still ~3% of the 0.108 ms/ray fixed overhead. The gap between
> f=0.1 and f=1.0 widens slightly (0.078% vs 0.333% at z_s=1), which if anything
> strengthens the original recommendation.
>
> ### Model-4/5 gate, re-run on the corrected profile at the production convention
>
> `scripts/convergence/subhalo_model5_gate.py --virial --factor 0.1`, **fresh outdir**
> (`tmp/m5gate_newprof_zs*`), 96k rays/arm at z_s=1 and 64k at z_s=0.5/5 — 8× and 5×
> deeper than the 2026-07-27 run.
>
> | z_s | rays/arm | JSD(m4,m5) | floor m4 | floor m5 | verdict | clipped sd ratio (shard-resolved) |
> |---|---|---|---|---|---|---|
> | 0.5 | 64k | 2.67e-4 | 5.05e-4 | 5.10e-4 | **AT FLOOR** | +0.157 ± 1.877 % (+0.08σ) |
> | 1.0 | 96k | 2.40e-4 | 4.39e-4 | 4.38e-4 | **AT FLOOR** | +0.581 ± 0.971 % (+0.60σ) |
> | 5.0 | 64k | 5.11e-4 | 8.94e-4 | 1.04e-3 | **AT FLOOR** | −0.024 ± 0.231 % (−0.10σ) |
>
> All three σ ratios are consistent with zero and with the predicted sub-0.1% loss. The
> SEM ordering (1.88 / 0.97 / 0.23 %) is the fixed ±1 clip doing different jobs at
> different z_s — it is ~30σ at z_s=0.5 and trims nothing, but 4.4σ at z_s=5 where it
> genuinely removes the heavy tail. The gate resolution is therefore **worst where the
> predicted effect is smallest**, which is why this gate can only exclude gross error.
>
> ⚠⚠ **Two traps this re-run hit — both now guarded in the script.**
>
> 1. **Stale shard cache.** `run_shards` skips any `.npy` that already exists, and the
>    default outdir still held 2026-07-27 arrays generated with the OLD profile. Rerunning
>    without a fresh `--outdir` would have silently compared old-profile model 4 against
>    new-profile model 5 and reported a confident, meaningless number. **Always pass a
>    fresh `--outdir` after any physics change.**
> 2. **`1/sqrt(2N)` is the wrong SEM here.** The pooled z_s=1 ratio reads +0.55%, and
>    `1/sqrt(2·96000)` = 0.23% would make that "+2.4σ" — a false positive of exactly the
>    kind CLAUDE.md §16 documents. The clipped sd is heavy-tail dominated, so its effective
>    sample size is far below the ray count; the **shard-resolved** SEM is 0.97%, giving
>    0.6σ. The script now computes, prints and stores the shard SEM and marks it as the
>    number to use.
>
> Data: `playground/analytic/subkappathr_population.json` (production),
> `subkappathr_population_legacy.json` (attribution),
> `gate_zs*_f0.1_virial.json`. Driver gained `--legacy-virial` and `--legacy-profile`;
> `subhalo_single_host_probe.cpp` gained a `virial` argv.
> Regression suite after the rebuild: 41/41 pass (`test_cosmology_params.py` incl.
> `test_backward_compat_bitwise`, `test_subhalo_virial.py`, `test_subhalo_model4.py`,
> `test_subhalo_carve.py`).



## Question

The supervisor prefers `subhalo_model=4` (brute force, every subhalo explicitly
created down to `psi_min = m_floor/M`) and dislikes model 3's weak/unresolved split.
Model 4 costs **~300–500× more per ray** than model 3, which is the blocker.

`docs/subhalo/subhalo_kappa_threshold_note.tex` showed that for a *single* host
(M=10¹⁴, z_l=0.5, z_s=1) a per-clump convergence cut κ_thr,sub at the host's own
counting threshold retains 0.15 clumps instead of 3.4×10⁴ for a 0.2% loss in σ_κ.
Cost and variance are both dominated by whichever hosts a ray actually meets, so that
single-host number could not be quoted. This is the population-weighted version.

## Method

Host weights come from the **production engine**, not a re-derived Python HMF:
`playground/host_weight_probe.cpp` dumps the per-(z_l, M) summand that
`NhfNFW` (`cpp/lensing.cpp:138`) integrates,

    dNh(z_l, M) = c·π·((1+z_l)·r_max)² / H(z_l) · dn/dlnM · dlnM · dz,

i.e. the expected number of hosts of that class per sightline. The probe's total
reproduces `NhfNFW` exactly (100.57 at z_s=1). Hosts are Poisson-independent, so both
quantities of interest are linear in dNh:

    clumps per ray          = Σ dNh · ⟨N_ret⟩(M, z_l, κ_thr,sub)
    substructure Var(κ)     = Σ dNh · Var_tot(M, z_l, κ_thr,sub)

⟨N_ret⟩ and Var_tot come from the exact Campbell quadrature in
`playground/analytic/sigma_vs_subkappathr.py::run_kthr`, already area-weighted over
the same aperture q(y)=2y/r_max² that dNh's πr_max² defines.

Driver: `playground/analytic/sweep_subkappathr_population.py` → `subkappathr_population.json`.
Nodes: 20 masses (10⁷–10¹⁶·⁵) × 7 lens redshifts, bilinear in (log M, z_l) on the log
of each field, then weighted onto the full 99×99 engine grid.

**Aperture consistency check** (would silently bias everything if wrong): the Python
`host_rmax` used by the quadrature vs the engine's `rmaxfNFW` carried in dNh, over the
200 heaviest cells — median ratio **0.9995 / 0.9999 / 0.9996** at z_s=0.5/1/5, full
range [0.993, 1.007].

## Result

κ_thr,sub expressed as a multiple of the host counting threshold κ_thr(z_s), so
mult = 1.0 is the self-consistent choice "subhalos obey the same rule as hosts".

| z_s | κ_thr | mult | κ_thr,sub | clumps/ray | reduction | σ/σ₀ | loss |
|---|---|---|---|---|---|---|---|
| 0.5 | 3.794e-5 | 0 | — | 1.461e6 | 1× | 1.00000 | — |
| | | 0.01 | 3.79e-7 | 492 | 2 968× | 0.99955 | 0.045% |
| | | **0.1** | **3.79e-6** | **78.5** | **1.86e4×** | **0.99916** | **0.084%** |
| | | 1.0 | 3.79e-5 | 8.35 | 1.75e5× | 0.99760 | 0.240% |
| | | 10 | 3.79e-4 | 0.704 | 2.07e6× | 0.98757 | 1.243% |
| 1.0 | 1.278e-4 | 0 | — | 1.110e6 | 1× | 1.00000 | — |
| | | 0.01 | 1.28e-6 | 534 | 2 079× | 0.99946 | 0.054% |
| | | **0.1** | **1.28e-5** | **84.9** | **1.31e4×** | **0.99889** | **0.111%** |
| | | 1.0 | 1.28e-4 | 8.83 | 1.26e5× | 0.99592 | 0.408% |
| | | 10 | 1.28e-3 | 0.672 | 1.65e6× | 0.97609 | 2.391% |
| 5.0 | 9.038e-4 | 0 | — | 4.149e5 | 1× | 1.00000 | — |
| | | 0.01 | 9.04e-6 | 636 | 653× | 0.99932 | 0.068% |
| | | **0.1** | **9.04e-5** | **102** | **4 069×** | **0.99790** | **0.210%** |
| | | 1.0 | 9.04e-4 | 9.48 | 4.38e4× | 0.98707 | 1.293% |
| | | 10 | 9.04e-3 | 0.459 | 9.05e5× | 0.91289 | 8.711% |

Clump cost concentrates in cluster-mass hosts: median host mass carrying the clump
budget is 2.98/2.36/1.18 ×10¹⁴ M⊙ at z_s=0.5/1/5, with 90% below ~10¹⁵. The fiducial
10¹⁴ host of the single-host note was a well-chosen representative.

Robustness: extending the mass nodes from [10⁹,10¹⁶] to [10⁷,10¹⁶·⁵] moved the z_s=0.5
result from 8.499 → 8.350 clumps/ray and 0.246% → 0.240% loss, i.e. the low-mass end
is negligible as expected.

## Cost model — confirmed independently

Measured marginal per-ray cost (two-point fit, `sample_lnmu`, z_s=1, this Mac):

| config | marginal ms/ray |
|---|---|
| model 4 brute (today) | **45.22** |
| subhalo off | **0.108** |
| model 3 thresholded (CLAUDE.md) | 0.19–0.20 |

Subtracting the fixed overhead, the clump term is 45.11 ms for 1.110×10⁶ clumps/ray
= **40.6 ns per clump**. The clump count (analytic Campbell + engine HMF) and the
runtime (actual C++) were derived by completely independent routes and agree on a
sane per-clump cost, so the cost model holds.

Predicted cost with the cut, z_s=1:

| setting | clumps/ray | clump term | total ms/ray | vs model 4 | vs model 3 |
|---|---|---|---|---|---|
| mult 0.1 | 84.9 | 3.4 µs | **0.111** | 407× faster | ~1.75× faster |
| mult 1.0 | 8.83 | 0.36 µs | **0.108** | 418× faster | ~1.8× faster |

**Model 4 with the cut is not merely affordable — it is cheaper than the model 3 it
replaces**, and within 3% of subhalo-off. The cost objection disappears.

## Recommendation

**κ_thr,sub = κ_thr/10 (mult 0.1).** The cost battle is already won at 0.1 — the clump
term is 3% of the fixed per-ray overhead, so the extra 10× fewer clumps at mult 1.0
buys nothing measurable while costing 3–6× more accuracy (notably 1.29% at z_s=5 vs
0.21%). Keeping a decade of margin below the host threshold also gives a cleaner
claim: substructure is resolved a factor of ten *past* the point where the host
population itself is truncated, so subhalos are never the limiting resolution.

Note the quoted loss is on the **substructure contribution** to Var(κ), not on total
Var(κ). Substructure is ~13% of ⟨κ²⟩ at z_s=5 (CLAUDE.md §4), so 0.21% of that is
~0.03% of the total — far below every other uncertainty in the model.

## `subhalo_model=5` — implemented and gated (2026-07-27)

Shipped as a new model rather than a flag on model 4, so model 4 stays available as the
reference. Same population (every subhalo still exists down to ψ_min = m_floor/M, no
unresolved/Gaussian stand-in), same carve; only the *rendering* is thresholded.

**Sampler.** Draw-and-reject saves nothing, so model 5 samples the restricted intensity
directly (`Subhalo::addClumpsRestricted`, `cpp/subhalo.cpp`). Retention is the disc
d ≤ D(m) around the ray, where D(m) is the clump reach — the existing `r_thr` table, built
at κ_thr,sub instead of `subhalo_factor·κ_thr`. With Σ_n(R) the normalized projected clump
density and Smax = max_R Σ_n a **global** envelope, we Poisson-thin: propose with
probability min(1, πD²Smax), then

- small-target branch (πD²Smax < 1): position uniform in the disc, accept with Σ_n(R)/Smax
  ⇒ E[accepted] = ∫_disc Σ_n dA exactly;
- big-reach branch (capped at 1): position from the full radial profile as model 4, accept
  iff d ≤ D ⇒ E[accepted] = P(d ≤ D) exactly.

Both branches are exact — the envelope costs extra *proposals*, never accuracy. Because
Smax is global rather than Σ_n(y), the whole proposal intensity is **y-independent** and
precomputes once per (z,M) bin (`buildRestrictedBin`), so the in-loop cost is a Poisson
draw plus a binary search per proposal.

Knobs: `subhalo_kappathr` (absolute) or `subhalo_kappathr_factor` (× host κ_thr,
default 0.1). Guards: requires `subhalo_carve`, rejects `subhalo_brute`.

**Measured cost** (marginal, two-point fit, z_s=1, `sample_lnmu`):

| model | ms/ray |
|---|---|
| 4 (full brute) | 45.22 |
| **5, factor 0.1** | **0.208** |
| **5, factor 1.0** | **0.119** |
| 3 (thresholded) | 0.19–0.20 |
| subhalo off | 0.108 |

**217× faster than model 4 at factor 0.1, and already at model 3's cost.** Slightly above
the 0.111 ms/ray projection because the global envelope over-proposes; factor 1.0 lands at
0.119, essentially the subhalo-off floor.

**Acceptance gate vs model 4** (`scripts/convergence/subhalo_model5_gate.py`, 8 subprocess
shards × 1500 rays = 12 000 rays per model per z_s, `kappa_anchor=1`, factor 0.1). Every
number is calibrated against a same-model seed-split floor:

| z_s | JSD(m4, m5) | floor m4 | floor m5 | verdict | clipped mean m4 / m5 |
|---|---|---|---|---|---|
| 0.5 | 1.18e-3 | 3.89e-3 | 2.02e-3 | **AT FLOOR** | +0.000168 / +0.000408 |
| 1.0 | 2.16e-3 | 4.18e-3 | 3.78e-3 | **AT FLOOR** | +0.001912 / +0.001811 |
| 5.0 | 2.80e-3 | 5.66e-3 | 6.57e-3 | **AT FLOOR** | +0.016452 / +0.014334 |

⚠ **Read this gate for what it is.** The predicted difference (0.08–0.21% on the
*substructure part* of σ_κ, itself ~13% of ⟨κ²⟩) is ~0.03% on the total — orders of
magnitude below what 12 000 rays resolve. So "at floor" is the *expected* outcome and
confirms no gross error; it is **not** a tight confirmation of the 0.1% budget. The
analytic sweep above, not this gate, is what bounds the actual error.

⚠ The clipped-σ ratios (1.006 / 1.026 / 0.948 at z_s=1/5/0.5) scatter at the few-percent
level because the fixed ±1 clip window is a ~30σ clip at z_s=0.5 and therefore trims
nothing — at low z_s the "clipped" σ is effectively the raw σ and inherits its monster-ray
noise. The JSD on the body binning is the reliable statistic here. A z_s-adaptive trim
would be the fix if a tighter σ comparison is ever wanted.

### What is and is not measurable by MC here (2026-07-27)

Worth recording, because it is easy to write a test or a check that cannot possibly
detect what it claims to. Measured at z_s=1, 6000 rays, κ≤1 core:

- **Total Var(κ) cannot gate the threshold at all.** Across factor 0.01→100 it reads
  1.23/1.13/1.20/1.19/1.18 ×10⁻³ — an 8% swing with *no ordering*. It is dominated by
  host placement; the substructure term is ~1% of it and the threshold moves a fraction
  of that.
- **The paired perturbation Δκ = κ − κ_nosub is the right observable** (κ_nosub is the
  full unperturbed host, so Δκ is substructure plus its carve, paired within a run).
  Var(Δκ) still carries **7–24% seed scatter** over 5 seeds at 6000 rays.
- Consequently, neighbouring production thresholds are **not** distinguishable by MC:
  factor 0.01 gives Var(Δκ) = 5.12±1.24e-5 vs factor 1 = 4.66±0.33e-5 — consistent,
  and the analytic prediction for that gap is ~0.8% in variance. **This flatness is
  the result, not a failure**: it is the σ-plateau of the analytic sweep, independently
  confirmed.
- Only a near-total cut is resolvable: factor 100 gives 1.99±0.43e-5, a 2.3× drop
  (~10σ). That is what `tests/test_subhalo_model5.py` gates.

The practical rule: **the analytic Campbell sweep is the instrument for choosing the
threshold; MC can only confirm the absence of gross error.** A first version of the
monotonicity test asserted an ordering between factor 0.03 and 3.0 — a sub-percent
effect against ~15% noise — and failed on noise as it deserved to.

## Still required before this can ship

1. **Sampler restructure — draw-and-reject saves nothing.** Testing κ_c(m,d) ≥ κ_thr,sub
   needs m and d, so rejecting after drawing still instantiates all 10⁶ clumps. The
   saving requires sampling the *restricted* intensity: draw Poisson(⟨N_ret⟩(y)) and
   place from the retention-conditioned distribution, never instantiating rejects.
   Structurally the same kind of precomputed object as the existing `r_thr` table.
2. **P(lnμ) JSD gate.** Everything above certifies the mean and σ_κ only. Structural
   argument for optimism: κ_c ≥ κ_thr,sub ⟺ d ≤ d_max, so the cut *retains* close
   encounters and discards distant faint ones — the tail is built from what is kept.
   Risk sits in the body, not the tail. Must still be measured at z_s = 0.5/1/5.
3. **Mean shift.** At the single-host level the mean κ moves more than the scatter
   (+1.1% vs −0.20% at mult 1.0, non-monotonic, crossing zero near 10⁻³). Partly real
   (less mass carved ⇒ heavier smooth host) and partly the first-order carve response
   the note flags as its least trustworthy piece. Watch it in the JSD gate.

## Files

- `playground/host_weight_probe.cpp` → `tmp/host_weights_zs{0.5,1.0,5.0}.txt`
- `playground/analytic/sweep_subkappathr_population.py` → `playground/analytic/subkappathr_population.json`
- Upstream single-host: `playground/analytic/sigma_vs_subkappathr.py`,
  `docs/subhalo/subhalo_kappa_threshold_note.tex`
