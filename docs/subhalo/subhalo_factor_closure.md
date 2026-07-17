# Closing the `subhalo_factor` question

**Status:** analytic model + independent MC + population aggregation + production
C++ endorsement complete (2026-07-08).
**Bottom line:** the production default `subhalo_factor = 1e-5` is the correct
choice — it is the largest factor that keeps the population-level substructure
κ-variance bias below 1% for every source redshift. It is *not* overkill.

---

## 1. What the parameter does

The subhalo model resolves clumps down to a dynamic mass floor set by a κ
threshold. A clump of mass *m* along a sightline is resolved iff its **reach**
`r_thr(m)` (the radius at which its NFW κ falls to the clump threshold) exceeds
the ray's host-center distance *r*:

```
kappa_thr,clump = subhalo_factor * kappa_thr,host          (lensing.cpp:389)
resolve clump iff  reach(m; kappa_thr,clump) >= r          (subhalo.cpp:240)
```

`kappa_thr,host` is **not** a fixed number — it is set so the mean number of
resolved host halos per sightline is 100 (`findkappathr`), giving
`kappa_thr,host ≈ 1.28e-4` at z_s = 1 (all lengths in **kpc**). Hence at the
default factor the clump threshold is `~1.3e-9`, and the corresponding reach of
a 1e7 M_sun clump is ~450 kpc. This smallness is inherited from the small host
threshold, not chosen independently.

As `subhalo_factor → 0` the floor bottoms out at `m_floor` and the dynamic split
becomes exact brute force, so the substructure variance excess

```
E = Var(kappa) - Var(kappa_nosub)
```

must plateau at the brute value. The question is: how large can the factor be
before *E* deviates from brute by more than a chosen tolerance?

## 2. Why the old convergence tests could not answer it

`scripts/subhalo_factor_{convergence,redshift_check,brute_multiseed}.py` compare
`E` between separate brute and dynamic **runs**. Those runs draw independent
clump catalogs, so the brute-vs-dynamic *difference* carries the full
realization scatter (~14% per seed, ~6–8% bootstrap at N=40k). The brute excess
alone swings 6.8e-5 → 8.3e-5 between N=40k and N=20k single-seed. The tests are
**noise-dominated** at ~5–10×the effect being measured, so their recommended
factors are not reliable.

## 3. The method that works: matched-realization + analytic Campbell

On a **single** clump catalog, compare three gates:

| gate | keep clump iff |
|---|---|
| brute | always (above m_floor) |
| proxy-r | `reach(m) >= r`  (production; r = ray-to-host-center) |
| truth-d | `reach(m) >= d`  (d = ray-to-clump distance) |

Because the catalog is shared, the deficit is measured with the noise cancelled.
The per-host paired excess is an exact Campbell integral plus mean-shift terms:

```
E_host = < C(y) + (mu(y)+delta(y))^2 + 2 kappa_full(y) (mu(y)+delta(y)) >_y
```

- `mu, C` = mean and Campbell variance of the kept clump κ-sum at ray impact y,
- `delta` = host-mass-reduction shift (host reduced by resolved clump mass),
- `<>_y` = area-weighted ray average inside the host κ-threshold disk.

Implemented in `playground/subhalo_factor_analytic_deficit.py` (pure quadrature,
no MC; converged to 4 digits with a log-y grid and a u-substitution that removes
the Abel-projection singularity).

### Validation (all 2026-07-07/08)

1. **Physics vs the C++** (`scripts/subhalo_factor_proxy_check.py` re-implementation
   checked against a standalone C++ probe linked to `libgwcore.a`):
   κ ≤ 0.3%, reach ≤ 0.75%, NFW params ≤ 0.2%, SHMF norm ≤ 1.6%.
2. **Component level:** analytic `mu(y)`, `C(y)` vs a 20M-clump brute MC — agree
   to ~2–4% (`subhalo_factor_analytic_components_check.py`).
3. **Full observable:** analytic curve vs an independent-catalog, log-y-stratified
   MC (`subhalo_factor_stratified_mc.py`, 40k rays) — agree within error bars at
   every factor.

### A subtlety that resolves an apparent contradiction

The old area-weighted **Σκ²** scan converges only at f ~ 0.1, seemingly allowing a
much larger factor. That plot measures only the Campbell term `<C(y)>` — the
within-ray Poisson piece — which is factor-insensitive up to ~0.1. The paper
observable `Var(κ) − Var(κ_nosub)` is instead dominated by the **covariance of
substructure with the smooth host across rays**, `2 Cov_y(kappa_full, mu+delta)`.
That term shrinks and *flips sign* as the factor rises (because the proxy gate
drops large-r clumps and distorts the y-profile of the mean), so the full
variance degrades at a much lower factor than Σκ². Same host, two observables,
two convergence factors — see the decomposition table in the closure note.

### The proxy tax

truth-d reaches brute accuracy at f ≈ 1e-4; proxy-r needs f ≈ 1e-5 for the same
accuracy. That order of magnitude is the price of keying the gate on r instead of
d. (A d-based gate is *not* a free win: the current host-reduction bookkeeping
would double-count — host under-reduced while only near-ray clumps are added.
Fixing that is a separate task.)

## 4. Population aggregation → the recommendation

For a full line of sight the substructure variance is additive over independent
host encounters (compound Poisson; the 2-halo term is dropped throughout the
paper), so

```
E_prod(gate) = sum_{M,z_l} Nbar(M,z_l) * E_host,nonpooled(M,z_l; gate)
```

`Nbar(M,z_l)` = mean resolved host encounters per sightline, taken **directly from
the C++** `dNH[jz][jM][0]` grid (sum = 100.6, confirming ⟨N⟩=100). Implemented in
`playground/subhalo_factor_population_aggregate.py`; grid-converged (bias shifts
< 0.05% between the 13×8 and 19×12 host grids).

**Population proxy-r bias in σ²_sub (worst over z_s ∈ {0.5, 1, 2, 5}):**

| subhalo_factor | κ_thr,clump (z_s=1) | worst bias |
|---|---|---|
| **1e-5** | 1.3e-9 | **−0.9%** |
| 3.2e-5 | 4e-9 | −2.8% |
| 1e-4 | 1.3e-8 | −6.3% |
| 1e-3 | 1.3e-7 | −18% |

Figure: `plots/figures/subhalo_factor_population_bias.png`.

The population deficit is **worse** than any single host (1e-4 costs −6% vs −3%
at M=1e13) because the substructure variance is weighted toward high-mass hosts
(1e13–1e15), where the r-vs-d proxy mismatch is largest.

**Only `subhalo_factor = 1e-5` holds |bias| < 1% for all source redshifts.**

### Production C++ endorsement (z_s = 1, N = 40k, 8 seeds)

`scripts/subhalo_factor_brute_multiseed.py`, brute vs dynamic through the real
production sampler (`playground/step3_data/production_endorse_zs1.npz`):

| factor | analytic population | production paired (per-seed) | production inv-var |
|---|---|---|---|
| 1e-4 | −5.6% | −4.8% ± 4.6% | −7.9% |
| 1e-3 | −16.4% | −14.1% ± 3.8% | −15.4% |

Both consistent within ~0.6σ; the production sampler confirms the analytic
aggregation end-to-end. (1e-5 is below the production noise floor, as expected —
it is pinned by the analytic+MC chain, not this run.)

## 5. Files

- `scripts/subhalo_factor_proxy_check.py` — shared physics (validated vs C++).
- `playground/subhalo_factor_analytic_deficit.py` — analytic per-host model.
- `playground/subhalo_factor_analytic_components_check.py` — μ,C component check.
- `playground/subhalo_factor_stratified_mc.py` — independent-catalog MC anchor.
- `playground/subhalo_factor_population_aggregate.py` — population aggregation.
- `playground/step3_data/` — C++ Nbar dumps, per-z_s results, production run.
- `plots/figures/subhalo_factor_population_bias.png` — the recommendation figure.
