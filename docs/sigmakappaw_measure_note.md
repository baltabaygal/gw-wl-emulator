# The radial measure in `sigmakappaW` undercounts by exactly 2

**Claim.** In `lensing.cpp`'s `sigmakappaW` (weak-lens κ variance; identical in
`vianvask/halos` `lensing.cpp:145-182` and in this repo at `cpp/lensing.cpp:157-193`),
the log-radial accumulator uses `PI*r^2*dlnr` where the annulus area element is
`2*PI*r^2*dlnr`. Consequently σ²_W is low by ×2 (σ_W by √2). A second, smaller issue:
the returned `sqrt(kappa2 - kappa1^2/Nh)` subtracts a term that does not apply to a
Poisson-distributed halo count (adds another ~4–7% underestimate).

**Impact (measured, end of note): σ²_W is only 0.03–0.4% of Var(κ_total) for
z_s = 0.5–5, so nothing scientific changes. This is a correctness note, not an
alarm.**

No step below relies on an area formula supplied by us — every formula, kernel,
table, and sampling convention is taken from the production code itself.

## 1. The two counting formulas already in the code

For halos of mass M in the shell [z, z+dz], the code counts halos around the
sightline in two places:

**(i) Disk count** — `NhfNFW`, `lensing.cpp:149` (this calibrates κ_thr via ⟨N⟩=100,
so it is the convention the whole model is anchored to; same form at line 121 for
the per-cell resolved counts):

```
N(disk of radius R) = 306.535·π·((1+z)R)² / H(z) · dn/dlnM · dlnM · dz
```

**(ii) Log-annulus count** — the `sigmakappaW` radial loop, `lensing.cpp:185`:

```
dN(annulus at r, width dlnr) = 306.535·π·((1+z)r)² / H(z) · dn/dlnM · dlnr · dlnM · dz
```

These two must be consistent: summing (ii) over a log grid from R₁ to R₂ has to
reproduce N(disk R₂) − N(disk R₁) from (i). It does not:

```
Σ π r² dlnr  →  ∫_{R₁}^{R₂} π r² (dr/r)  =  π (R₂² − R₁²) / 2
```

i.e. half of the disk difference π(R₂² − R₁²). Equivalently: the annulus element is
the derivative of the disk area, d(πr²)/dlnr = **2**πr², while the loop uses πr².
The factor does not cancel upon integration — the measure is low by ×½ at every r,
so the count, κ¹, and κ² accumulators are all low by the same ×2.

## 2. Probe: the code against itself

`playground/sigmakappaw_probe.cpp` (build/run instructions in its header) does three
things at the fiducial cosmology, with κ_thr from the code's own `findkappathr(100)`:

- **[B] Self-check.** Replicates the `sigmakappaW` loop verbatim; reproduces the
  library `sigmakappaW()` to 6 digits at every z_s — so the probe tests the real
  code path.
- **[A] Count contradiction.** Over the loop's own integration domain
  [r_max, r_stop] (from κ=κ_thr down to the loop's 0.001·κ_thr stop), it accumulates
  the loop measure and, in parallel, formula (i) as a difference of two disks.
  Same domain, same prefactor. Ratio = **0.495** at every z_s (0.5 exactly, minus
  ~1% left-Riemann discretization of the dlnr=0.01 grid).
- **[C] Brute-force Monte Carlo.** Samples the weak-band halos explicitly:
  counts Poisson with mean from disk formula (i), positions **uniform in area** —
  which is the production code's own convention for placing resolved halos
  (`r = sqrt(U)·rmax`, `lensing.cpp:533`) — κ from the same
  `kappagammaNFWeps`/`kappa0NFW`/`Sigmacf`/`NFWlist` calls the loop uses.
  Then compares the empirical Var(Σκ) against `sigmakappaW()²` and against the
  Campbell integral (the loop's κ² accumulator with the 2π measure, no κ₁²/Nh term).

Results (Nreal = 2000 at z_s=1, 800 elsewhere; Var estimator error ~3–5%):

| z_s | κ_thr | count ratio loop/disk | σ²_W code | Var MC | MC / code | MC / Campbell-2π |
|---|---|---|---|---|---|---|
| 0.5 | 3.794e-5 | 0.495 | 7.196e-8 | 1.467e-7 ± 0.07e-7 | **2.04** | 0.97 |
| 1 | 1.278e-4 | 0.495 | 8.393e-7 | 1.798e-6 ± 0.06e-6 | **2.14** | 1.02 |
| 2 | 3.563e-4 | 0.495 | 6.748e-6 | 1.409e-5 ± 0.07e-5 | **2.09** | 0.99 |
| 5 | 9.038e-4 | 0.495 | 4.715e-5 | 9.353e-5 ± 0.47e-5 | **1.98** | 0.94 |

The MC also reproduces the analytic mean of the weak sum (2·κ₁ with the corrected
measure) to 0.02% at z_s=1, confirming domain and kernels are identical.

The MC/code ratio is 2 × (1 + the κ₁²/Nh effect). That subtraction is the
fixed-N iid formula Var = N(⟨κ²⟩−⟨κ⟩²); for a Poisson-distributed count the
correct result is the Campbell theorem Var = ∫n κ² with no subtraction (count
fluctuations restore the ⟨κ⟩² piece).

## 3. Why the rest of the model is unaffected

`findkappathr`/`NhfNFW` (⟨N⟩=100 calibration) and the resolved-halo sampling
(counts from line 121, positions from line 533) all use the disk form, which is
correct. The halving is confined to the weak Gaussian's σ.

## 4. Impact on observables — negligible

Measured with the production module (`sample_lensing_raw_ml`, 20k realizations):

| z_s | Var(κ_total) | σ²_W share | fix shifts Var(κ) by |
|---|---|---|---|
| 0.5 | 2.51e-4 | 0.03% | +0.03% |
| 1 | 1.19e-3 | 0.07% | +0.08% |
| 2 | 3.99e-3 | 0.17% | +0.19% |
| 5 | 1.17e-2 | 0.40% | +0.44% |

So the fix moves total Var(κ) by ≤0.5% (worst case z_s=5); published results are
safe. It is still worth fixing for correctness, and any future *subhalo* analog of
this integral should use the 2π measure and no κ₁²/Nh subtraction from the start.

**Patch:** in `sigmakappaW`, `2.0*PI` in the three accumulator lines
(`lensing.cpp:185-187` here; `173-175` in halos) and `return sqrt(kappa2)`.

**Status (2026-07-08): BOTH fixes APPLIED in this repo, supervisor-approved**
(`cpp/lensing.cpp::sigmakappaW`): `2.0*PI` measure + `return sqrt(kappa2)` (Campbell,
no κ₁²/Nh subtraction; the now-dead `Nh`/`kappa1` accumulators removed). Verified:
σ²_W = 1.76175e-6 at z_s=1 = Campbell K2 within the ~1% left-Riemann grid error,
and within ~1% of the 20k-realization Poisson MC (1.7426e-6 ± 0.017e-6). Chain vs
original: 8.393e-7 → ×2 → 1.679e-6 → drop subtraction → 1.762e-6. Bitwise test
reference recaptured. NOT yet applied in halos (user pushes halos): there the patch
is `2.0*PI` at `lensing.cpp:173-175` + `return sqrt(kappa2)` at line 181.

*(2026-07-08; probe: `playground/sigmakappaw_probe.cpp`, binary `build/sigmakappaw_probe`.)*
