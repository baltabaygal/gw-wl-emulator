# Subhalo radial profile shape fix — acceptance (2026-07-28)

The subhalo radial distribution used `dN/dx ~ x^2 B(x)/(1+cx)^2` at all three sites
(the 3D inverse CDF in `precompute`, `buildRestrictedBin`'s `p3`, and `buildWsubBin`'s
`sig2d`). That is wrong by one power of x.

## 1. The error

Green+21 (§3.2.1, Fig. 7 caption) define the bias function as a ratio of **volume
number densities**, `B(x) = (dN~/dx^3)_sub / (dN~/dx^3)_NFW`, → 1 when unbiased. So B
multiplies the host **density**:

    n_sub(x) = rho_NFW(x) B(x)  ~  B(x) / [x (1 + c x)^2]
    dN/dx    = 4 pi r^2 n_sub   ~  x B(x) / (1 + c x)^2

The x^2 shell-volume factor cancels ONE power of x against the NFW 1/x cusp, leaving
x^1. The old form implied `n_sub ~ B/(1+cx)^2` — a cored profile with an extra power of
x of central suppression stacked on top of B (inner slope x^2.25 rather than x^1.25) and
an x^-2 rather than x^-3 outskirt falloff. It contradicted both `B -> 1` at large radius
("subhaloes trace the host outside") and the Han+16 x^1.3 inner bias B is fitted to
reproduce.

## 2. Population impact (analytic)

Substructure mass fraction within 0.3 r_200, old vs corrected profile:

| z | M | old | new | ratio |
|---|---|-----|-----|-------|
| 0.5 | 1e13 | 0.0367 | 0.1098 | **2.99** |
| 0.5 | 1e14 | 0.0333 | 0.0998 | **3.00** |
| 0.5 | 1e15 | 0.0300 | 0.0903 | **3.01** |
| 1.0 | 1e14 | 0.0315 | 0.0945 | **3.00** |

Remarkably stable at 3.0x across mass and redshift. `f_s` and the `psi_res = 1e-4`
anchoring are unchanged, so the TOTAL substructure mass per host is the same — the fix
**redistributes** it inward, it does not add more.

## 3. PDF impact (MC)

480k rays/arm, 8 subprocess shards, production config
(`ml.params.PRODUCTION_CONFIG`, `subhalo_virial=False`), old `.so` vs new `.so`,
identical settings and seeds. Clipped sd with shard-SEM errors:

**480k rays/arm** (`ab_n60000.json`):

| z_s | sd old | sd new | Delta sd/sd | sigma | JSD | q99 | q99.9 | edge q01 |
|-----|--------|--------|-------------|-------|-----|-----|-------|----------|
| 0.5 | 0.032797 | 0.033179 | +1.17 +/- 0.53 % | +2.2 | 1.00e-4 (1.0x floor) | +2.2% | +2.8% | -4.3e-4 |
| 1.0 | 0.072140 | 0.073568 | +1.98 +/- 0.43 % | +4.6 | 1.04e-4 (1.0x floor) | +4.4% | +4.5% | -9.9e-4 |
| 5.0 | 0.214312 | 0.217320 | +1.40 +/- 0.24 % | +6.0 | 1.08e-4 (at floor) | +3.0% | +1.7% | -2.5e-3 |

**2M rays/arm — CONFIRMATION** (`ab_n250000.json`):

| z_s | sd old | sd new | Delta sd/sd | sigma | JSD | q99 | q99.9 | edge q01 |
|-----|--------|--------|-------------|-------|-----|-----|-------|----------|
| 1.0 | 0.072436 | 0.073809 | **+1.895 +/- 0.219 %** | **+8.7** | 5.35e-5 (**2.2x floor**) | +2.9% | +1.8% | -9.0e-4 |
| 5.0 | 0.214539 | 0.217280 | **+1.277 +/- 0.110 %** | **+11.7** | 6.08e-5 (**1.4x floor**) | +2.7% | +3.0% | -3.1e-3 |

sigma INCREASES, as expected: clumps move inward, so sightlines near host centres gain
substructure convergence.

**This is a REAL effect, and the depth test is how we know.** Quadrupling the rays left
the central value essentially unchanged (1.98 -> 1.90 %, 1.40 -> 1.28 %) while halving
the error, so significance scaled as sqrt(N) (4.6 -> 8.7 sigma, 6.0 -> 11.7 sigma). Two
other candidate effects measured the same way FAILED exactly this test in the same
session — `subhalo_virial` at z_s=0.5 (+2.17% "3.3 sigma" -> +0.045% at 2M) and
`fil_bias` at z_s=1 (-0.78% "2.2 sigma" -> -0.376%) — where the central value collapsed
instead. **Signature: a real effect holds its central value and gains sigma as sqrt(N);
a fluctuation loses the central value.**

⚠⚠ **STANDING RULE (2026-07-28): an 8-shard SEM produces 2-3 sigma false positives.**
The SEM is a chi^2 estimate on 7 dof (~27% uncertain in itself) and several quantities
get read across three redshifts. **Require a depth-doubling confirmation, or >= 5 sigma,
before calling a sigma-ratio real.**

⚠ **At 480k rays, use sigma rather than JSD for a width change**: the binned JSD sat at
~1.0x the sampling floor even where the sd change was genuine at 6 sigma, because a 1-2%
width change barely reshapes a 161-bin histogram over [-0.6, 0.6]. At 2M rays the JSD
does clear the floor (1.4-2.2x), independently confirming the effect. So "JSD at floor"
is evidence about the binning and the depth, not by itself evidence of no effect.

## 3b. DECOMPOSITION — the 2026-07-28 session changed TWO things, not one

`cpp/cosmology.cpp` was also edited that day (`halobias` q = 0.75 -> 0.8, to make b the
peak-background split of the code's own `pFC` barrier, which is (p,q) = (0.3, 0.8); the
draft already stated 0.8, so the code contradicted the text). The first old-vs-new A/B
above therefore measured **both** changes. Separated with an intermediate build
(corrected profile + q = 0.75), z_s = 1, 2M rays/arm:

| change | Delta sd/sd | sigma | JSD |
|--------|-------------|-------|-----|
| **radial profile shape fix alone** | **+1.539 +/- 0.194 %** | **+7.9 (REAL)** | at floor |
| **halobias q 0.75 -> 0.8 alone** | +0.351 +/- 0.243 % | +1.4 (not resolved) | at floor |
| sum of the two | +1.890 % | — | — |
| measured combined | **+1.895 +/- 0.219 %** | +8.7 | 2.2x floor |

**Additivity closes to 0.005%**, an independent check that the decomposition is complete
and the three builds differ only in what we think they differ in.

Coherent JSD pattern: each change alone is at the floor, but JSD grows ~quadratically in
the width mismatch, so the sum clears the floor while neither part does.

**So: the profile fix is the physics result (+1.54%, 7.9 sigma). `halobias` is a
correctness fix adopted on PBS consistency, not because it moves P(lnmu)** — same
situation as `fil_bias` and `subhalo_virial`. Do NOT attribute +1.90% to the profile.

## 4. Consequences

- **NOT bitwise** with any pre-2026-07-28 subhalo-ON run. Anything comparing to a stored
  subhalo-on reference must be regenerated. (`test_backward_compat_bitwise` is
  unaffected: its reference is the default path, which has `subhalo = false`.)
- **Fig. 4** (`fig_subhalo_sigma_decomposition`, single-host Campbell) must be
  regenerated: the host/subhalo split changes shape — sigma_sub rises at small r and
  falls at large r — though the total substructure mass is unchanged.
- The **model-4/5 gate** and the **subkappathr population sizing** were measured on the
  old profile and should be re-run; the clump reach D(m) is unchanged but the projected
  profile `Sigma_n` that model 5's envelope thins against is not.
- **The profile fix alone is +1.54% on sigma(lnmu) at z_s=1, confirmed at 7.9 sigma**
  (and the combined profile+halobias at 8.7-11.7 sigma across z_s=1/5). In KL terms a
  ~1.5% width change is only ~2e-4, below the emulator's median KL 7.3e-3 — but it is a
  **systematic** shift in the width, and the width carries the sigma_8 information in
  Vaskonen (2026). At sigma(lnmu) ~ sigma_8^2 and fixed geometry this maps to roughly
  **0.75% in sigma_8**, a central-value shift rather than a widening. State it in the
  paper rather than absorbing it into the emulator budget. (`subhalo_virial` and
  `halobias` contribute nothing resolvable here — both at the sampling floor.)

## 5. Reproduce

The old-vs-new comparison needs a pre-fix build, so it is a one-time measurement; the
driver is kept in the session scratchpad (`profile_ab.py`) and the numbers in `ab.json`
here. Going forward the corrected profile is the only path — there is no flag, because
the old form was simply wrong.
