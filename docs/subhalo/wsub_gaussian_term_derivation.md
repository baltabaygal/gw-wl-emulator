# The unresolved-subhalo weak term: derivation and proof of exactness

Derivation and numerical proof for the proposed per-host substructure decomposition

$$
\kappa_{\rm halo} \;=\; \kappa_{\rm host}^{\rm smooth}
\;+\; \sum_{w\,\in\,\text{resolved}} \kappa_{c,w}
\;+\; \kappa_{W,\rm sub},
$$

the within-host analog of the field-level split (explicit halos above
$\kappa_{\rm thr}$ + Gaussian $\sigma_W$ below it, `sigmakappaW`). Main results:

1. **Exactness (any floor):** with the correct bookkeeping, the split reproduces the
   brute-force mean and variance of $\kappa$ *identically at every* `subhalo_factor`
   — proven analytically (§3) and verified to 6 digits (§6).
2. **The zero-mean Gaussian alone is NOT enough.** The deficit of the current
   resolved-only scheme is dominated by the *mean-profile mismatch*, not by the dropped
   shot-noise variance. $\kappa_{W,\rm sub}$ must be
   $\mu_{\rm unres}(y) + \mathcal{N}(0, \sigma^2_{W,\rm sub}(y))$: a deterministic mean
   profile plus a Gaussian fluctuation (§4, §6).
3. **`subhalo_factor` stops being an accuracy knob.** Mean and variance are exact at any
   factor; the only approximation is Gaussianization of the dropped third-and-higher
   cumulants, quantified in §5 (dropped $c_3$ share: 0.2% at factor $10^{-3}$, 0.7% at
   $10^{-2}$, 3% at $10^{-1}$).

Written 2026-07-09. Companion to `docs/subhalo/variance_derivation.md` (general
Campbell/Neyman–Scott machinery) and `docs/subhalo/subhalo_factor_closure.md` (the
resolved-only deficit this fixes). Numerical proof:
`playground/analytic/wsub_partition_proof.py` → `wsub_partition_proof.{png,json}`.

---

## 1. Setup: the per-host clump field is exactly a marked Poisson process

Fix one host $(M, z_l)$ with the ray at host-centric impact parameter $y$ in the lens
plane, and a source at $z_s$. The code (`Subhalo::addClumps`) draws

- a count $N \sim \text{Poisson}(\bar N)$ with $\bar N = \frac{\gamma}{\alpha}(\psi_{\max}^\alpha - \psi_{\rm lo}^\alpha)$,
- masses from the power-law proposal $\propto \psi^{\alpha-1}$, thinned by
  $e^{-\beta\psi^\omega}$ — by the Poisson thinning theorem this realizes *exactly* the
  intensity of the evolved SHMF $dN/d\ln\psi = \gamma\,\psi^\alpha e^{-\beta\psi^\omega}$,
- i.i.d. 3D positions from the anti-biased radial profile
  $p_3(x) \propto x^2(1+cx)^{-2} B(x)$, isotropically projected to a 2D host-centric
  radius $R$ with surface pdf $\sigma_{2D}(R)$ (Abel projection).

So the total clump convergence at the ray,

$$
S \;=\; \sum_i \kappa_c\big(d_i;\, m_i\big), \qquad
d_i = |\vec y - \vec R_i| \;\; \text{(true clump--ray distance)},
$$

is a **marked inhomogeneous Poisson functional** with product intensity
$\lambda(m, \vec R)\, dm\, d^2R = \frac{dN}{dm}\,\sigma_{2D}(R)\, dm\, d^2R$ — not an
approximation of the sampler but an exact description of it.

## 2. Campbell cumulants

For any Poisson functional the cumulant generating function is exact
(`variance_derivation.md` §3.2):

$$
\ln \mathbb{E}\,e^{isS} = \int \lambda \left(e^{is\kappa_c} - 1\right)
\;\;\Longrightarrow\;\;
c_n(S \mid y) = \int dm \int d^2R \;\lambda(m,\vec R)\; \kappa_c^n\big(d;m\big).
$$

In particular, conditional on the encounter geometry $y$:

$$
\mu(y) = c_1, \qquad \sigma^2(y) = c_2, \qquad \text{skew} \cdot \sigma^3 = c_3 .
$$

All $c_n$ are finite: $\kappa_c \sim \ln d$ as $d\to 0$ (integrable against $d\,dd$)
and $\sim d^{-2}$ at large $d$.

## 3. Exact partition at any floor (the theorem)

The production gate keeps a clump iff $\text{reach}(m) \ge y$, i.e. iff
$m \ge m_{\rm res}(y)$ — a measurable partition of the *mass* axis into resolved
$\mathcal{R} = \{m \ge m_{\rm res}(y)\}$ and unresolved
$\mathcal{U} = \{m_{\rm floor} \le m < m_{\rm res}(y)\}$.

**Poisson restriction theorem:** the restrictions of a Poisson process to disjoint
measurable sets are *independent* Poisson processes. Hence, conditional on $y$:

$$
S = S_\mathcal{R} + S_\mathcal{U}, \qquad S_\mathcal{R} \perp\!\!\!\perp S_\mathcal{U},
\qquad c_n(S) = c_n(S_\mathcal{R}) + c_n(S_\mathcal{U}) \;\;\; \forall n,
$$

**exactly, for any placement of the floor.** The Monte Carlo already samples
$S_\mathcal{R}$ exactly. Therefore replacing $S_\mathcal{U}$ by *any* surrogate with
the correct first two cumulants,

$$
\boxed{\;\kappa_{W,\rm sub} \;=\; \mu_{\mathcal U}(y) \;+\; \mathcal{N}\!\big(0,\; \sigma^2_{\mathcal U}(y)\big)\;}
\qquad
\mu_{\mathcal U} = \int_{\mathcal U} \lambda\, \kappa_c, \quad
\sigma^2_{\mathcal U} = \int_{\mathcal U} \lambda\, \kappa_c^2,
$$

preserves the total mean and variance of $\kappa$ identically at every
`subhalo_factor`. The only approximation, at any order, is that the dropped set's
$c_{n\ge3}$ are replaced by Gaussian values (zero) — quantified in §5.

Two structural points:

- **The r-vs-d proxy bias becomes irrelevant.** The gate is a function of $(m, y)$
  only, so the unresolved integrals run over the *true* clump–ray distance $d$ — the
  proxy never enters the analytic piece. Whatever the gate misclassifies is restored
  exactly by the complement. `subhalo_factor` degrades from a tuned accuracy knob
  (closure study: $-16\%$ variance bias at $10^{-3}$) to a pure
  performance/Gaussianity knob.
- **Host–clump covariance is captured.** Both $\mu_\mathcal{U}$ and
  $\sigma_\mathcal{U}$ are drawn at the *same* $y$ as the host profile, so the
  covariance between host and unresolved-clump convergence (the dominant term found in
  `validate_split_vs_brute.py`) is carried by the $y$-dependence of the tables.

## 4. Mean bookkeeping: why zero-mean-only fails

The current scheme keeps the unresolved mass *smoothly inside the host*: the host is
reduced only by the resolved fraction, $M_{\rm eff} = (1 - f_{s,\rm res}(y))\,M$. That
approximates the unresolved mean field $\mu_\mathcal{U}(y)$ (anti-biased clump profile,
convolved with the clump kernel) by an NFW mass-difference profile — a *shape* error
that survives averaging over $y$ and feeds the paired variance through
$\mathrm{Var}_y[\mu(y)]$ and the $2\kappa_{\rm host}(y)\,\mu(y)$ covariance term.

The numbers (§6) show this mismatch — not the dropped shot noise — dominates the
resolved-only deficit. The exact scheme is therefore:

1. reduce the host by the **full** bound fraction over the sampled SHMF range,
   $M_{\rm eff} = (1 - f_{s,b})\,M$ with
   $f_{s,b} = \int_{m_{\rm floor}}^{\psi_{\max}M} \lambda\, m\, dm / M$
   ($y$-independent, one number per $(z, M)$ bin);
2. add the resolved clumps as now;
3. add the deterministic unresolved mean $\mu_\mathcal{U}(y)$;
4. add the zero-mean Gaussian $\mathcal{N}(0, \sigma^2_\mathcal{U}(y))$.

Then $\mu_\mathcal{R}(y) + \mu_\mathcal{U}(y) = \mu_{\rm brute}(y)$ and
$c_2$ likewise, term by term, and the identity of §3 closes *including* the host
bookkeeping. (Equivalently: steps 1+3 replace today's "unresolved mass hides in the
NFW host" by "unresolved mass follows the actual subhalo profile", which is also
physically the better mean model.)

## 5. Gaussianity ceiling

The surrogate errs only through $c_{n \ge 3}(S_\mathcal{U})$. From §2,
$c_3(S_\mathcal{U} \mid y) = \int_\mathcal{U} \lambda\,\kappa_c^3$. Two diagnostics:

- **Share of the clump third cumulant that is Gaussianized:**
  $\langle c_3^\mathcal{U}\rangle_y / \langle c_3^{\rm brute}\rangle_y$ — the
  fraction of the substructure skewness budget the surrogate erases. This is the
  binding criterion, because the *variance* is exact at any floor (the trap flagged in
  the handoff note: a large factor can look "converged" in $\sigma^2$ while breaking
  the PDF).
- **Pointwise skewness of the dropped set**, $\gamma_1(y) = c_3^\mathcal{U}/(c_2^\mathcal{U})^{3/2}$,
  is $\mathcal{O}(1)$ — the dropped set alone is *not* deeply Gaussian. Its imprint on
  the total-$\kappa$ PDF is nonetheless second order because $\sigma_\mathcal{U}$ is a
  small fraction of the total per-realization spread (host + resolved clumps + field
  $\sigma_W$ all add), and because the $m^{+0.85}$ per-decade top-heaviness of $c_3$
  (`variance_derivation.md` §5.3) concentrates skewness in the *resolved* masses.

Final acceptance must still be PDF-level (KL + far-tail exceedances, brute vs new
scheme) — the third design rule of the handoff note stands.

## 6. Numerical proof

`playground/analytic/wsub_partition_proof.py` extends the validated Campbell-integral
machinery (`subhalo_factor_analytic_deficit.py`, cross-checked against stratified MC in
the closure study) to $c_3$ and to the four bookkeeping variants. Fiducial single host
$M = 10^{13} M_\odot$, $z_l = 0.5$, $z_s = 1$, $\kappa_{\rm thr,host} = 1.276\times10^{-4}$,
$m_{\rm floor} = 10^7 M_\odot$. Columns are the paired $\kappa^2$ excess relative to
brute (1 = exact):

| factor | resolved only (current) | + Gaussian only | + mean profile only | + both | dropped $c_3$ share |
|---|---|---|---|---|---|
| $10^{-5}$ | 0.9990 | 0.9990 | 1.0000 | **1.000000** | 0.0000 |
| $10^{-4}$ | 0.9708 | 0.9719 | 0.9988 | **1.000000** | 0.0002 |
| $10^{-3}$ | 0.8444 | 0.8514 | 0.9930 | **1.000000** | 0.0016 |
| $10^{-2}$ | 0.6587 | 0.6811 | 0.9776 | **1.000000** | 0.0073 |
| $10^{-1}$ | 0.4278 | 0.4901 | 0.9378 | **1.000000** | 0.0304 |
| $1$ | 0.1323 | 0.3009 | 0.8314 | **1.000000** | 0.1268 |

Readings:

- **"Both" = 1 to six digits at every factor** — the closure identity of §3–4, and a
  check that the excess estimator bookkeeping (host reduction, mean shift, covariance
  terms) is implemented consistently.
- The mean profile does most of the work; the Gaussian is the smaller (but required)
  correction. Neither alone closes.
- The Gaussianity ceiling is generous: at factor $10^{-2}$ (three decades above the
  current tuned $10^{-5}$) only 0.7% of the clump $c_3$ is Gaussianized.
- **Bonus:** the population below $m_{\rm floor} = 10^7$ (integrated down to
  $10^3 M_\odot$) carries 0.42% of the clump variance and 0.03% of $c_3$ — once the
  analytic term exists it can absorb this for free, removing `m_floor` as a physical
  parameter too.

## 7. Implementation sketch (for the next step; not yet built)

In `Subhalo::precompute`, per active $(jz, jM)$ bin, tabulate on a $y$ grid:

$$
\mu_\mathcal{U}(y), \;\; \sigma^2_\mathcal{U}(y)
= \int_{m}\!\! \frac{dN}{d\ln\psi}\, \big[1 - \text{keep}(m, y)\big]
\int_0^\infty\!\! f(d\,|\,y)\; \kappa_c^{1,2}(d; m)\; dd ,
$$

with $f(d|y)$ the clump–ray distance kernel from the projected anti-biased profile
(same integrals as the proof script; ~$n_m \times n_y \times n_d$ quadrature per bin,
precompute-time only). Replace the host reduction $f_{s,\rm res}(y) \to f_{s,b}$
(drops the per-encounter incomplete-Gamma evaluation in `lensing.cpp` — simpler than
now), and per encounter add `mu_U(y) + sigma_U(y) * N(0,1)` — one table interpolation
and one Gaussian draw. `kappa_nosub` bookkeeping unchanged.

Validation battery: (i) factor-sweep flatness of total $\mathrm{Var}(\kappa)$
(implementation test of the identity); (ii) brute-vs-split PDF comparison (KL,
far-tail exceedance) to set the production factor; (iii) re-run
`validate_split_vs_brute.py` — the split moments should now land on the brute
$+14.9\%/+4.5\%$ ($z_s=5$) instead of undershooting.

## References

- Campbell (1909); Kingman, *Poisson Processes* (1993) — restriction/superposition
  and marked-process theorems.
- `docs/subhalo/variance_derivation.md` — cumulant machinery, Neyman–Scott terms,
  top-heavy mass scaling.
- `docs/subhalo/subhalo_factor_closure.md` — the resolved-only deficit measurements
  this scheme eliminates.
- `docs/subhalo/subhalo_unresolved_handoff.md` — design traps (mean bookkeeping,
  Gaussianity ceiling, PDF-level acceptance) that framed this note.
