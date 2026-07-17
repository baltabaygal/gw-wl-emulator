# Variance of the convergence from a discrete lens population

Full derivation behind the substructure variance used in `scripts/subhalo_screen.py`,
`scripts/subhalo_gate.py`, and the `subhalo_factor` / `m_floor` convergence arguments:
(i) why the lens counts are treated as Poisson and what corrects it, (ii) Campbell's
theorem for the cumulants of κ, (iii) the exact expressions implemented in the code,
(iv) the top-heavy $m^{+0.5}$ per-decade scaling that justifies the low-mass cutoff.

Written 2026-07-03. Companion to `docs/subhalo_combining.md`.

---

## 1. Setup

Place the ray (line of sight to the source) at the origin of each lens plane. The total
convergence is a sum over the discrete lens population,

$$
\kappa \;=\; \sum_i \kappa_c\!\left(|\vec{x}_i|;\, m_i\right),
$$

where $\vec{x}_i$ is the position of lens $i$ in the lens plane, $m_i$ its mass, and
$\kappa_c(d; m)$ the single-lens convergence profile at projected separation $d$. The
positions (and counts) are the random ingredients. Everything below is about the
statistics of this sum.

For an NFW lens (Wright & Brainerd 2000, as implemented in `FgNFW`):

$$
\kappa_c(d) = 2\kappa_0\, f(d/r_s), \qquad \kappa_0 = \frac{r_s\,\rho_s}{\Sigma_{\rm cr}},
$$

with $f(x)$ the universal NFW shape: $f(x) \sim \ln x$ (integrable) as $x\to 0$ and
$f(x)\sim x^{-2}$ as $x\to\infty$.

---

## 2. Why Poisson — and what corrects it

### 2.1 Poisson is the $\xi \to 0$ limit of the general count variance

For **any** point population with mean density $\bar n(\vec x)$, partition space into
infinitesimal cells with occupation numbers $n_i \in \{0,1\}$, $\mathbb{E}[n_i] = \bar n\, dV$.
The two-point correlation function $\xi$ is *defined* by the covariance of cell counts,

$$
\mathrm{Cov}(n_i, n_j) = \bar n^2\, \xi(|\vec x_i - \vec x_j|)\, dV\, dV' \qquad (i \neq j).
$$

For the count $N = \sum_i n_i$ in a region:

$$
\boxed{\;\mathrm{Var}(N) \;=\; \bar N \;+\; \bar n^2 \iint \xi(|\vec x - \vec x'|)\, dV\, dV'\;}
$$

The first term is discreteness (shot noise); the second is clustering. **The Poisson
assumption is exactly the statement that the second term is negligible**, i.e.
$\bar N\,\bar\xi \ll 1$ with $\bar\xi$ the correlation function averaged over the
sampling geometry. It must be checked, not postulated.

### 2.2 Why it holds for this geometry

1. **Radially (along the LOS):** the population is built from thin redshift shells
   separated by tens–hundreds of comoving Mpc — far beyond the halo correlation length
   ($\xi \lesssim 1$ past $\sim 10$ Mpc). Counts in different shells are effectively
   independent. Moreover, the superposition of many sparse independent point processes
   converges to a Poisson process (Palm–Khintchine theorem); the LOS population is such a
   superposition ($\sim 100$ shells, $\mathcal{O}(1)$ relevant halo each).

2. **Transversely (within a shell):** here $\xi$ is *not* small — the sampling tube has
   radius $r_{\max} \sim$ kpc–100 kpc, inside the clustering length. But the mean count
   per $(z, M)$ bin per tube is tiny ($\bar N \ll 1$; the total over all bins is
   $N_{\rm halos} = 100$). The clustering correction is $\mathcal{O}(\bar N^2 \bar\xi)$
   against the Poisson term $\mathcal{O}(\bar N)$: pairs in a kpc-scale tube are rare.

3. **Halo exclusion** (two massive halos cannot overlap) is a negative $\xi$ at small
   separations — it suppresses the already-rare same-bin pairs (sub-Poisson direction),
   again $\mathcal{O}(\bar N^2)$.

### 2.3 The surviving correction: large-scale clustering as a Cox (log-normal bias) layer

What does survive is the *coherent* modulation: the tube pierces over/under-dense regions,
so all bins at a given $z$ feel a common shift of their mean. The code restores this with
the `bias` option (see the sampling loop in `lensing.cpp`):

```
deltab = sigma * pG(mt);  lambda = exp(deltab - sigma^2/2);  N ~ Poisson(lambda * Nbar)
```

a **doubly stochastic (Cox) process**: Poisson counts with a log-normally distributed
intensity of unit mean, where $\sigma = D(z)\,\sigma(M_b)\,b(\nu)$ is the halo bias times
the matter fluctuation smoothed on the tube scale (the `Baumann (5.129)` line building
`dNh[jz][jM][1]`). The law of total variance recovers the general formula of §2.1:

$$
\mathrm{Var}(N) = \mathbb{E}[\mathrm{Var}(N|\lambda)] + \mathrm{Var}(\mathbb{E}[N|\lambda])
= \bar N + \bar N^2\left(e^{\sigma^2} - 1\right),
$$

with $b^2\bar\xi \leftrightarrow \mathrm{Var}(\lambda)$. So the model is precisely
**"Poisson given the large-scale environment"**, with the environment sampled per
realization. Sections 3–4 below are the `bias = 0` limit; with bias on, Campbell's
theorem applies conditionally on $\lambda$ and the extra variance appears as a coherent
broadening.

### 2.4 Poisson occupation for subhalos

Within a host, the code draws $N_{\rm sub} \sim \mathrm{Poisson}(N_{\rm res})$ with
positions i.i.d. from the anti-biased radial profile. Simulation support
(Boylan-Kolchin et al. 2010; Jiang & van den Bosch substructure statistics): occupation
is very close to Poisson for $m \ll M$ (the regime dominating the count), turning mildly
sub-Poisson only for the few most massive subhalos ($\psi \gtrsim 0.1$), largely excluded
by the $\psi_{\max} = 0.1$ cap. Correlated-accretion effects ("merger-tree clump
clustering") are on the deferred list. The dominant real departure from naive Poisson —
clumps arriving *bundled inside hosts* — is not assumed away: it is the cluster-process
structure of §6, which the Monte Carlo implements by construction.

---

## 3. Campbell's theorem

### 3.1 Mean and variance by the cell method

Partition the lens plane into cells of area $d^2x$. Occupation $n(\vec x) \in \{0,1\}$ with

$$
\mathbb{E}[n] = \lambda(\vec x)\, d^2x, \qquad
\mathrm{Var}[n] = \lambda\, d^2x\,(1 - \lambda\, d^2x) \to \lambda(\vec x)\, d^2x,
$$

(Bernoulli variance, first order in $d^2x$). The convergence is the deterministic profile
summed over occupied cells, $\kappa = \sum_{\rm cells} n(\vec x)\, \kappa_c(\vec x)$.

**Mean** (linearity, no assumptions):

$$
\langle\kappa\rangle = \int \lambda(\vec x)\, \kappa_c(\vec x)\, d^2x .
$$

**Variance** — Poisson independence across cells kills every cross-covariance, so the
variance of the sum is the sum of variances:

$$
\boxed{\;\mathrm{Var}(\kappa) = \int \lambda(\vec x)\, \kappa_c^2(\vec x)\, d^2x\;}
$$

(Campbell 1909; identical to shot noise in electronics, Rice 1944.) Note the *square of
the profile* weighted by the *density* — granularity noise: each lens is either there,
contributing its full $\kappa_c$, or absent. For a marked process (masses):
$\mathrm{Var}(\kappa) = \int dm \int d^2x\; \lambda(\vec x, m)\, \kappa_c^2(\vec x; m)$.

### 3.2 All cumulants from the generating functional

Because cells are independent,

$$
\mathbb{E}\!\left[e^{is\kappa}\right]
= \prod_{\rm cells}\left(1 + \lambda\, d^2x\, (e^{is\kappa_c} - 1)\right)
= \exp\left\{\int \lambda(\vec x)\left(e^{is\kappa_c(\vec x)} - 1\right) d^2x\right\}.
$$

Expanding the log in $s$, the $n$-th cumulant is

$$
\boxed{\;c_n = \int \lambda(\vec x)\, \kappa_c^{\,n}(\vec x)\, d^2x\;}
$$

Mean, variance, and third central moment are $n = 1, 2, 3$ of one formula — hence the
"Campbell-cumulant screen": the $\langle\kappa^3\rangle$ integrand is the
$\langle\kappa^2\rangle$ integrand with one more power of the profile.

---

## 4. Specialization to the code

### 4.1 Field halos (`subhalo_gate.py :: host_moments`)

Per redshift shell $[z, z+dz]$ and mass bin $[\ln M, \ln M + d\ln M]$, the mean count of
halos with centers within $r_{\max}$ of the ray is

$$
\bar N_H = \frac{c}{H(z)}\,\pi\left[(1+z)\, r_{\max}\right]^2 \frac{dn}{d\ln M}\, d\ln M\, dz ,
$$

uniform intensity $\lambda_{2D} = \bar N_H / (\pi r_{\max}^2)$ inside the disc. Campbell with
$\kappa = 2\kappa_0 f(r/r_s)$:

$$
c_2 = \int_0^{r_{\max}} \lambda_{2D}\, [2\kappa_0 f(r/r_s)]^2\, 2\pi r\, dr
    = \bar N_H\, (2\kappa_0)^2\, \frac{2 r_s^2}{r_{\max}^2}\, J_2(x_{\max}),
\qquad J_n(x) \equiv \int_0^x f^n(u)\, u\, du,
$$

which is line-for-line `c2 = barNH*(2*k0s)**2 * 2*(rss/rmax)**2 * J2(xmax)` (and `c3`
with $J_3$). Convergence: at large $x$, $f\sim x^{-2}$ so $f^2 u \sim u^{-3}$,
$f^3 u \sim u^{-5}$; at small $x$, $f \sim \ln$ and $\ln^n u \cdot u$ is integrable.
All moments $n \ge 2$ of the NFW profile exist.

### 4.2 Subhalos, unclustered approximation (`subhalo_screen.py :: dK2c`)

Treat every clump as an independent Poisson point anywhere in the plane. Intensity =
(host abundance) × (SHMF per host); integrating the clump position over the full plane:

$$
I_2(m, z) = \int_0^\infty [2\kappa_{0c} f(d/r_{sc})]^2\, 2\pi d\, \mathrm{d}d
          = 2\pi\, r_{sc}^2\, (2\kappa_{0c})^2\, J_2(\infty)
$$

(code: `I2 = 2*pi*rs_c**2*(2*k0_c)**2*J2_inf`), and

$$
\Delta\langle\kappa^2\rangle_c = \sum_{z,M} (\text{geometry})
\int d\ln m\; \frac{dN}{d\ln\psi}\; I_2(m, z).
$$

This is the **pure Poisson component only** — see §6 for what the Monte Carlo adds.

---

## 5. The per-decade mass scaling: $d\,\mathrm{Var}/d\ln m \propto m^{+0.3\ldots0.5}$

Goal: the mass dependence of $d\,\mathrm{Var}/d\ln m = (dN/d\ln m)\cdot I_2(m)$.

### 5.1 Lensing weight of one clump

$$
I_2 \propto r_s^2\,\kappa_0^2 \propto r_s^2\,(r_s\rho_s)^2 = r_s^4\,\rho_s^2 .
$$

NFW mass–structure relations ($M_{200}$ convention): with $h(c) = \ln(1+c) - c/(1+c)$,

$$
m = 4\pi \rho_s r_s^3\, h(c), \qquad
\rho_s = \frac{200}{3}\rho_c \frac{c^3}{h(c)}, \qquad r_s = \frac{r_{200}}{c},
\qquad r_{200} \propto m^{1/3}.
$$

- **Fixed concentration:** $\rho_s = \text{const}$, $r_s \propto m^{1/3}$
  $\Rightarrow I_2 \propto m^{4/3}$.
- **Running concentration** (Dutton–Macciò 2014, $c \propto m^{-0.1}$):
  $\rho_s \propto c^3 \propto m^{-0.3}$; then $r_s^3 = m/(4\pi\rho_s h) \Rightarrow
  r_s \propto m^{0.43}$, and $I_2 \propto r_s^4 \rho_s^2 \propto m^{1.73}\, m^{-0.6}
  = m^{1.13}$.

### 5.2 Abundance

Evolved SHMF (Jiang & van den Bosch): $dN/d\ln\psi = \gamma\, \psi^\alpha e^{-\beta\psi^\omega}$
with $\alpha = -0.82$; for $\psi < \psi_{\max} = 0.1$ the exponential is $\approx 1$, so
$dN/d\ln m \propto m^{-0.82}$.

### 5.3 Combine

$$
\frac{d\,\mathrm{Var}}{d\ln m} \propto
\begin{cases}
m^{-0.82}\cdot m^{4/3} = m^{+0.51} & \text{(fixed } c\text{)}\\[2pt]
m^{-0.82}\cdot m^{1.13} = m^{+0.31} & \text{(with } c(m)\text{)}
\end{cases}
$$

**Positive exponent: the variance budget is top-heavy in subhalo mass.** Each higher
decade contributes more than the one below. Integrating up from a floor, the *missing*
variance below a cutoff $m_c$ is

$$
\frac{\mathrm{Var}(<m_c)}{\mathrm{Var}_{\rm tot}} \approx
\left(\frac{m_c}{m_{\max}}\right)^{0.3\text{–}0.5},
\qquad m_{\max} = 0.1\,M .
$$

Demanding $< 5\%$ missing gives $m_c/m_{\max} \approx 10^{-4}$–$10^{-3}$: for a
$10^{14}\,M_\odot$ host, $m_c \sim 10^{9\text{–}10}\,M_\odot$ — thousands of lower-mass
subhalos are individually *and collectively* irrelevant to $\mathrm{Var}(\kappa)$, and
leaving them in the smooth $(1-f_{s,\rm res})M$ host costs nothing.

**Third moment:** $I_3 \propto r_s^2 (2\kappa_0)^3 \propto r_s^5 \rho_s^3 \propto m^{5/3}$
(fixed $c$) $\Rightarrow dc_3/d\ln m \propto m^{+0.85}$ — the tail statistics are even more
top-heavy; variance is the binding criterion for the floor.

**Numerical confirmation:** `data/subhalo_mfloor_sensitivity_z1.npz` — lowering
`m_floor` $10^7 \to 10^5$ (100× more clumps, 6× runtime) leaves the variance excess flat
within errors.

### 5.4 Relation to `subhalo_factor`

`subhalo_factor` is *not* a mass cut: it is the κ-relevance tolerance
$\varepsilon = f \cdot \kappa_{\rm thr}$ that generates a **host-distance-dependent**
cutoff $m_c(r)$ via $\kappa_c(r;\, m_c) = \varepsilon$ (inverted from the monotone
$r_{\rm thr}(m)$ table at the host-center distance $r$). Far hosts get a high cutoff
(nothing discrete), near hosts a low one (everything resolved down to `m_floor`, whose
harmlessness is guaranteed by §5.3). The default $f = 10^{-5}$ is the empirical
cross-redshift plateau of $\mathrm{Var}(\kappa)$
(`scripts/subhalo_factor_convergence.py`, `scripts/subhalo_factor_redshift_check.py`).

---

## 6. Beyond Poisson: the cluster-process terms the Monte Carlo keeps

Campbell needs independent placements; subhalos violate this twice (they come in
clusters, and they sit where their host is). The correct model is **Neyman–Scott**:
hosts are (Cox-modulated) Poisson events, each carrying a composite profile

$$
\kappa_{\rm event}(\vec x) = \kappa_H\!\big((1-f_s)M;\, \vec x\big) + \sum_j \kappa_c(\vec x;\, m_j),
$$

with the cluster random. Campbell still applies at the level of events,
$c_2 = \int \lambda_H\, \mathbb{E}[\kappa_{\rm event}^2]\, d^2x$, and expanding:

$$
\mathbb{E}[\kappa_{\rm event}^2] =
\underbrace{\kappa_H^2}_{\text{host, reduced } (1-f_s)^2}
+ \underbrace{2\,\kappa_H\, \mathbb{E}\!\left[\textstyle\sum \kappa_c\right]}_{\text{host–clump covariance } (>0)}
+ \underbrace{\textstyle\sum \mathbb{E}[\kappa_c^2]}_{\text{Poisson term (§4.2)}}
+ \underbrace{\mathbb{E}[N(N-1)]\,\mathbb{E}[\kappa_c]^2}_{\text{within-host pairs } (>0)}
$$

The analytic screen computes only the third term; the MC contains all four. Consequences,
both measured:

- the full MC substructure effect exceeds the screen's Poisson component several-fold
  (e.g. $+15\%$ vs $+3.4\%$ on $\langle\kappa^2\rangle$ at $z_s = 5$) — clustering and
  host–clump covariance dominate; **never compare the MC $\Delta\langle\kappa^2\rangle$
  to `dK2c` directly**;
- the first term is *negative* relative to the unperturbed host — the smooth-swap
  softening (host loses central $\Sigma$, the anti-biased clump profile returns less mass
  near the ray), visible as the significantly negative excess at large `subhalo_factor`.

**Robustness of §5:** the $m^{+0.3\ldots0.5}$ scaling came from the shot-noise term alone,
but every correction term is equally or more top-heavy in $m$ (the pair term scales as
$\mathbb{E}[\kappa_c]^2$, steeper in the mass-increasing amplitude), so the low-mass
cutoff conclusion survives all clustering corrections.

---

## References

- Campbell, N. (1909), Proc. Camb. Phil. Soc. 15, 117 — the shot-noise theorem.
- Rice, S. O. (1944), Bell Syst. Tech. J. 23, 282 — "Mathematical analysis of random noise".
- Neyman, J. & Scott, E. L. (1958), JRSS B 20, 1 — cluster point processes.
- Wright, C. O. & Brainerd, T. G. (2000), ApJ 534, 34 — NFW $\kappa$, $\gamma$ profiles.
- Jiang, F. & van den Bosch, F. C. (2014/16) — evolved SHMF ($\alpha, \beta, \omega$), $f_s(N_\tau)$.
- Giocoli, C. et al. (2007), astro-ph/0611221 — formation-redshift $\tilde w_f$ (see the
  corrected coefficient note in `docs/subhalo_combining.md`).
- Boylan-Kolchin, M. et al. (2010) — subhalo occupation statistics (near-Poisson for $m \ll M$).
- Dutton, A. A. & Macciò, A. V. (2014) — $c(m, z)$ used in `cons14`.
- Vaskonen (2026), arXiv:2601.06023 — the host lensing Monte Carlo this extends.

## Code pointers

| formula | where |
|---|---|
| $c_2, c_3$ host moments | `scripts/subhalo_gate.py :: host_moments` (`c2`, `c3`, `J2`, `J3`) |
| $I_2$ clump Poisson term | `scripts/subhalo_screen.py` (`I2`, `J2_inf`) |
| log-normal bias (Cox layer) | `cpp/lensing.cpp` sampling loop (`deltab`, `lambda`); `dNh[jz][jM][1]` |
| SHMF + $f_s$ | `cpp/subhalo.cpp :: precompute` |
| $m_c(r)$ inversion | `cpp/subhalo.cpp :: addClumps` (`r_thr`, `lower_bound`) |
| convergence scans | `scripts/subhalo_factor_convergence.py`, `scripts/subhalo_factor_redshift_check.py` |
| $m_{\rm floor}$ sensitivity | `data/subhalo_mfloor_sensitivity_z1.npz` |
