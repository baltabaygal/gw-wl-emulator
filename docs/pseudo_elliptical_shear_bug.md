# pseudo_elliptical_shear_bug.md — `kappagammaNFWeps` uses cos² where the
# derivation gives sin², and that is what causes `nan_gamma`

Found 2026-07-30 while asking why `datasets/*` record `invalid_fraction ≈ 0.16`.
Living document — date every edit, delete what turns out wrong.

**Status: NOT FIXED. No C++ was changed.** This is a physics change on the
default path; it needs your decision, an A/B, and Ville's sign-off. Proposed
patch and verification protocol in §6.

---

## 1. The claim

`cpp/lensing.cpp::kappagammaNFWeps` (line 69) computes the pseudo-elliptical
shear as

```cpp
double gammaeps2 = pow(gammaeps0,2.0)
                 + 2.0*epsilon*cos(2.0*phieps)*gammaeps0*kappaeps0
                 + pow(epsilon,2.0)*(pow(kappaeps0,2.0)
                                     - pow(cos(2.0*phieps)*gammaeps0,2.0));   // <-- cos²
```

The last term should carry **sin²(2φ_ε)**, not cos²:

$$\gamma_\epsilon^2 = \gamma^2 + 2\epsilon\cos2\phi_\epsilon\,\gamma\kappa
+ \epsilon^2\left(\kappa^2 - \sin^2 2\phi_\epsilon\,\gamma^2\right)$$

equivalently, in components (this is the form to implement — it cannot go
negative by construction):

$$\gamma_{\epsilon,1} = \gamma\cos2\phi_\epsilon + \epsilon\kappa,\qquad
\gamma_{\epsilon,2} = \sqrt{1-\epsilon^2}\;\gamma\sin2\phi_\epsilon$$

`kappaeps` on line 68 is **correct** — verified to 1.4e-16. Only γ is wrong.

## 2. Evidence (derivation, not recollection)

The pseudo-elliptical prescription substitutes $x_\epsilon =
\sqrt{a_1x_1^2+a_2x_2^2}$, $a_{1,2}=1\mp\epsilon$, into the *circular*
potential: $\psi_\epsilon(x_1,x_2)=\psi(x_\epsilon)$. Everything then follows
from second derivatives by the chain rule, with $\psi'(u)=u\bar\kappa(u)$ and
$\psi''(u)=2\kappa(u)-\bar\kappa(u)$:

$$\psi_{,11}=\psi''\frac{a_1^2x_1^2}{u^2}+\psi'\frac{a_1(u^2-a_1x_1^2)}{u^3},
\quad\text{(sym. for }\psi_{,22}),\quad
\psi_{,12}=a_1a_2x_1x_2\left(\frac{\psi''}{u^2}-\frac{\psi'}{u^3}\right)$$

then $\kappa_\epsilon=\tfrac12(\psi_{,11}+\psi_{,22})$,
$\gamma_{\epsilon,1}=\tfrac12(\psi_{,11}-\psi_{,22})$,
$\gamma_{\epsilon,2}=\psi_{,12}$.

Against this exact reference, over 3000 random $(\epsilon,x,\phi)$ draws with
$\epsilon\in[0.15,0.6]$, $x\in[10^{-2},10]$:

| quantity | median rel. error | 90th pct | max |
|---|---|---|---|
| `kappaeps` (code) | 1.4e-16 | — | 7.6e-13 |
| `gammaeps2` **code, cos²** | **3.8%** | **33%** | **101×** |
| `gammaeps2` sin² | 3.7e-16 | — | 9.4e-14 |

⚠ An earlier finite-difference check of the same thing was GARBAGE (it appeared
to reject both forms). The NFW potential integral $\int^x 4\kappa_0F_1/x'\,dx'$
is log-divergent at the origin, so `quad` from 1e-9 plus a 1e-5 step produced
catastrophic cancellation. The analytic chain rule above is the trustworthy
route. Do not redo this with finite differences.

## 3. Why it produces NaN

With the correct sin² form, $\gamma_\epsilon^2=\gamma_{\epsilon,1}^2+
\gamma_{\epsilon,2}^2$ is a sum of squares and is **never negative**. With cos²
it is not a sum of squares and it goes negative. Setting $c=\cos2\phi_\epsilon$,
the cos² expression factorizes at $c=\pm1$ as

$$\left(\gamma(1-\epsilon)-\epsilon\kappa\right)\left(\gamma(1+\epsilon)+\epsilon\kappa\right)$$

so it is negative whenever $\gamma < \epsilon\kappa/(1-\epsilon)$ near the minor
axis — i.e. in the **inner region**, where the NFW cusp makes $\kappa>\gamma$.
Then `sqrt` returned NaN, `gammaj` was NaN, `detA` was NaN, and the whole ray was
discarded.

Measured rate of $\gamma_\epsilon^2<0$ per halo encounter, and the median error
where it stays positive:

| ε | γ² < 0 | median &#124;error&#124; |
|---|---|---|
| 0.25 | 1.34% | 3.0% |
| 0.30 | 1.90% | 4.0% |
| 0.35 | 2.37% | 4.9% |
| 0.40 | 3.14% | 5.8% |
| 0.45 | 3.94% | 6.6% |
| 0.50 | 5.11% | 7.3% |

These ε are exactly what `epsilonNFW` produces — $s=0.54(M/M_{\rm char})^{-0.05}$,
$\epsilon=(1-s)/(1+s)$ gives ε = 0.30 at $M=M_{\rm char}$, 0.40 at $10^2
M_{\rm char}$, 0.49 at $10^4M_{\rm char}$. **The code lives in the failure band.**

## 4. Why the failure rate grows with redshift

A ray is lost if **any** of its ~100 halos triggers the negative branch, and
$M_{\rm char}(z)$ falls steeply with z, so $M/M_{\rm char}$ — and hence ε — rises
with z. Both effects push the same way. Measured in
`datasets/backend_current_1k` (per-config invalid fraction):

| z | invalid | (max) |
|---|---|---|
| 0.0–0.5 | 0.03% | 0.1% |
| 0.5–1.0 | 0.15% | 0.3% |
| 1.0–2.0 | 0.48% | 1.3% |
| 2.0–4.0 | 2.82% | 9.6% |
| 4.0–7.0 | 14.6% | 42.6% |
| 7.0–10.1 | **36.9%** | **67.9%** |

corr(invalid, z) = **+0.886**. corr(invalid, σ₈) = −0.26, which the mechanism
also explains: larger σ₈ ⇒ larger $M_{\rm char}$ ⇒ smaller ε ⇒ fewer failures.
corr with Ωm and h are ≤ 0.11.

## 5. Blast radius

**Historic (pre-2026-07-03): rays silently deleted.** The stored datasets lost
16.3% of samples overall (`backend_current_1k`: 884 820 of 5 440 000), 37% at
z ≳ 7. The deletion is **not random** — it removes rays that passed a halo's
inner region near its minor axis, i.e. preferentially high-shear encounters. Any
tail statistic from those datasets at z ≳ 2 is suspect. `nan_gamma` and
`nonfinite_mu` count the same rays twice (881 705 each); genuine strong-lensing
rejects (`detA ≤ 0`) are only 3 115, so **the NaN, not strong lensing, is what
emptied the tail**.

**Current (post-`4b95454`, 2026-07-03): rays kept, shear silently zeroed.** That
commit added `sqrt(max(0.0, gammaeps2))`, an `xeps` floor and
`safeNFWGammaCore`. This treats the symptom: where the expression is negative,
γ is set to **0** for that encounter. Since $\det A=(1-\kappa)^2-\gamma^2$,
zeroing γ raises detA and so **lowers μ**, and it fires precisely on
high-shear inner-region encounters — i.e. it suppresses the high-μ tail, at a
rate that grows with z. There is **no counter** for how often the clamp fires.

Even where the expression stays positive it is wrong by a median 3.8%, so this
touches **every ray with an elliptical halo**, not just the pathological ones.

Affected work, in rough order of exposure:
- everything at $z_s\gtrsim2$, hardest at the LISA arm ($z_s$ to 10);
- the μ⁻²/μ⁻³ tail studies and any q99/q99.9 claim;
- all stored `datasets/` (⇒ the emulator's training data, and
  `hubble_reconstruct`'s `DatasetPDF` calibration, which is fitted to them);
- ⟨γ²⟩ and anything comparing to ACE.
Not affected: κ-only quantities (`kappaeps` is exact), and the shear-convention
question of 2026-07-02 (single- vs double-angle), which is separate.

⚠ This is *upstream* code — `halos` almost certainly carries the same
expression. Under the standing rule Claude did not look at or touch that repo;
check it yourself.

## 6. Proposed fix and verification protocol — NOT APPLIED

```cpp
// components: cannot go negative by construction
double g1 = gammaeps0*cos(2.0*phieps) + epsilon*kappaeps0;
double g2 = sqrt(max(0.0, 1.0 - epsilon*epsilon))*gammaeps0*sin(2.0*phieps);
double gammaeps = sqrt(g1*g1 + g2*g2);
```

Keep `sqrt(max(0.0, ...))` only as a belt-and-braces guard on $1-\epsilon^2$;
the γ clamp itself becomes dead code, which is the point.

Before adopting:
1. **Reproduce the exact-reference test in the repo** (`tests/`), asserting
   `kappagammaNFWeps` matches the chain-rule result to ~1e-12 over a grid in
   (ε, x, φ). That test is what should have caught this.
2. **Add a counter** for negative-γ² encounters and run the CURRENT code at
   $z_s$ = 1, 5, 10 to measure how often the clamp is firing today. This sizes
   the bug in the shipped model, which nothing currently does.
3. **A/B on the production config**, ≥2M rays/arm, at $z_s$ = 0.5, 1, 5: clipped
   σ(lnμ), q99, q01, JSD. Expect the fix to *raise* σ and the high-μ tail at
   high z. ⚠ Per CLAUDE.md §15/§18: use σ and quantiles, not JSD-at-floor, for a
   width change; get the shard SEM, never `1/sqrt(2N)`; and
   `git diff --stat cpp/` before attributing.
4. **`test_backward_compat_bitwise` will break.** That is the test doing its
   job. Follow the re-baseline protocol in `tests/capture_reference_lnmu.py`:
   revert this change alone, confirm the old reference passes, then re-baseline
   and keep the pre-fix vectors as `reference_lnmu_pre_shear_fix.npz`.
5. **Datasets and the emulator must be regenerated**, not patched — the loss is
   in the stored samples. Fold into the retrain bundle.
6. Tell Ville: it changes ⟨γ²⟩, the high-z tail, and therefore the σ₈ forecast.

## 7. What this says about the invalid-sample accounting

`invalid_fraction` was recorded faithfully in every dataset and nobody read it.
A 16% discard rate that reaches 37% at high z is not a quiet numerical detail; it
was visible in `metadata/invalid_stats` the whole time. Worth a standing check:
**fail dataset generation loudly when `invalid_fraction` exceeds a few 1e-3**,
rather than recording it and moving on.
