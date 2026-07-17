# Why the strong-lensing tail does not converge with Nz — mechanism identified (2026-07-13)

> Consolidated write-up (incl. the peak-background-split framing, the
> reference model's own validity condition, and the recommended fixed-R_bg
> field fix): `docs/nz_bias_convergence_note.md` (2026-07-14).

## Phenomenon
Raw-kappa MC (`vark_vs_nz.py`, namespace 8e8): f(kappa>1) grows with Nz with
no saturation through Nz=1600 (z_s=10: 4.5e-3 at Nz=25 -> 1.30e-2 at 1600),
q99.99(kappa) 2.05 -> 8.8; any variance window admitting the strong tail
therefore fails to converge, while P(lnmu)/JSD (body-weighted) is at the
floor at Nz=100.

## Hypotheses tested
1. **kappa_thr split position (fixed-<N> rule floats with Nz)** — RULED OUT.
   Frozen arm (`vark_vs_nz_frozen.py`, kappathr_flat pinned to the default's
   Nz=100 value, namespace 8.1e8): growth identical to the rule arm
   (z=10: 7.11e-3 / 9.66e-3 / 1.36e-2 at Nz=100/400/1600).
2. **Strong-encounter count quadrature** — RULED OUT. Analytic
   `get_expected_halo_count(z, kappathr=kc, Nz)` for kc in {0.5, 1}
   converges first-order in 1/Nz (0.967 at Nz=100, 0.999 at 1600, all z_s).
   Single-halo strong encounters (~9e-5 at z=10) are also ~100x rarer than
   the measured f(kappa_total>1): the tail events are COMPOUND.
3. **Bernoulli shortcut (`lambda*barNH < 0.2` branch, lensing.cpp:572)** —
   exonerated analytically: exactly mean-preserving; kills only same-cell
   multiplicity (rate Sum m_i^2/2, orders below f(kappa>1), and shrinking
   ~1/Nz); bias bursts route to the exact Poisson branch anyway; and the
   bias-off arm (which shares the branch) shows no growth.
4. **BIAS layer** — **CONFIRMED** (`vark_vs_nz_nobias.py`, bias=0, namespace
   8.2e8): growth vanishes — z=10 f(kappa>1) = 5.05/5.34/5.47e-3 at
   Nz=100/400/1600 (flat within errors; vs 6.96/9.53/13.0e-3 bias-on);
   z=5 flat at ~1.0e-3 (vs 1.39 -> 3.20e-3 bias-on). Clincher: bias-off
   |kappa|<1 variance ratio at Nz=100 is 0.966 vs the analytic count
   quadrature 0.967 — with bias off, ALL remaining Nz-dependence is the
   ordinary O(1/Nz) shell quadrature, converged by Nz~400.

## Mechanism
`lensing.cpp::deltaNhfNFW` (lines 123-127): each (jz, jM) cell's Poisson
mean is modulated by an INDEPENDENT log-normal lambda = exp(delta_b -
sigma_b^2/2) with sigma_b = sigma(M_b), M_b = 2 pi rmax^2 [dc(zl)-dc(zl-dz)]
rhoM0 — the tube-SEGMENT mass whose length is the SHELL WIDTH. Refining Nz
shrinks M_b, raises sigma_b (sigma(M) decreasing), so each cell's modulation
becomes heavier-tailed while draws stay white in z. This is not a
discretization of a continuum density field — it has no Nz->infinity limit
by construction. Burst cells (lambda >> 1 -> several same-(M,z) halos on one
ray) generate the growing compound kappa>1 population. Related known gotcha:
the bias blow-up for kappa_thr >~ 3e-3 is the same sigma_b sensitivity via
rmax instead of dz; z_s=10 is worst because its fixed-<N> kappa_thr_eff
(1.37e-3) already gives small tubes.

## How to converge it (options)
- **Proper fix:** grid-independent bias — correlated delta_b along the LOS
  (draw LOS density modes from P(k)), or one delta_b per FIXED comoving
  segment length L_b (Nz-independent), shared by all shells inside the
  segment. Either gives a continuum limit; the second is a small change.
- **Pragmatic:** declare the bias layer's Nz=100 discretization part of the
  MODEL definition (a regularization scale, like a halo-model smoothing).
  Then Nz-refinement tests must hold the bias segmenting fixed. Weak-lensing
  -valid observables are insensitive either way.
- Non-options: raising Nz (makes it worse), kappa_thr (ruled out).

## Scope / what is unaffected
P(lnmu) body (JSD at floor at Nz=100), all five knob-audit verdicts, the
emulator pipeline (trims these rays). Affected: raw or wide-support
<kappa^2>, f(kappa>1), q>=99.9 quantiles at z_s >~ 5 — all bias-tail-driven
and currently grid-defined.

Data: `data/results/vark_nz/{report,report_frozen,report_nobias}.md`,
shards under the same dir. Figures: `plots/vark_total_vs_nz.png`.
