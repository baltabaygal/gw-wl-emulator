# SPEC — scripts/convergence/bias_field_joint.py (joint-field sizing: clustered weak background)

## Context (self-contained)

The repo's Monte-Carlo lensing model splits each line of sight into (a) explicit halos
(Poisson counts per (jz, jM) grid cell, only encounters with NFW convergence
kappa > kappa_thr) and (b) a zero-mean Gaussian "weak background" with std sigma_W =
the Campbell shot-noise amplitude of all sub-threshold encounters. A redesign of the
clustering ("bias") layer realizes ONE correlated 1D Gaussian environment field
delta(chi) along the LOS and modulates the explicit-halo counts by
lambda = exp(g - Var(g)/2), g = b(M,z) * Dg(z) * delta_segment.

THIS script sizes the missing piece of that design: the same matter field that
modulates the counts also *is* mass, so the weak background must contain a clustered
component perfectly correlated with the count modulation:

    kappa_W = kappa_W,shot (current sigma_W, kept)  +  kappa_W,clust
    kappa_W,clust = sum_shells  a_w[jz] * delta_seg[jz]

with the amplitude a_w fixed by mass bookkeeping (sum rule), NOT by a new parameter.
The deliverable is analytic: per source redshift, the variance budget
{Poisson explicit, Poisson weak, clustered explicit-explicit, clustered weak-weak,
clustered cross}, so we can decide whether the clustered-weak + cross terms are worth
implementing in C++.

All machinery already exists, validated against the C++ (<=5e-9 rel. dev.), in
`/Users/baltabay/Desktop/gw-wl-emulator/scripts/convergence/bias_field_prototype.py`:
class `Cosmo` (grids, P(k), sigma(M), HMF `HMF0`, bias `biaslist`, NFW tables,
`find_kappathr`, `sigmakappaW`, `cell_tables`), class `LOSField` (pencil-projected
P_1D, `segment_corr`), helper `_localize_jz`. READ THAT FILE FIRST. Import from it
(same directory: `sys.path.insert(0, str(Path(__file__).resolve().parent))`).

Run with `/Users/baltabay/miniforge3/envs/test/bin/python` (Python 3.12; numpy/scipy
present). NO gwlensing / build/ needed — everything here is the pure-Python port.

## Hard constraints

- CREATE exactly one file: `scripts/convergence/bias_field_joint.py`. Do NOT modify
  any existing file.
- Follow the style of bias_field_prototype.py: module docstring with context,
  constants at top, argparse stages, `print(..., flush=True)` progress lines.
- Outputs: `data/results/bias_field_joint/sizing.npz` and
  `data/results/bias_field_joint/report.md` (create the directory).
- Default config: Nz=100, NM=100, fiducial cosmology (the prototype's H, OM, S8),
  kappa_thr from the fixed-<N>=100 rule: `kt = C.find_kappathr(zs, 100)`.
- ZS_LIST = [0.2, 1.0, 5.0, 10.0]. CLI: `stage` in {sizing, report, all}; optional
  `--zs 1.0` to restrict (for smoke tests).

## Physics spec — formulas (implement EXACTLY these)

### 1. Extended bias-weighted mass integrals (sum-rule bookkeeping)

Extended mass grid: `Mext = np.exp(np.linspace(np.log(1e3), np.log(1e17), 240))`,
`dlogMext = log-spacing`. For each M: `s, ds = C._sigma_smoothk(M, C.deltaH8)`
(returns sigma and d sigma/dM). Then, vectorized over the code's z grid `C.zlist`:

    dn/dlnM (z, M) = -C.rhoM0 * Cosmo._pFC(C.deltac(z), s**2) * 2 * s * ds     [>0]

(the same recipe as `Cosmo._build_hmf_bias_nfw`, extended down to M=1e3).

Bias, two variants (p=0.3 both):
- `b075(z, s)` = `C.halobias(z, s)` — the code's SMT01 bias with q=0.75;
- `b080(z, s)` — the SAME formula but q=0.8 (PBS-consistent with pFC's q=0.8):
  `qnu2 = 0.8*(C.deltac(z)/s)**2; 1 + (qnu2-1)/DELTAC0 + 2*0.3/(DELTAC0*(1+qnu2**0.3))`
  (import DELTAC0 from the prototype or recompute `3/5*(3*pi/2)**(2/3)`).

Integrals (per z on C.zlist):

    I_m(z)      = sum_j  M_j * dn/dlnM_j * dlogMext / C.rhoM0          # mass fraction in halos >= 1e3
    I_b075(z)   = sum_j  M_j * dn/dlnM_j * b075_j * dlogMext / C.rhoM0
    I_b080(z)   = same with b080

Also compute I_m with lower cut 1e5 instead of 1e3 (mask the sum) to report floor
sensitivity. Sum-rule check: as Mlo -> 0 and with the PBS-consistent bias,
I_b/I_m -> 1 and I_m -> 1. Report I_m, I_b075/I_m, I_b080/I_m at z in
{0.2, 1, 5, 10} (nearest zlist nodes). Assert 0.5 < I_m(z) < 1.1 for all z < 10.

### 2. Per-shell amplitude vectors at each zs

Shells: all jz with 1 <= jz and C.zlist[jz] < zs. Per shell:

    dz_jz    = zlist[jz] - zlist[jz-1]
    dchi_jz  = CLIGHT * dz_jz / C.Hz(zl_jz)                       # comoving path, CLIGHT=306.535
    dktot_jz = dchi_jz * C.rhoM0 * (1 + zl_jz)**2 / C.Sigmacf(zs, zl_jz)

dktot is the mean convergence rate of ALL matter in the shell (each halo's total
projected mass integral is M/Sigma_c exactly, so the full-population mean rate is
rho_M0 (1+z)^2 dchi / Sigma_c — mass conservation, no profile needed).
Sanity print: `sum(dktot)` per zs (the filled-beam mean convergence; order 0.01–0.5).

From `T = C.cell_tables(zs, kt)`: per explicit cell i, w_i = T['barN']*T['kbar']
(the cell's contribution to the mean explicit kappa) and b_i =
`C.biaslist[T['jz'], T['jM']]` (bias WITHOUT the Dg factor; T['bDg'] has Dg folded
in — don't double-count). Aggregate onto shells:

    We_jz  = sum_{i in shell jz} w_i                 # explicit mean-kappa rate
    Web_jz = sum_{i in shell jz} w_i * b_i           # bias-weighted

Amplitude vectors (Dg = C.Dg(zl_jz); I_b interpolated onto zl_jz — it's already on
zlist so just index):

    a_e[jz] = Dg * Web_jz                            # explicit count-modulation arm
    a_w[jz] = Dg * (dktot_jz * I_b075(zl_jz) - Web_jz)   # clustered weak arm (sum rule)
    a_w080[jz] = same with I_b080                    # PBS-consistent variant

a_w >= 0 must hold (explicit clipped mass is a subset of all halo mass): assert
a_w > -1e-12, clip tiny negatives to 0 with a warning. Also report the "kappa-weighted
explicit capture fraction" f_exp(zs) = sum(We)/sum(dktot) and the bias-weighted one
sum(Web)/sum(dktot*I_b075) — diagnostics of how much of the matter budget the explicit
layer carries.

### 3. Clustered variance budget

Segment covariance over ALL weak shells (not only those containing explicit cells):

    corr, Cov = fld.segment_corr(C.dc(zlist[jz_arr - 1]), C.dc(zlist[jz_arr]))

with `fld = LOSField(C)` built ONCE (it is zs/Nz-independent at fixed cosmology).
Cov is the z=0 pencil segment-average covariance; the Dg factors are already inside
the a vectors. Then:

    S_ee = a_e @ Cov @ a_e        # count-modulation clustered variance (linearized)
    S_ww = a_w @ Cov @ a_w        # clustered weak variance      <-- the new object
    S_ew = a_e @ Cov @ a_w        # cross term (the Cov(kappa_W, N_h) piece)
    S_tot = S_ee + 2*S_ew + S_ww  # = (a_e + a_w) @ Cov @ (a_e + a_w); assert equal to 1e-10 rel

plus the S_ww/S_ew variants with a_w080.

### 4. Poisson references (denominators)

    Var_e_pois = sum_i T['barN'] * T['k2bar']        # explicit-layer Poisson variance (Campbell)
    sigW       = C.sigmakappaW(zs, kt);  Var_w_pois = sigW**2
    Var_tot    = Var_e_pois + Var_w_pois

### 5. Report (report.md, one row per zs) — all of:

kt, <N> = sum(barN), Var_e_pois, Var_w_pois, S_ee, S_ww, 2*S_ew, S_tot,
rho_ew = S_ew/sqrt(S_ee*S_ww), and the headline ratios:
S_ww/Var_w_pois, (S_ww + 2*S_ew)/Var_tot, S_tot/Var_tot, S_ee/Var_tot.
Second table: the sum-rule/bookkeeping numbers of §1 (I_m at Mlo=1e3 vs 1e5,
I_b075/I_m, I_b080/I_m at z = 0.2/1/5/10) and f_exp(zs).
Third table: a_w-variant sensitivity (q=0.75 vs q=0.8 ratios of S_ww, S_ew).
Save EVERYTHING (per-zs shell arrays a_e, a_w, dktot, We, Web, scalars) to sizing.npz
with keys like `z1_a_e`, `z1_S_ww`, ... (zs formatted `z{zs:g}`).

## Numerical cautions

- `C.cell_tables` returns arrays over nonzero cells with global `jz`; build the shell
  aggregation with `np.bincount(jz, weights=..., minlength=Nz)` then slice to the
  shell index array.
- The Cov from `segment_corr` is PSD by construction; no regularization needed for
  quadratic forms (no Cholesky here).
- `_sigma_smoothk` is a per-M Python loop (~240 calls, each a 1000-pt trapezoid) —
  fine, but cache the extended (s, ds) table once (M-grid is z-independent).
- Everything at fixed cosmology: build ONE `Cosmo(Nz=100)` and ONE `LOSField`.
- Runtime target: well under 2 min per zs (sigmakappaW's annulus loop dominates).

## Acceptance (do these before declaring done)

1. Smoke run: `/Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/bias_field_joint.py all --zs 1.0`
   completes without error; paste its stdout in your final answer.
2. The §3 assert (S_tot decomposition) passes.
3. a_w >= 0 on every shell at every zs run.
4. I_m(z=1, Mlo=1e3) in (0.5, 1.1) and I_b080/I_m closer to 1 than I_b075/I_m at z=1
   (the PBS-consistent bias should satisfy the sum rule better) — if NOT, do not
   "fix" it, just print both and flag it in the report; that itself is a finding.
5. Full run over all four zs; paste the report.md content in your final answer.
