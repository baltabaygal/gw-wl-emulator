"""Fast (non-MC) arms of the Mmin / Nz convergence studies.

Arm A (Mmin): sigma_W(Mmin) on a dense log grid -> per-decade variance
contribution d(sigma_W^2)/dlnMmin by finite difference -> local power-law
exponent alpha(M). alpha > 0 at low M means the background-variance integral
converges as Mmin -> 0, and the extrapolated residual at the default
Mmin = 1e7 is the physics error the MC arm must resolve.
kappa_thr is held at the per-z_s default-rule value so the curve isolates
the Mmin dependence; kappa_thr(Mmin) itself is recorded separately.
NM controls at the lowest Mmin separate M-grid coarsening (NM=100 fixed
while the range widens) from real physics.

Arm B (Nz): kappa_thr(Nz) under the fixed-<N>=100 rule, <N>(Nz) at the
frozen default threshold, and sigma_W(Nz) both at the rule's own moving
threshold (production coupling) and at the frozen one (pure quadrature).
Right-edge shell rule -> expect ~1/Nz convergence; Richardson-extrapolate
to estimate the Nz=100 discretization error and pick the MC truth Nz.

Output: data/results/convergence_analytic/analytic_arms.npz + stdout tables.
Runtime: minutes (helper calls rebuild lookup tables each time).
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, 'build')
import gwlensing as gw

H, OM, S8 = 0.674, 0.315, 0.811
NHALOS = 100
ZS_LIST = [0.2, 1.0, 5.0, 10.0]

OUTDIR = 'data/results/convergence_analytic'
os.makedirs(OUTDIR, exist_ok=True)

out = {}
t00 = time.time()

# ---------------- Arm A: Mmin ----------------
# dense grid for the derivative + the MC-arm grid points
MMIN_GRID = np.logspace(3, 10, 15)          # 1e3 .. 1e10, half-decade steps
out['mmin_grid'] = MMIN_GRID

kthr_default = {}
for zs in ZS_LIST:
    t0 = time.time()
    kthr = gw.get_kappa_threshold(zs, H, OM, S8, NHALOS)   # default Mmin/NM/Nz
    kthr_default[zs] = kthr
    sig = np.array([gw.get_sigma_background(zs, H, OM, S8, kthr, Mmin=m)
                    for m in MMIN_GRID])
    # does the fixed-<N> threshold itself move with Mmin?
    kthr_vs_m = np.array([gw.get_kappa_threshold(zs, H, OM, S8, NHALOS, Mmin=m)
                          for m in MMIN_GRID])
    out[f'sigW_mmin_z{zs}'] = sig
    out[f'kthr_mmin_z{zs}'] = kthr_vs_m
    print(f'[Mmin] z_s={zs}: kthr_default={kthr:.4e}, '
          f'sigW(1e7)={sig[np.argmin(np.abs(MMIN_GRID-1e7))]:.5e}  '
          f'({time.time()-t0:.1f}s)', flush=True)

# NM controls at the widest range (Mmin=1e4) and default range
for zs in [1.0, 10.0]:
    row = {}
    for mmin in (1e4, 1e7):
        for nm in (100, 200, 400):
            row[(mmin, nm)] = gw.get_sigma_background(zs, H, OM, S8,
                                                      kthr_default[zs],
                                                      Mmin=mmin, NM=nm)
    out[f'sigW_nmctl_z{zs}'] = np.array(
        [[row[(m, n)] for n in (100, 200, 400)] for m in (1e4, 1e7)])
    print(f'[Mmin/NM ctl] z_s={zs}: {[f"{v:.6e}" for v in row.values()]}',
          flush=True)

# per-decade contribution + local exponent, per z_s
lnM = np.log(MMIN_GRID)
for zs in ZS_LIST:
    var = out[f'sigW_mmin_z{zs}']**2
    dvar = -np.gradient(var, lnM)            # contribution density at M=Mmin
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.gradient(np.log(np.abs(dvar) + 1e-300), lnM)
    out[f'dvar_dlnM_z{zs}'] = dvar
    out[f'alpha_z{zs}'] = alpha

# ---------------- Arm B: Nz ----------------
NZ_GRID = np.array([25, 50, 100, 200, 400, 800, 1600])
out['nz_grid'] = NZ_GRID

for zs in ZS_LIST:
    t0 = time.time()
    kthr_rule = np.array([gw.get_kappa_threshold(zs, H, OM, S8, NHALOS, Nz=int(n))
                          for n in NZ_GRID])
    kfix = kthr_default[zs]
    nexp = np.array([gw.get_expected_halo_count(zs, H, OM, S8, kfix, Nz=int(n))
                     for n in NZ_GRID])
    sig_fix = np.array([gw.get_sigma_background(zs, H, OM, S8, kfix, Nz=int(n))
                        for n in NZ_GRID])
    sig_rule = np.array([gw.get_sigma_background(zs, H, OM, S8, kthr_rule[i],
                                                 Nz=int(n))
                         for i, n in enumerate(NZ_GRID)])
    out[f'kthr_nz_z{zs}'] = kthr_rule
    out[f'nexp_nz_z{zs}'] = nexp
    out[f'sigW_nz_fixkthr_z{zs}'] = sig_fix
    out[f'sigW_nz_rulekthr_z{zs}'] = sig_rule
    print(f'[Nz] z_s={zs}: kthr(25..1600) = '
          + ' '.join(f'{k:.4e}' for k in kthr_rule)
          + f'  ({time.time()-t0:.1f}s)', flush=True)

# Richardson (assume first order, right-edge rule): f_inf ~ 2 f(2n) - f(n)
def rich(f_n, f_2n):
    return 2.0 * f_2n - f_n

summary = {}
for zs in ZS_LIST:
    k = out[f'kthr_nz_z{zs}']
    # use the two finest levels for the extrapolant
    kinf = rich(k[-2], k[-1])
    err100 = k[NZ_GRID.tolist().index(100)] / kinf - 1.0
    # convergence order from the last three levels
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.log2(abs((k[-3] - k[-2]) / (k[-2] - k[-1])))
    s = out[f'sigW_nz_rulekthr_z{zs}']
    sinf = rich(s[-2], s[-1])
    serr100 = s[NZ_GRID.tolist().index(100)] / sinf - 1.0
    summary[zs] = dict(kthr_inf=kinf, kthr_rel_err_nz100=err100, order=p,
                       sigW_inf=sinf, sigW_rel_err_nz100=serr100)
    print(f'[Nz summary] z_s={zs}: kthr_inf={kinf:.5e}, '
          f'kthr err@Nz100={err100:+.3%}, order~{p:.2f}, '
          f'sigW err@Nz100={serr100:+.3%}', flush=True)

np.savez(os.path.join(OUTDIR, 'analytic_arms.npz'), **{
    k: v for k, v in out.items()})
with open(os.path.join(OUTDIR, 'nz_summary.json'), 'w') as f:
    json.dump({str(z): {k: float(v) for k, v in d.items()}
               for z, d in summary.items()}, f, indent=2)
print(f'TOTAL {time.time()-t00:.1f}s -> {OUTDIR}/analytic_arms.npz', flush=True)
