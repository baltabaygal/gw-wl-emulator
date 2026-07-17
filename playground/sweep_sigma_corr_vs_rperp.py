#!/usr/bin/env python3
"""sigma_corr (weak-arm clustering component) vs bias_Rperp, z_s=1, bias era.

Production MC (halo-only) with bias_model=1 + bias_weak=True at the DEFAULT
threshold kappa_thr(<N>=100, z=1) = 1.2777e-4, where the weak arm is
corr-dominated (shot = 1.327e-3 analytic, R_perp-independent). Per R_perp,
per-ray split via kappa_weak; clip1 = kappa_tot <= 1 core mask (certified
estimator);  sigma_corr = sqrt(Var_w,clip1 - shot^2).

Grid: 17 points, 1 kpc .. 10 Mpc (quarter decades). The BiasField1D guard
(N_max = L/R_perp <= 5e6) admits R_perp = 1 kpc at z_s=1 (L ~ 3.5e6 kpc).

Run (repo root, test env):  python playground/sweep_sigma_corr_vs_rperp.py [seedbase=0]
Writes playground/sigma_corr_vs_rperp_zs1_seed<sb>.npz. Launch 8 seedbases in
parallel processes for the ensemble.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "build")
import gwlensing  # noqa: E402

SEEDBASE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
MODE = sys.argv[3] if len(sys.argv) > 3 else "main"
ZS = 1.0
SHOT = 1.327308e-03          # get_sigma_background at the default kappa_thr, z=1
if MODE == "main":
    rperp = 10.0 ** (0.25 * np.arange(17))          # 1 .. 1e4 kpc
    seed0 = 20260900 + 100000 * SEEDBASE
    OUT = Path(f"playground/sigma_corr_vs_rperp_zs{ZS:g}_seed{SEEDBASE}.npz")
elif MODE == "ext":
    # extension toward the no-bias limit: 10^4.25 .. 10^6 kpc (1 Gpc). Above
    # R_perp ~ L/4 (~0.9 Gpc) BiasField1D keeps a token Nmax=4 modes; the disk
    # window suppresses them regardless, so sigma_corr -> 0 (weak arm -> pure
    # Gaussian shot draw = the no-bias-correlation limit).
    rperp = 10.0 ** (4.25 + 0.25 * np.arange(8))    # 10^4.25 .. 1e6 kpc
    seed0 = 20260900 + 100000 * SEEDBASE + 100      # never overlaps main's i=0..16
    OUT = Path(f"playground/sigma_corr_vs_rperp_zs{ZS:g}_ext_seed{SEEDBASE}.npz")
else:
    raise SystemExit(f"unknown mode {MODE!r}")
rows, nclip = [], []
for i, rp in enumerate(rperp):
    t0 = time.time()
    r = gwlensing.sample_lensing_raw_ml(
        z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
        nsamples=NSAMP, seed=seed0 + i,
        filaments=False, bias=True, ell=False, subhalo=False,
        bias_model=1, bias_Rperp=float(rp), bias_weak=True)
    ktot = np.asarray(r["kappa"])
    kw = np.asarray(r["kappa_weak"])
    ks = ktot - kw
    m = ktot <= 1.0
    w, s = kw[m], ks[m]
    t = w + s
    rows.append((w.var(), s.var(), np.mean((w - w.mean()) * (s - s.mean())),
                 t.var(), w.mean(), s.mean()))
    nclip.append(int(m.sum()))
    corr = np.sqrt(max(rows[-1][0] - SHOT**2, 0.0))
    print(f"Rperp={rp:9.2f}  sig_w={np.sqrt(rows[-1][0]):.6f}  corr={corr:.6f}  "
          f"sig_t={np.sqrt(rows[-1][3]):.6f}  (n={nclip[-1]})  [{time.time()-t0:.0f}s]",
          flush=True)

np.savez(OUT, rperp=rperp, zs=ZS, nsamples=NSAMP, shot=SHOT,
         clip1=np.array(rows), nclip=np.array(nclip))
print("wrote", OUT)
