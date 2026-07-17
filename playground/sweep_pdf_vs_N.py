#!/usr/bin/env python3
"""ln(mu) PDF vs explicit-halo count <N>: empirical non-Gaussianity check.

Companion MC to the analytic cumulant knob (weak_band_cumulants.cpp): arms at
target <N> = 300..0.1 via custom_kappathr (inverted from nexp_vs_kthr_zs1.txt),
kappathr_flat = -1 so the ABSOLUTE background floor stays at the production
kappa_min = 1.28e-7 for every arm — only the split moves. Production MC
halo-only (filaments/ell/subhalo OFF), bias_model=1 (Rperp 8441), bias_weak=True.

lnmu built from raw kappa/gamma with the ROBUST anchor (batch mean of kappa over
the kappa<=1 core — sample_lnmu's empirical full-batch anchor is monster-ray
noise, CLAUDE.md item 12): lnmu = -ln|(1-kappa')^2 - gamma^2|.

Writes playground/pdf_vs_N_zs1_seed<seedbase>.npz (float32 lnmu per arm).
Run (repo root, test env):
  python playground/sweep_pdf_vs_N.py [seedbase=0] [nsamples=30000]
Launch seedbases 0..7 in parallel for the 240k/arm ensemble.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "build")
import gwlensing  # noqa: E402

SEEDBASE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
ZS = 1.0
RPERP = 8441.0
N_TARGETS = np.array([300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1])

# invert <N>(kappa_thr) from the precomputed table (monotone below the collapse)
nx = np.loadtxt("playground/nexp_vs_kthr_zs1.txt")
m = (nx[:, 1] > 1e-3) & (nx[:, 0] <= 10.0)
lk, lN = np.log(nx[m, 0]), np.log(nx[m, 1])
kthr = np.exp(np.interp(np.log(N_TARGETS), lN[::-1], lk[::-1]))
seed0 = 20261100 + 100000 * SEEDBASE
OUT = Path(f"playground/pdf_vs_N_zs1_seed{SEEDBASE}.npz")

save = {"N_targets": N_TARGETS, "kthr": kthr, "zs": ZS, "nsamples": NSAMP,
        "rperp": RPERP}
for i, (Nt, kt) in enumerate(zip(N_TARGETS, kthr)):
    t0 = time.time()
    r = gwlensing.sample_lensing_raw_ml(
        z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
        nsamples=NSAMP, seed=seed0 + i,
        filaments=False, bias=True, ell=False, subhalo=False,
        bias_model=1, bias_Rperp=RPERP, bias_weak=True,
        custom_kappathr=float(kt))
    kap = np.asarray(r["kappa"])
    gam2 = np.asarray(r["gamma1"]) ** 2 + np.asarray(r["gamma2"]) ** 2
    anchor = kap[kap <= 1.0].mean()
    kc = kap - anchor
    lnmu = -np.log(np.abs((1.0 - kc) ** 2 - gam2))
    save[f"lnmu_{i}"] = lnmu.astype(np.float32)
    save[f"kmax_{i}"] = kap.max()
    core = lnmu[kap <= 1.0]
    print(f"N={Nt:7.1f} kt={kt:.3e}  sig(lnmu|k<=1)={core.std():.5f} "
          f"kmax={kap.max():.2f}  [{time.time()-t0:.0f}s]", flush=True)

np.savez(OUT, **save)
print("wrote", OUT)
