#!/usr/bin/env python
"""Save raw κ, γ² for the sign-flip artifact check. Usage: OUT ZS ARM SEED N."""
import sys, os, numpy as np
sys.path.insert(0, 'build'); import gwlensing  # noqa
OUT, ZS, ARM, SEED, N = sys.argv[1:6]; ZS, SEED, N = float(ZS), int(SEED), int(N)
SKW = dict(subhalo=True, subhalo_model=3, subhalo_factor=1e-2)
BKW = dict(bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True)
kw = {"base": {}, "sub": SKW, "bias": BKW, "full": {**BKW, **SKW}}[ARM]
r = gwlensing.sample_lensing_raw_ml(ZS, 0.674, 0.315, 0.811, N, SEED, **kw)
kap = np.asarray(r["kappa"]); g2 = np.asarray(r["gamma1"])**2 + np.asarray(r["gamma2"])**2
np.savez_compressed(OUT, zs=ZS, arm=ARM, seed=SEED,
                    kappa=kap.astype(np.float32), g2=g2.astype(np.float32))
print("done", os.path.basename(OUT))
