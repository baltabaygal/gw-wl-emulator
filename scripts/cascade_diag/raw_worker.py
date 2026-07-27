#!/usr/bin/env python
"""Certified-core σ via the raw κ field. One (zs, arm, seed).
Usage: raw_worker.py OUT ZS ARM SEED N
σ_core = std of anchored lnμ over the κ_tot≤1 subcritical core (production's
certified clip); σ_p999 = lnμ-percentile-trim std (the biased proxy) for contrast.
"""
import sys, os, numpy as np
sys.path.insert(0, 'build')
import gwlensing  # noqa

OUT, ZS, ARM, SEED, N = sys.argv[1:6]
ZS, SEED, N = float(ZS), int(SEED), int(N)
SKW = dict(subhalo=True, subhalo_model=3, subhalo_factor=1e-2)
BKW = dict(bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True)
kw = {"base": {}, "sub": SKW, "bias": BKW, "full": {**BKW, **SKW}}[ARM]

r = gwlensing.sample_lensing_raw_ml(ZS, 0.315, 0.811, 0.674, N, SEED, **kw)
kap = np.asarray(r["kappa"]); g2 = np.asarray(r["gamma1"])**2 + np.asarray(r["gamma2"])**2
kap_a = kap - kap.mean()                      # anchor ⟨κ⟩→0 (matches sample_lnmu)
arg = (1.0 - kap_a)**2 - g2
good = arg > 0
lnmu = np.full(kap.shape, np.nan); lnmu[good] = -np.log(arg[good])
core = good & (kap <= 1.0)                     # certified κ_tot≤1 subcritical core
lo, hi = np.nanpercentile(lnmu, [0.1, 99.9])
ptrim = good & (lnmu >= lo) & (lnmu <= hi)
np.savez(OUT, zs=ZS, arm=ARM, seed=SEED,
         sig_core=lnmu[core].std(), n_core=int(core.sum()),
         sig_p999=lnmu[ptrim].std(), frac_super=float((kap > 1).mean()))
print("done", os.path.basename(OUT), "sig_core=%.5f n_core=%d fsuper=%.3f" % (
    lnmu[core].std(), int(core.sum()), float((kap > 1).mean())))
