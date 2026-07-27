#!/usr/bin/env python
"""Worker for the clustering/composition arms (#2/#3). One config, one ARM.

Usage: gen_worker2.py OUT ZS OM S8 H N SEED ARM
ARM in {base, sub, bias, full}.  base/sub duplicate the original cascade arms
(sub0/sub1) for completeness; #2/#3 normally only regenerate bias & full and
reuse the existing base/sub shards.  Clustering = production config per CLAUDE.md
(2026-07-20): bias_model=1, spherical top-hat window, R_s=20 Mpc, weak arm on.
"""
import sys, os, numpy as np
sys.path.insert(0, 'build')
import gwlensing  # noqa: E402

OUT, ZS, OM, S8, H, N, SEED, ARM = sys.argv[1:9]
ZS, OM, S8, H = float(ZS), float(OM), float(S8), float(H)
N, SEED = int(N), int(SEED)

SKW = dict(subhalo=True, subhalo_model=3, subhalo_factor=1e-2)
BKW = dict(bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True)
ARMS = {"base": {}, "sub": SKW, "bias": BKW, "full": {**BKW, **SKW}}
kw = ARMS[ARM]

x = np.asarray(gwlensing.sample_lnmu(ZS, OM, S8, H, N, SEED, **kw), dtype=np.float64)
np.savez_compressed(OUT, lnmu=x.astype(np.float32), zs=ZS, Om=OM, s8=S8, h=H,
                    N=N, seed=SEED, arm=ARM, std=x.std())
print("done", os.path.basename(OUT), "n=%d std=%.5f" % (len(x), x.std()))
