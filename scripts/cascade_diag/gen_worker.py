#!/usr/bin/env python
"""Cascade-residual diagnostic worker: generate ONE paired arm of lnmu samples.

Usage:
    gen_worker.py OUT ZS OM S8 H N SEED SUB

SUB = 0 (halo-only, arm A)  or  1 (halo+sub model 3, factor 1e-2, arm B).
Writes OUT.npz with the raw lnmu array (float32) + config.  Run in the `test`
conda env (python 3.12) from the repo root.
"""
import sys, os, numpy as np
sys.path.insert(0, 'build')
import gwlensing  # noqa: E402

OUT, ZS, OM, S8, H, N, SEED, SUB = sys.argv[1:9]
ZS, OM, S8, H = float(ZS), float(OM), float(S8), float(H)
N, SEED, SUB = int(N), int(SEED), int(SUB)

kw = dict(subhalo=bool(SUB))
if SUB:
    kw.update(subhalo_model=3, subhalo_factor=1e-2)

x = gwlensing.sample_lnmu(ZS, OM, S8, H, N, SEED, **kw)
x = np.asarray(x, dtype=np.float64)
np.savez_compressed(
    OUT,
    lnmu=x.astype(np.float32),
    zs=ZS, Om=OM, s8=S8, h=H, N=N, seed=SEED, sub=SUB,
    mean=x.mean(), std=x.std(), mn=x.min(), mx=x.max(),
)
print("done", os.path.basename(OUT), "n=%d mean=%.5f std=%.5f" % (len(x), x.mean(), x.std()))
