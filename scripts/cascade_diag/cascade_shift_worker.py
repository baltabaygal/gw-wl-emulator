#!/usr/bin/env python
"""Measure production-currency scalars per (arm, z) via the raw-κ sampler, at
fiducial θ. Saves anchored lnμ (float32) + the operator-currency shifts inputs:
  sig_core = std(anchored lnμ | κ≤1 subcritical core)   [production clipped-Var σ]
  lnS2/3/8 = ln P(μ>2,3,8)                               [POT survival anchors]
  edge     = q0.5% of lnμ                                [empty-beam wall]
  flux     = ln <1/μ> over [q0.5,q99.5] core            [flux target]
  mean, skew = clipped-core moments
Usage: cascade_shift_worker.py OUT ZS ARM SEED N
"""
import sys, os, numpy as np
sys.path.insert(0, 'build')
import gwlensing  # noqa

OUT, ZS, ARM, SEED, N = sys.argv[1:6]
ZS, SEED, N = float(ZS), int(SEED), int(N)
SKW = dict(subhalo=True, subhalo_model=3, subhalo_factor=1e-2)
BKW = dict(bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True)
kw = {"base": {}, "sub": SKW, "bias": BKW, "full": {**BKW, **SKW}}[ARM]

r = gwlensing.sample_lensing_raw_ml(ZS, 0.674, 0.315, 0.811, N, SEED, **kw)
kap = np.asarray(r["kappa"]); g2 = np.asarray(r["gamma1"])**2 + np.asarray(r["gamma2"])**2
kap_a = kap - kap.mean()                         # anchor ⟨κ⟩→0 (matches sample_lnmu)
arg = (1.0 - kap_a)**2 - g2
good = arg > 0
lnmu = -np.log(arg[good])
kap_g = kap[good]                                # unanchored κ for the core mask
mu = np.exp(lnmu)

core = kap_g <= 1.0
q05, q995 = np.percentile(lnmu, [0.5, 99.5])
cmask = (lnmu >= q05) & (lnmu <= q995)
lnS = {u: np.log(max((mu > u).mean(), 1e-9)) for u in (2.0, 3.0, 8.0)}
np.savez_compressed(
    OUT, zs=ZS, arm=ARM, seed=SEED, lnmu=lnmu.astype(np.float32),
    sig_core=lnmu[core].std(), n_core=int(core.sum()),
    lnS2=lnS[2.0], lnS3=lnS[3.0], lnS8=lnS[8.0],
    edge=q05, flux=np.log(np.mean(np.exp(-lnmu[cmask]))),
    mean=lnmu[cmask].mean(),
    skew=float(((lnmu[cmask] - lnmu[cmask].mean())**3).mean() / lnmu[cmask].std()**3),
)
print("done", os.path.basename(OUT), "sig_core=%.4f lnS3=%.3f edge=%.4f" % (
    lnmu[core].std(), lnS[3.0], q05))
