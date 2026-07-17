#!/usr/bin/env python3
"""Measured total sigma_kappa vs custom_kappathr under the correlated bias field
(bias_model=1, production Rperp) WITH the conditional weak arm (bias_weak=True).

The bias-era counterpart of sweep_sigma_total_vs_kthr.py: halo-only production MC
(filaments/ell/subhalo OFF, bias ON), floor-consistent background injection
(kappa_min held at 0.001*kappa_thr_default). If the weak arm's shot+correlation
bookkeeping is right, sigma_total is flat in kappa_thr at the full (shot +
clustering) variance — the analytic counterpart curves come from
playground/sigma_partition_bias_vs_kthr.cpp.

Appends one line per point to playground/sigma_total_vs_kthr_bias_zs<zs>.txt
(resumable: existing points skipped).

Columns: kappa_thr  sigma_total  relerr(Var)  nsamples
Run from the repo root with the test-env python:
  python playground/sweep_sigma_total_vs_kthr_bias.py [zs=1.0] [nsamples=100000] [stride=1]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "build")
import gwlensing  # noqa: E402

ZS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 1
RPERP = 8441.0  # production default (R_L(1e14 Msun), 2026-07-16 decision)
OUT = Path(f"playground/sigma_total_vs_kthr_bias_zs{ZS:g}.txt")

kthr = (10.0 ** (-5.0 + 8.0 * np.arange(33) / 32.0))[::STRIDE]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            done.add(float(line.split()[0]))
else:
    OUT.write_text("# bias_model=1 Rperp=%g bias_weak=True; halo-only\n"
                   "# kappa_thr   sigma_total   relerr(Var)   nsamples\n" % RPERP)

for i, kt in enumerate(kthr):
    if any(abs(np.log(kt / d)) < 1e-9 for d in done):
        print(f"skip {kt:.3e} (already done)", flush=True)
        continue
    t0 = time.time()
    r = gwlensing.sample_lensing_raw_ml(
        z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
        nsamples=NSAMP, seed=20260716 + i,
        filaments=False, bias=True, ell=False, subhalo=False,
        bias_model=1, bias_Rperp=RPERP, bias_weak=True,
        custom_kappathr=float(kt))
    k = np.asarray(r["kappa"])
    var = k.var()
    m4c = np.mean((k - k.mean()) ** 4)
    relerr = np.sqrt(max(m4c - var**2, 0.0) / k.size) / var  # heavy-tail error on Var
    with OUT.open("a") as f:
        f.write(f"{kt:.6e} {np.sqrt(var):.8e} {relerr:.4f} {k.size}\n")
    print(f"kt={kt:.3e}  sigma={np.sqrt(var):.6f}  relerr(Var)={relerr:.3f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

print("sweep complete ->", OUT)
