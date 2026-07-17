#!/usr/bin/env python3
"""Expected explicit-halo count <N>(kappa_thr) on the quarter-decade lattice.

The x-axis mapping for the <N> version of the variance-partition figure:
<N> = NhfNFW(zs, kappa_thr) via gwlensing.get_expected_halo_count (fiducial
cosmology, default grid). Monotone decreasing in kappa_thr; <N> = 100 at the
production threshold by construction. Writes playground/nexp_vs_kthr_zs<zs>.txt
(columns: kappa_thr  Nexp).

Run (repo root, test env):
  python playground/compute_nexp_vs_kthr.py [zs=1]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "build")
import gwlensing  # noqa: E402

ZS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = Path(f"playground/nexp_vs_kthr_zs{ZS:g}.txt")

kthr = 10.0 ** (-8.0 + 11.0 * np.arange(45) / 44.0)   # probe lattice 1e-8..1e3

lines = ["# kappa_thr   Nexp   (zs=%g, fiducial cosmology, default grid)" % ZS]
for kt in kthr:
    t0 = time.time()
    n = gwlensing.get_expected_halo_count(ZS, 0.674, 0.315, 0.811, float(kt))
    lines.append(f"{kt:.6e} {n:.8e}")
    print(f"kt={kt:.3e}  <N>={n:.6e}  [{time.time()-t0:.1f}s]", flush=True)

OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
