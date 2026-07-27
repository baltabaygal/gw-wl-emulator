#!/usr/bin/env python
"""Engine-vs-analytic checkpoint comparison (2026-07-24).

Reads the printed output of playground/engine_checkpoint_probe (run it first,
or pipe its stdout to a file) and evaluates the spec-validated analytic module
(scripts/comparisons/analytic_pdf.py) at the SAME (M, z) points. First miss
identifies the source of the unexplained +10%-in-Var sigma_DL residual.

Usage:
  ./playground/engine_checkpoint_probe > tmp/engine_checkpoints.txt
  python scripts/comparisons/engine_checkpoint_compare.py tmp/engine_checkpoints.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analytic_pdf as A

def parse(path):
    d, hmf, prof = {}, [], []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "HMF_grid":
            hmf.append((float(t[2]), float(t[4]), float(t[6])))
        elif t[0] == "profile":
            prof.append((float(t[2]), float(t[4]), float(t[6])))
        elif t[0] == "NFW_grid":
            d["zl"], d["M"] = float(t[2]), float(t[4])
        elif len(t) == 2:
            d[t[0]] = float(t[1])
    return d, hmf, prof

def main(path):
    d, hmf, prof = parse(path)
    print("== amplitude / window ==")
    print(f"  engine tophat sigma8       {d['sigma8_TOPHAT_engine']:.4f}  vs analytic 0.8110"
          f"  -> P(k) ratio {(d['sigma8_TOPHAT_engine']/0.811)**2:.4f}")
    print(f"  engine Ws sigma(1e12)      {d['sigma_1e12_engineWs']:.4f}  vs analytic tophat"
          f" {A.sigmaM(1e12):.4f}  ratio {d['sigma_1e12_engineWs']/A.sigmaM(1e12):.4f}")
    print(f"  engine tophat sigma(1e12)  {d['sigma_1e12_TOPHAT_engine']:.4f}"
          f"  (amplitude+wiggles only)   ratio {d['sigma_1e12_TOPHAT_engine']/A.sigmaM(1e12):.4f}")

    print("== growth ==")
    for z in (0.5, 1.0, 2.0):
        eng = d[f"Dg_{z:g}"]
        ana = A.D(z)
        print(f"  Dg({z:g})  engine {eng:.4f}  exact {ana:.4f}  ratio {eng/ana:.4f}")

    print("== halo mass function (engine grid points; Mpc^-3 per lnM) ==")
    for z, M, eng in hmf:
        ana = A.dndlnM(M, z)
        print(f"  z={z:.3f} M={M:.3e}  engine {eng:.3e}  analytic {ana:.3e}"
              f"  ratio {eng/ana:.3f}")

    print("== NFW structure at engine grid host (zs=2) ==")
    M, zl = d["M"], d["zl"]
    C, rs, ks, fC = A.nfw_params(M, zl, 2.0)
    Scr = A.Sigma_cr(zl, 2.0)
    print(f"  host M={M:.3e} zl={zl:.3f}")
    print(f"  c200     engine {d['NFW_c200']:.3f}   analytic {C:.3f}   ratio {d['NFW_c200']/C:.4f}")
    print(f"  rs [kpc] engine {d['NFW_rs_kpc']:.2f}  analytic {rs*1e3:.2f}  ratio {d['NFW_rs_kpc']/(rs*1e3):.4f}")
    print(f"  Sigma_c  engine {d['Sigmac_MsunMpc2']:.4e}  analytic {Scr:.4e}  ratio {d['Sigmac_MsunMpc2']/Scr:.4f}")
    print(f"  kappa_s  engine {d['kappa_s']:.5f}  analytic {ks:.5f}  ratio {d['kappa_s']/ks:.4f}")

    print("== projected profile shape (normalized by the ENGINE kappa_s) ==")
    k0 = d["kappa_s"]
    for x, kap, gam in prof:
        kw = 2*k0*A._F(np.array([x]))[0]
        gw = 4*k0*A._hfun(np.array([x]))[0]/x**2 - kw
        print(f"  x={x:5.2f}  kappa engine {kap:.6e} WB {kw:.6e} ratio {kap/kw:.5f}"
              f"   gamma engine {gam:.6e} WB {gw:.6e} ratio {gam/gw:.5f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tmp/engine_checkpoints.txt")
