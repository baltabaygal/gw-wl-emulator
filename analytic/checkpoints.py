#!/usr/bin/env python
"""Hold analytic/sgl.py to (a) the analytic_chain_spec.md checkpoints and
(b) the C++ engine checkpoints dumped by playground/engine_checkpoint_probe.

Run:  python analytic/checkpoints.py [tmp/engine_checkpoints.txt]

(a) proves the module is the same validated chain as
    scripts/comparisons/analytic_pdf.py; (b) shows which conventions still
    differ between the analytic chain and the engine, and by how much.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sgl


def spec_checkpoints():
    print("=" * 72)
    print("SPEC CHECKPOINTS  (analytic_chain_spec.md, tophat + no-wiggle)")
    print("=" * 72)
    cos = sgl.Cosmology(window="tophat", transfer="nowiggle")

    print("-- Sec.0 growth")
    print(f"   D(0.5)          = {cos.D(0.5):.4f}   spec 0.7689")

    print("-- Sec.1 sigma(M)")
    s12 = cos.sigmaM(1e12)
    print(f"   sigma(1e12,z=0) = {s12:.3f}    spec 2.223")
    print(f"   nu(1e12,z=0)    = {(cos.dc/s12)**2:.3f}    spec 0.575")

    print("-- Sec.3 NFW (M=1e12, zl=0.5, zs=2)")
    C, rs, ks, fC = cos.nfw_params(1e12, 0.5, 2.0)
    print(f"   C = {C:.2f} (6.76)   rs = {rs*1e3:.2f} kpc (25.98)   "
          f"ks = {ks:.4f} (0.0495)")

    print("-- Sec.5 R(xi) at zs = 2")
    R2 = sgl.R_of_xi(cos, 2.0)
    sl = sgl.slope_R(R2)
    tab = [(1e-4, 8.35e6, 2.02), (1e-3, 7.82e4, 2.06), (1e-2, 618, 2.11),
           (0.05, 15.1, 2.39), (0.1, 2.47, 2.69), (0.2, 0.323, 3.46),
           (0.4, 2.81e-2, 3.90), (0.8, 1.25e-3, 3.60)]
    print("     xi        R spec      R calc     ratio   slope spec  calc")
    for xiv, Rsp, ssp in tab:
        j = np.argmin(np.abs(sgl.XI - xiv))
        print(f"   {xiv:7.0e}  {Rsp:10.3e} {R2[j]:10.3e}  {R2[j]/Rsp:6.3f}"
              f"      {ssp:5.2f}   {sl[j]:5.2f}")

    print("-- Sec.6 sigma_DL/DL(zs)")
    spec = {0.5: 0.0114, 1: 0.0233, 2: 0.0412, 5: 0.0682, 10: 0.0848}
    print("     zs    spec     calc    ratio")
    for zs in (0.5, 1, 2, 5, 10):
        R = R2 if zs == 2 else sgl.R_of_xi(cos, zs)
        v = sgl.sigma_DL_over_DL(sgl.tilt_source(R))
        print(f"   {zs:5.1f}  {spec[zs]:.4f}  {v:.4f}   {v/spec[zs]:.3f}")

    print("-- Sec.7 inversion sanity (zs = 2, source plane)")
    Rs = sgl.tilt_source(R2)
    xi, P = sgl.P_of_xi(Rs)
    norm = np.trapezoid(P, xi)
    mean = np.trapezoid(P * xi, xi) / norm
    print(f"   int P dxi = {norm:.4f} (spec 1.0000)   <xi> = {mean:+.2e} "
          f"(compensated: 0)   min P = {P.min():+.2e}")
    var_inv = np.trapezoid(P * (xi - mean)**2, xi) / norm
    print(f"   Var from inversion {var_inv:.6f}  vs exact m2 "
          f"{sgl.moments(Rs)['var']:.6f}   ratio "
          f"{var_inv/sgl.moments(Rs)['var']:.4f}")
    return cos


def engine_checkpoints(path):
    p = Path(path)
    if not p.exists():
        print(f"\n[skip engine checkpoints: {p} not found -- build and run "
              f"playground/engine_checkpoint_probe first]")
        return
    d, hmf = {}, []
    for line in p.read_text().splitlines():
        t = line.split()
        if not t or t[0].startswith("#"):
            continue
        if t[0] == "HMF_grid":
            hmf.append((float(t[2]), float(t[4]), float(t[6])))
        elif t[0] == "NFW_grid":
            d["zl"], d["M"] = float(t[2]), float(t[4])
        elif len(t) == 2:
            d[t[0]] = float(t[1])

    print()
    print("=" * 72)
    print("ENGINE CHECKPOINTS  (C++ production cosmology, zero MC)")
    print("=" * 72)
    for win, trf, tag in (("tophat", "nowiggle", "spec  "),
                          ("smoothk", "eh98", "engine")):
        cos = sgl.Cosmology(window=win, transfer=trf)
        print(f"-- analytic mode {tag}  (window={win}, transfer={trf})")
        print(f"   sigma8 tophat  {cos.sigma_tophat8():.4f}"
              f"   engine {d['sigma8_TOPHAT_engine']:.4f}"
              f"   ratio {cos.sigma_tophat8()/d['sigma8_TOPHAT_engine']:.4f}")
        s12 = cos.sigmaM(1e12)
        print(f"   sigma(1e12)    {s12:.4f}"
              f"   engine(Ws) {d['sigma_1e12_engineWs']:.4f}"
              f"   ratio {s12/d['sigma_1e12_engineWs']:.4f}")
        for z, M, eng in hmf:
            ana = cos.dndlnM(M, z)
            print(f"   dn/dlnM z={z:.3f} M={M:.3e}   analytic {ana:.4e}"
                  f"   engine {eng:.4e}   ratio {ana/eng:.4f}")
        M, zl = d["M"], d["zl"]
        C, rs, ks, fC = cos.nfw_params(M, zl, 2.0)
        print(f"   c200 {C:.4f} vs {d['NFW_c200']:.4f}"
              f"   rs {rs*1e3:.3f} vs {d['NFW_rs_kpc']:.3f} kpc"
              f"   Sigma_cr ratio "
              f"{cos.Sigma_cr(zl,2.0)/d['Sigmac_MsunMpc2']:.4f}"
              f"   kappa_s ratio {ks/d['kappa_s']:.4f}")


if __name__ == "__main__":
    spec_checkpoints()
    engine_checkpoints(sys.argv[1] if len(sys.argv) > 1
                       else "tmp/engine_checkpoints.txt")
