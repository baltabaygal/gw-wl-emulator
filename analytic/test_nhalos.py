#!/usr/bin/env python
"""Is the z_s = 5 analytic-vs-MC residual caused by the engine's <N> = 100
explicit/Gaussian split?

The engine renders only halos above kappa_thr explicitly and replaces the whole
sub-threshold band by ONE Gaussian of the matching variance.  <N> = 100 sets
kappa_thr.  If the residual came from that approximation it must shrink as
<N> grows (kappa_thr falls, the Gaussian stand-in shrinks toward nothing).

Run analytic/run_mc.py --zs 5.0 --Nhalos {300,1000} --tag _N{300,1000} first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "build"))
import sgl
import gwlensing as gw
from compare_pdf import (mc_lnmu, source_plane_hist, analytic_curve,
                         weighted_stats, sigma_DL_mc, jsd)

ZS = 5.0
ARMS = [(100, "", "0.25"), (300, "_N300", "tab:orange"),
        (1000, "_N1000", "tab:green")]


def main():
    R = analytic_curve(ZS)
    Rs = sgl.tilt_source(R)
    an_var = sgl.moments(Rs)["var"]
    an_sdl = sgl.sigma_DL_over_DL(Rs)

    edges = np.linspace(-0.35, 0.55, 161)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    _, Pa = sgl.P_of_xi(Rs, xi_out=ctr)

    fig, (ax, ar) = plt.subplots(2, 1, figsize=(6.4, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.3, 1]))
    ax.plot(ctr, Pa, "-", lw=2.2, color="crimson", zorder=5,
            label="semi-analytic (no threshold)")
    print(f"analytic: Var = {an_var:.5f}   sigma_DL = {an_sdl:.5f}")
    print()
    print("   <N>   kappa_thr    Var MC   ratio   sigma_DL  ratio     JSD"
          "     body|d|")
    for N, tag, col in ARMS:
        lnmu, _ = mc_lnmu(HERE / "data" / f"mc_halo_only_zs{ZS:g}{tag}.npz")
        kt = gw.get_kappa_threshold(z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
                                    Nhalos=N)
        pdf, err = source_plane_hist(lnmu, edges)
        st, sdl = weighted_stats(lnmu), sigma_DL_mc(lnmu)
        g = (pdf > 0) & (Pa > 0)
        wt = (pdf * width)[g]
        bd = float(np.sum(wt * np.abs(pdf[g] / Pa[g] - 1)) / wt.sum())
        print(f"  {N:5d}  {kt:.3e}  {st['var']:.5f}  {st['var']/an_var:.3f}"
              f"   {sdl:.5f}  {sdl/an_sdl:.3f}  {jsd(pdf, Pa, width):.2e}"
              f"   {100*bd:5.1f}%")
        ax.plot(ctr, pdf, "o", ms=2.4, color=col, alpha=0.85,
                label=rf"MC, $\langle N\rangle$={N} "
                      rf"($\kappa_{{\rm thr}}$={kt:.1e})")
        ar.plot(ctr[g], pdf[g] / Pa[g], "o", ms=2.4, color=col, alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylim(3e-3, 5)
    ax.set_ylabel(r"$P(\ln\mu)$  (source plane)")
    ax.set_title(rf"$z_s={ZS:g}$: the residual is NOT the "
                 rf"$\langle N\rangle$ threshold split", fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    ar.axhline(1.0, color="crimson", lw=1.2)
    ar.set_ylim(0.6, 1.4)
    ar.set_xlabel(r"$\ln\mu$")
    ar.set_ylabel("MC / analytic")
    ar.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    out = HERE / "figures" / "nhalos_convergence_zs5.png"
    fig.savefig(out, dpi=160)
    print(f"\n[fig] {out}")


if __name__ == "__main__":
    main()
