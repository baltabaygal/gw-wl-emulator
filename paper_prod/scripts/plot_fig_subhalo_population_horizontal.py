#!/usr/bin/env python3
"""
Horizontal (side-by-side) variant of plot_fig_subhalo_population.py.

Identical physics, curves, and styling to the two-row figure; the only
difference is layout: the two panels are placed left/right (1x2) instead of
top/bottom (2x1). See plot_fig_subhalo_population.py for the full description
of each panel and its inputs.

Run: python3 paper_prod/scripts/plot_fig_subhalo_population_horizontal.py
Outputs: paper_prod/plots/figures/fig_subhalo_population_horizontal.{pdf,png}
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "convergence"))

from bias_field_prototype import Cosmo, PI
from paper_prod.plot_style import apply_style

# reuse the physics/port + digitized-curve helpers from the vertical script
from plot_fig_subhalo_population import (
    ALPHA, BETA, OMEGA, M_FLOOR, S_M, WF, AF, Z_L, M_HOSTS, M_PYHALO,
    COL_OURS, COL_DH, COL_PYH,
    BOLSHOI_LOGX, BOLSHOI_LOGB, GREEN_LOGX, GREEN_LOGB, HAN16_GAMMA,
    diffhalos_dndlnpsi, bias_adopted, bias_springel08, gamma_inc,
    fs_gamma, guard_broken_latex,
)

OUT_DIR = REPO / "paper_prod" / "plots" / "figures"


def main():
    apply_style()
    guard_broken_latex()
    import matplotlib.pyplot as plt

    C = Cosmo(Nz=100)

    # Canvas is the full two-column text width (7.1"), so \includegraphics at
    # width=\linewidth inside a figure* renders 1:1 (no upscaling of panels or
    # fonts). Margins/wspace are chosen so each panel's axes box is IDENTICAL
    # to a single-column figure: single = (3.37 x 2.6)" with left=0.20,
    # right=0.95, bottom=0.16, top=0.92 -> axes box 2.529 x 1.976". Here
    # left=0.674"/7.1, right edge 1-0.1685"/7.1, wspace tuned to reproduce the
    # 2.529" axes width; height 2.6" with the same top/bottom gives 1.976".
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.6))
    fig.subplots_adjust(left=0.0949, right=0.9763, bottom=0.16, top=0.92,
                        wspace=0.473)

    psi = np.exp(np.linspace(np.log(1e-6), 0.0, 800))
    info = []
    for j, M in enumerate(M_HOSTS):
        fs, gam, zf, Ntau = fs_gamma(C, Z_L, M)
        pf = M_FLOOR / M
        dNdlnpsi = gam * psi ** ALPHA * np.exp(-BETA * psi ** OMEGA)
        lab = (rf"This work (JvdB14, bound), "
               rf"$f_{{\rm s}} = {fs:.2f}$")
        axa.plot(psi, dNdlnpsi, color=COL_OURS, label=lab)

        fsb = gam / (OMEGA * BETA ** S_M) * (
            gamma_inc(S_M, BETA * pf ** OMEGA) - gamma_inc(S_M, BETA))

        lnb = np.linspace(math.log(pf), 0.0, 20_000)
        psib = np.exp(lnb)
        f_peak = np.trapezoid(psib * diffhalos_dndlnpsi(math.log10(M), psib),
                              lnb)
        s_eff = fsb / f_peak
        psi_dh = psi[psi <= 0.5 * s_eff]
        axa.plot(psi_dh, diffhalos_dndlnpsi(math.log10(M), psi_dh / s_eff),
                 color=COL_DH, ls="dashed", lw=1.0, alpha=0.9)

        if pf > psi[0]:
            axa.plot(pf, gam * pf ** ALPHA * math.exp(-BETA * pf ** OMEGA),
                     "o", ms=3, color=COL_OURS)
        band = np.exp(np.linspace(math.log(pf), 0.0, 4000))
        Nsub = np.trapezoid(gam * band ** ALPHA
                            * np.exp(-BETA * band ** OMEGA), np.log(band))
        info.append((M, fs, fsb, zf, Ntau, Nsub, s_eff))

    # pyHalo bound masses (evolved), matched 1e13 host, deprojected per-host
    pyh = np.load(REPO / "data" / "pyhalo_vs_ours.npz")
    a_ap = math.pi * float(pyh["R_ap"]) ** 2
    ok = pyh["h_bnd"] > 0
    psi_py = pyh["ctr"][ok] / M_PYHALO
    dn_py = pyh["h_bnd"][ok] * a_ap / float(pyh["f_ap"])
    axa.plot(psi_py, dn_py, "s", ms=2.4, mfc="none", mew=0.7, color=COL_PYH)

    from matplotlib.lines import Line2D
    proxies = [
        Line2D([], [], color=COL_DH, ls="dashed", lw=1.0,
               label=rf"Diffhalos (peak$\to$bound, $s_{{\rm eff}}={s_eff:.2f}$)"),
        Line2D([], [], color=COL_PYH, marker="s", ms=2.4, mfc="none", mew=0.7,
               ls="none", label=r"pyHalo (bound)"),
    ]
    leg1 = axa.legend(fontsize=5.8, frameon=False, loc="lower left",
                      title=rf"$M = 10^{{13}}\,M_\odot$, $z = {Z_L:g}$",
                      title_fontsize=5.8)
    axa.add_artist(leg1)
    axa.legend(handles=proxies, fontsize=5.8, frameon=False,
               loc="upper right")
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(1e-4, 1.0)
    axa.set_ylim(1e-2, 1e3)
    axa.set_xlabel(r"$\psi = m/M$")
    axa.set_ylabel(r"${\rm d}N/{\rm d}\ln\psi$")

    # ---------------- panel (b): radial bias function, log-log
    x = np.logspace(-1.6, 0.0, 400)
    axb.plot(x, bias_adopted(x), color="C0", lw=1.4,
             label=r"$[1+(x/0.54)^{-5/2}]^{-1/2}$ (this work)")
    axb.plot(10.0 ** GREEN_LOGX, 10.0 ** GREEN_LOGB, color="C2", lw=1.0,
             label="Green+21 (withering$+$disruption)")
    axb.plot(10.0 ** BOLSHOI_LOGX, 10.0 ** BOLSHOI_LOGB, "s", ms=2.2,
             color="k", ls="none", label="Bolshoi (Klypin+11)")
    axb.plot(x, bias_springel08(x), color="C4", ls="dashdot", lw=1.0,
             label=r"Springel+08 (Aq-A-1, Einasto/NFW)")
    axb.plot(x, x ** HAN16_GAMMA, color="C3", ls="dotted", lw=1.2,
             label=rf"Han+16: $x^{{{HAN16_GAMMA:g}}}$ (Aq-A)")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlim(10 ** -1.6, 1.0)
    axb.set_ylim(10 ** -2.3, 1.6)
    axb.set_xlabel(r"$x = r/r_{200}$")
    axb.set_ylabel(r"$B(x) = n_{\rm sub}/n_{\rm host}$")
    axb.legend(fontsize=5.8, frameon=False, loc="upper left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig_subhalo_population_horizontal.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    print(f"wrote {out_pdf.relative_to(REPO)}")
    print(f"wrote {out_png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
