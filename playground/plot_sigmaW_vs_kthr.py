#!/usr/bin/env python3
"""Analytic background sigma_W vs kappa_thr: fixed-fraction floor vs fixed-absolute floor.

Reads playground/sigmaW_vs_kthr_zs1.txt (from sigmaW_vs_kthr.cpp) and, if present,
playground/sigma_explicit_vs_kthr_zs1.txt (MC-measured explicit-halo sigma from
sigma_explicit_vs_kthr.cpp). The fixed eps_floor=1e-3 curve reproduces the
turnover/collapse seen in the MC sweep plot (the absolute floor kappa_min =
eps*kappa_thr rises with the threshold, discarding real variance); holding
kappa_min fixed instead makes sigma_W plateau at the full Campbell variance, and
the measured explicit variance completes it in quadrature: sigma_explicit^2 +
sigma_W^2 = const. Writes plots/sigmaW_vs_kthr.{png,pdf}.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED, GREEN, ORANGE = "#2a78d6", "#e34948", "#2e9e62", "#e28425"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

kt, sA, sB = np.loadtxt("sigmaW_vs_kthr_zs1.txt", unpack=True)
plateau = sB[-1]

fig, ax = plt.subplots(figsize=(8.6, 6.2))
ax.plot(kt, sA, "-o", color=RED, ms=4, lw=1.7, zorder=4,
        label=r"fixed fraction: $\kappa_{\rm min}=10^{-3}\,\kappa_{\rm thr}$  (old sweep)")
ax.plot(kt, sB, "-s", color=BLUE, ms=4, lw=1.7, zorder=5,
        label=r"fixed absolute: $\kappa_{\rm min}=10^{-3}\,\kappa_{\rm thr}^{\rm fid}=1.28\times10^{-7}$")
ax.axhline(plateau, color=MUTED, ls=":", lw=1.2)

expl = Path("sigma_explicit_vs_kthr_zs1.txt")
if expl.exists():
    ktE, nE, sE, rE, nreal = np.loadtxt(expl, unpack=True)
    # Campbell band additivity: explicit band (kt, inf) = full - background (kmin, kt)
    sB_on_E = np.interp(np.log(ktE), np.log(kt), sB)
    ax.plot(ktE, np.sqrt(np.maximum(plateau**2 - sB_on_E**2, 0.0)), "--", color=GREEN,
            lw=1.5, zorder=3,
            label=r"predicted explicit: $\sqrt{\sigma_{\rm full}^2-\sigma_W^2}$ (band additivity)")
    ax.errorbar(ktE, sE, yerr=sE * rE / 2.0, fmt="^", color=GREEN, ms=6, lw=1.2,
                capsize=2, zorder=6,
                label=r"measured explicit: MC Var$(\Sigma\kappa,\ \kappa>\kappa_{\rm thr})$")
    quad = np.sqrt(sE**2 + sB_on_E**2)
    ax.plot(ktE, quad, "-", color=INK, lw=1.3, alpha=0.75, zorder=3,
            label=r"quadrature sum: $\sqrt{\sigma_{\rm explicit}^2+\sigma_W^2}$")
    print(f"quadrature sum / plateau: min {quad.min()/plateau:.4f}  max {quad.max()/plateau:.4f}")
ax.annotate(rf"plateau = full Campbell variance: $\sigma={plateau:.5f}$"
            "\n(MC measured total $\\approx 0.0275$)",
            xy=(1.3e-5, 0.0035), xytext=(1.3e-5, 0.0035), color=INK, fontsize=9.5)
ax.annotate("floor artifact:\n$\\kappa_{\\rm min}$ rises with $\\kappa_{\\rm thr}$,\n"
            "real variance discarded",
            xy=(kt[-4], sA[-4]), xytext=(2e0, 0.010), color=RED, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

ax.set_xscale("log")
ax.set_xlabel(r"host resolution threshold $\kappa_{\rm thr}$")
ax.set_ylabel(r"analytic background $\sigma_W$   ($z_s=1$)")
ax.set_title("Fixing the absolute floor removes the high-$\\kappa_{\\rm thr}$ collapse")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
          fontsize=9.5)
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigmaW_vs_kthr.{ext}", dpi=200)
print("wrote plots/sigmaW_vs_kthr.{png,pdf}")
print(f"plateau sigma_W (kappa_thr -> inf, fixed kappa_min) = {plateau:.6f}")
i = int(np.argmax(sA))
print(f"fixed-eps curve peaks at kappa_thr = {kt[i]:.3g}, sigma = {sA[i]:.5f}, then collapses to {sA[-1]:.2e}")
