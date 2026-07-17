#!/usr/bin/env python3
"""One figure, one panel per z_s: full variance partitioning in each panel.

Per panel: analytic background sigma_W (line, fixed absolute floor), measured
explicit (triangles), measured total (circles), dotted full-Campbell plateau.
Auto-discovers the z_s set like plot_sigma_partition_zs_study.py.

Run from playground/. Writes plots/sigma_partition_panels.{png,pdf}.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, GREEN = "#2a78d6", "#2e9e62"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 9.5})

zs_list = sorted(
    float(m.group(1))
    for p in Path(".").glob("sigma_total_vs_kthr_zs*.txt")
    if (m := re.match(r"sigma_total_vs_kthr_zs([\d.]+)\.txt", p.name))
    and Path(f"sigmaW_vs_kthr_zs{m.group(1)}.txt").exists()
    and Path(f"sigma_explicit_vs_kthr_zs{m.group(1)}.txt").exists()
)

ncol = 4
nrow = int(np.ceil(len(zs_list) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13.6, 3.4 * nrow), sharex=True)
axes = np.atleast_2d(axes)

for i, zs in enumerate(zs_list):
    ax = axes[i // ncol, i % ncol]
    kt, sW_old, sW = np.loadtxt(f"sigmaW_vs_kthr_zs{zs:g}.txt", unpack=True)
    ktE, nE, sE, rE, _ = np.loadtxt(f"sigma_explicit_vs_kthr_zs{zs:g}.txt", unpack=True)
    ktT, sT, rT, _ = np.loadtxt(f"sigma_total_vs_kthr_zs{zs:g}.txt", unpack=True)
    plateau = sW[-1]

    ax.axhline(plateau, color=MUTED, ls=":", lw=1.0, zorder=1)
    ax.plot(kt, sW, "-", color=BLUE, lw=1.6, zorder=3, label=r"$\sigma_W$ (analytic)")
    ax.errorbar(ktE, sE, yerr=sE * rE / 2.0, fmt="^", color=GREEN, ms=3.6, lw=0.8,
                capsize=1.2, zorder=4, label=r"$\sigma_{\rm explicit}$ (MC)")
    ax.errorbar(ktT, sT, yerr=sT * rT / 2.0, fmt="o", color=INK, ms=3.4, lw=0.8,
                capsize=1.2, zorder=5, label=r"$\sigma_{\rm total}$ (MC)")

    ax.set_xscale("log")
    ax.set_ylim(-0.06 * plateau, 1.22 * plateau)
    ax.text(0.04, 0.93, rf"$z_s = {zs:g}$", transform=ax.transAxes, fontsize=11,
            va="top", fontweight="bold")
    ax.text(0.04, 0.80, rf"$\sigma_{{\rm full}} = {plateau:.4f}$", transform=ax.transAxes,
            fontsize=8.5, va="top", color=MUTED)
    ax.grid(color=GRID, lw=.5, which="both"); ax.set_axisbelow(True)
    if i // ncol == nrow - 1:
        ax.set_xlabel(r"$\kappa_{\rm thr}$")
    if i % ncol == 0:
        ax.set_ylabel(r"$\sigma_\kappa$")

axes[0, 0].legend(frameon=False, fontsize=8, loc="center left")
for j in range(len(zs_list), nrow * ncol):
    axes[j // ncol, j % ncol].axis("off")

fig.suptitle("Convergence variance partitioning vs. host resolution threshold "
             "(halo-only)", y=0.995)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigma_partition_panels.{ext}", dpi=200)
print("wrote plots/sigma_partition_panels.{png,pdf}")
