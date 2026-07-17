#!/usr/bin/env python3
"""Measured sigma_total (mean over the kappa_thr sweep) vs source redshift.

Top: MC measured sigma_total per z_s (mean of the sweep points; error bars = the
std of the points, i.e. the sweep-to-sweep MC scatter) with the dense analytic
sigma_full(z_s) curve underneath. Bottom: ratio measured/analytic.

Run from playground/. Writes plots/sigma_total_vs_zs.{png,pdf}.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED = "#2a78d6", "#e34948"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

zs_c, _, sfull_c = np.loadtxt("sigma_full_vs_zs.txt", unpack=True)

zs_pts, mean_pts, std_pts = [], [], []
for p in sorted(Path(".").glob("sigma_total_vs_kthr_zs*.txt")):
    m = re.match(r"sigma_total_vs_kthr_zs([\d.]+)\.txt", p.name)
    if not m:
        continue
    sT = np.loadtxt(p, unpack=True)[1]
    zs_pts.append(float(m.group(1)))
    mean_pts.append(sT.mean())
    std_pts.append(sT.std())
order = np.argsort(zs_pts)
zs_pts = np.array(zs_pts)[order]
mean_pts = np.array(mean_pts)[order]
std_pts = np.array(std_pts)[order]
sfull_at = np.interp(np.log(zs_pts), np.log(zs_c), sfull_c)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                              gridspec_kw={"height_ratios": [2.6, 1], "hspace": 0.06})

def sigma_fit(z, A=0.0431, z0=1.834, q=1.207, p=1.407):
    """A z^{3/2} [1+(z/z0)^q]^{-p/q}: exact z->0 limit + fitted smooth break."""
    return A * z**1.5 / (1 + (z / z0)**q)**(p / q)

ax.plot(zs_c, sfull_c, "-", color=BLUE, lw=1.8, zorder=3,
        label=r"analytic $\sigma_{\rm full}(z_s)$ (Campbell)")
ax.plot(zs_c, sigma_fit(zs_c), "--", color="#2e9e62", lw=1.6, zorder=4,
        label=r"fit $0.0431\,z_s^{3/2}[1+(z_s/1.83)^{1.21}]^{-1.17}$")
ax.errorbar(zs_pts, mean_pts, yerr=std_pts, fmt="o", color=RED, ms=6, lw=1.3,
            capsize=3, zorder=5,
            label=r"measured $\langle\sigma_{\rm total}\rangle$ (mean over $\kappa_{\rm thr}$ sweep)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel(r"$\sigma_\kappa$  (halo-only)")
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)
ax.set_title(r"measured total convergence $\sigma$ vs. source redshift")

axr.axhline(1.0, color=MUTED, ls=":", lw=1.2)
axr.plot(zs_c, sigma_fit(zs_c) / sfull_c, "--", color="#2e9e62", lw=1.6, zorder=4,
         label="fit / analytic")
axr.errorbar(zs_pts, mean_pts / sfull_at, yerr=std_pts / sfull_at, fmt="o", color=RED,
             ms=5, lw=1.2, capsize=3, zorder=5, label="measured / analytic")
axr.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2)
axr.set_xscale("log")
axr.set_ylim(0.94, 1.06)
axr.set_xlabel(r"$z_s$")
axr.set_ylabel(r"measured / analytic")
axr.grid(color=GRID, lw=.6, which="both"); axr.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigma_total_vs_zs.{ext}", dpi=200)
print("wrote plots/sigma_total_vs_zs.{png,pdf}")
for z, m_, s_, f_ in zip(zs_pts, mean_pts, std_pts, sfull_at):
    print(f"zs={z:<5g} mean={m_:.6f} +- {s_:.6f}   analytic={f_:.6f}   ratio={m_/f_:.4f}")
