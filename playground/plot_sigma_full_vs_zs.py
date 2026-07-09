#!/usr/bin/env python3
"""sigma_full(z_s): the analytic (Campbell, halo-only) convergence sigma vs source z.

Left: dense curve from sigma_full_vs_zs.txt, with the partition-study plateaus and
the MC measured-total means (from sigma_total_vs_kthr_zs*.txt) overlaid.
Right: local logarithmic slope d ln(sigma)/d ln(zs) — the effective power law.

Run from playground/. Writes plots/sigma_full_vs_zs.{png,pdf}.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED, GREEN = "#2a78d6", "#e34948", "#2e9e62"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

zs, kfid, sfull = np.loadtxt("sigma_full_vs_zs.txt", unpack=True)

# overlay: study plateaus (analytic, from the kappa_thr sweeps) + MC total means
zs_pts, plat_pts, mc_pts = [], [], []
for p in sorted(Path(".").glob("sigmaW_vs_kthr_zs*.txt")):
    m = re.match(r"sigmaW_vs_kthr_zs([\d.]+)\.txt", p.name)
    tot = Path(f"sigma_total_vs_kthr_zs{m.group(1)}.txt")
    if not (m and tot.exists()):
        continue
    zs_pts.append(float(m.group(1)))
    plat_pts.append(np.loadtxt(p, unpack=True)[2][-1])
    mc_pts.append(np.loadtxt(tot, unpack=True)[1].mean())
order = np.argsort(zs_pts)
zs_pts, plat_pts, mc_pts = (np.array(zs_pts)[order], np.array(plat_pts)[order],
                            np.array(mc_pts)[order])

fig, (ax, axs) = plt.subplots(1, 2, figsize=(11.2, 4.6))

ax.plot(zs, sfull, "-", color=BLUE, lw=1.9, zorder=3, label=r"analytic $\sigma_{\rm full}(z_s)$")
ax.plot(zs_pts, plat_pts, "s", color=INK, ms=6, mfc="none", mew=1.4, zorder=4,
        label="partition-study plateaus")
ax.plot(zs_pts, mc_pts, "o", color=RED, ms=4.5, zorder=5,
        label="MC measured total (sweep mean)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$z_s$")
ax.set_ylabel(r"$\sigma_\kappa$  (halo-only)")
ax.set_title(r"full convergence $\sigma$ vs. source redshift")
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)

lnz, lns = np.log(zs), np.log(sfull)
slope = np.gradient(lns, lnz)
axs.plot(zs, slope, "-", color=GREEN, lw=1.9, zorder=3)
axs.set_xscale("log")
axs.set_xlabel(r"$z_s$")
axs.set_ylabel(r"$d\ln\sigma_{\rm full}\,/\,d\ln z_s$")
axs.set_title("local power-law slope")
axs.grid(color=GRID, lw=.6, which="both"); axs.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigma_full_vs_zs.{ext}", dpi=200)
print("wrote plots/sigma_full_vs_zs.{png,pdf}")
for a, b in [(0.2, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10)]:
    ia, ib = np.argmin(np.abs(zs - a)), np.argmin(np.abs(zs - b))
    g = (lns[ib] - lns[ia]) / (lnz[ib] - lnz[ia])
    print(f"effective slope {a}->{b}: sigma ~ zs^{g:.2f}")
