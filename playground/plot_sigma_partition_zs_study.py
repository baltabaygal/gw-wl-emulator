#!/usr/bin/env python3
"""z_s dependence of the variance partitioning vs kappa_thr.

Auto-discovers playground/sigma_total_vs_kthr_zs<zs>.txt (+ matching sigmaW / explicit
files) for all z_s present. Fiducial thresholds kappa_thr(N=100, zs) are computed via
gwlensing.get_kappa_threshold and cached in kthr_fid_cache.json.

Panels:
  1. absolute: analytic background curves + measured totals (flat at each plateau);
  2. rescaled by each zs's plateau and fiducial threshold: universality of the handover;
  3. flatness summary: mean/min/max of sigma_total/plateau per zs.

Run from playground/ with the test env. Writes plots/sigma_partition_zs_study.{png,pdf}.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

zs_list = sorted(
    float(m.group(1))
    for p in Path(".").glob("sigma_total_vs_kthr_zs*.txt")
    if (m := re.match(r"sigma_total_vs_kthr_zs([\d.]+)\.txt", p.name))
    and Path(f"sigmaW_vs_kthr_zs{m.group(1)}.txt").exists()
    and Path(f"sigma_explicit_vs_kthr_zs{m.group(1)}.txt").exists()
)
print("z_s found:", zs_list)

CACHE = Path("kthr_fid_cache.json")
kfid = json.loads(CACHE.read_text()) if CACHE.exists() else {}
missing = [z for z in zs_list if f"{z:g}" not in kfid]
if missing:
    sys.path.insert(0, "../build")
    import gwlensing
    for z in missing:
        kfid[f"{z:g}"] = gwlensing.get_kappa_threshold(z, 0.674, 0.315, 0.811, 100)
        print(f"kappa_thr_fid(zs={z:g}) = {kfid[f'{z:g}']:.6e}")
    CACHE.write_text(json.dumps(kfid, indent=1))

data = {}
for zs in zs_list:
    kt, sW_old, sW = np.loadtxt(f"sigmaW_vs_kthr_zs{zs:g}.txt", unpack=True)
    ktE, nE, sE, rE, _ = np.loadtxt(f"sigma_explicit_vs_kthr_zs{zs:g}.txt", unpack=True)
    ktT, sT, rT, _ = np.loadtxt(f"sigma_total_vs_kthr_zs{zs:g}.txt", unpack=True)
    data[zs] = dict(kt=kt, sW=sW, ktE=ktE, sE=sE, rE=rE, ktT=ktT, sT=sT, rT=rT,
                    plateau=sW[-1], kfid=kfid[f"{zs:g}"])

norm = plt.Normalize(np.log10(0.15), np.log10(12.0))
color = lambda zs: cm.viridis(norm(np.log10(zs)))

fig, (ax, axn, axf) = plt.subplots(1, 3, figsize=(14.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1.15, 0.85]})

for zs, d in data.items():
    c = color(zs)
    ax.plot(d["kt"], d["sW"], "-", color=c, lw=1.6, zorder=3)
    ax.errorbar(d["ktT"], d["sT"], yerr=d["sT"] * d["rT"] / 2.0, fmt="o", color=c,
                ms=3.6, lw=0.9, capsize=1.5, zorder=5, label=f"$z_s={zs:g}$")
    ax.axhline(d["plateau"], color=c, ls=":", lw=0.9, alpha=0.55)
ax.set_xscale("log")
ax.set_xlabel(r"$\kappa_{\rm thr}$")
ax.set_ylabel(r"$\sigma_\kappa$  (halo-only)")
ax.set_title("absolute: line = analytic background,\ncircles = measured total")
ax.legend(frameon=False, fontsize=8.5, loc="center left", ncol=2)
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)

for zs, d in data.items():
    c = color(zs)
    p, kf = d["plateau"], d["kfid"]
    axn.plot(d["kt"] / kf, d["sW"] / p, "-", color=c, lw=1.6, zorder=3)
    axn.errorbar(d["ktE"] / kf, d["sE"] / p, yerr=d["sE"] * d["rE"] / 2.0 / p, fmt="^",
                 color=c, ms=3.4, lw=0.8, capsize=1.2, zorder=4, alpha=0.75)
    axn.errorbar(d["ktT"] / kf, d["sT"] / p, yerr=d["sT"] * d["rT"] / 2.0 / p, fmt="o",
                 color=c, ms=3.6, lw=0.9, capsize=1.5, zorder=5)
axn.axhline(1.0, color=MUTED, ls=":", lw=1.2)
axn.set_xscale("log")
axn.set_xlabel(r"$\kappa_{\rm thr}\,/\,\kappa_{\rm thr}^{\rm fid}(z_s)$")
axn.set_ylabel(r"$\sigma_\kappa\,/\,\sigma_{\rm full}(z_s)$")
axn.set_title("rescaled: circles = total, triangles = explicit,\nline = background")
axn.grid(color=GRID, lw=.6, which="both"); axn.set_axisbelow(True)

for zs, d in data.items():
    r = d["sT"] / d["plateau"]
    axf.errorbar([zs], [r.mean()], yerr=[[r.mean() - r.min()], [r.max() - r.mean()]],
                 fmt="o", color=color(zs), ms=6, lw=1.4, capsize=3, zorder=5)
axf.axhline(1.0, color=MUTED, ls=":", lw=1.2)
axf.set_xscale("log")
axf.set_xlabel(r"$z_s$")
axf.set_ylabel(r"$\sigma_{\rm total}\,/\,\sigma_{\rm full}$")
axf.set_title("flatness of the measured total\n(mean over the sweep; bars = min/max)")
axf.set_ylim(0.9, 1.1)
axf.grid(color=GRID, lw=.6, which="both"); axf.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigma_partition_zs_study.{ext}", dpi=200)
print("wrote plots/sigma_partition_zs_study.{png,pdf}")
for zs, d in data.items():
    r = d["sT"] / d["plateau"]
    print(f"zs={zs:<4g} plateau={d['plateau']:.6f}  kfid={d['kfid']:.3e}  "
          f"total/plateau: mean={r.mean():.3f} min={r.min():.3f} max={r.max():.3f}")
