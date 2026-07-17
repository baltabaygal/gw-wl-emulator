#!/usr/bin/env python3
"""Transverse footprint of the weak band vs kappa_thr (z_s=1).

Data: playground/weak_band_footprint_zs1.txt (weak_band_footprint.cpp probe):
weighted r-quantiles of all band annuli under the clustering weight
(b Dg x mean contribution — what the corr term responds to) and the shot
weight (kappa^2 — Campbell term). Reference lines: production R_perp
(= R_L(1e14) = 8441 kpc), default kappa_thr, floor kappa_min.
Output: plots/weak_band_footprint.png
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_prod.plot_style import apply_style  # noqa: E402

apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

C_CORR, C_SHOT, C_DEF = "#5b8fbe", "#0f4c81", "#888888"
RPERP_DEFAULT = 8441.0
KTHR_DEFAULT = 1.277748e-4
KMIN = 1.278e-7

d = np.loadtxt(ROOT / "playground" / "weak_band_footprint_zs1.txt")
kt = d[:, 0]
rC10, rC50, rC90 = d[:, 1], d[:, 2], d[:, 3]
rS10, rS50, rS90 = d[:, 4], d[:, 5], d[:, 6]

# local log-slopes in the falling region (kt <= 1e-3) for the annotation
w = kt <= 1e-3
slC = np.polyfit(np.log(kt[w]), np.log(rC50[w]), 1)[0]
slS = np.polyfit(np.log(kt[w]), np.log(rS50[w]), 1)[0]
iC = np.argmin(np.abs(kt - KTHR_DEFAULT))
print(f"slopes (kt<=1e-3): corr {slC:+.3f}, shot {slS:+.3f}; "
      f"at default kthr: rC50 = {rC50[iC]:.0f} kpc, rS50 = {rS50[iC]:.0f} kpc")

fig, ax = plt.subplots(figsize=(4.4, 3.3))
fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.86)

ax.fill_between(kt, rC10, rC90, color=C_CORR, alpha=0.22, lw=0)
ax.plot(kt, rC50, color=C_CORR, lw=1.5,
        label=r"corr weight $b\,D_g\,\bar\kappa$ (q10–q90)")
ax.fill_between(kt, rS10, rS90, color=C_SHOT, alpha=0.15, lw=0)
ax.plot(kt, rS50, color=C_SHOT, lw=1.5, ls="-.",
        label=r"shot weight $\kappa^2$ (q10–q90)")

ax.axhline(RPERP_DEFAULT, color="#d97706", lw=1.1, ls="--",
           label=r"$R_\perp$ default $= R_L(10^{14}) = 8.4\,$Mpc")
ax.axvline(KTHR_DEFAULT, color=C_DEF, lw=0.9, ls=":")
ax.axvline(KMIN, color=C_DEF, lw=0.9, ls="-.", alpha=0.8)
ax.text(KTHR_DEFAULT * 1.25, 5e6, r"default $\kappa_{\rm thr}$",
        fontsize=6, color=C_DEF, rotation=90, va="top")
ax.text(KMIN * 1.25, 5e6, r"floor $\kappa_{\rm min}$ (band empties)",
        fontsize=6, color=C_DEF, rotation=90, va="top")

ax.annotate(rf"$r \propto \kappa_{{\rm thr}}^{{{slC:+.2f}}}$",
            xy=(3e-6, 3.6e4), fontsize=7, color=C_CORR)
ax.annotate(rf"$r \propto \kappa_{{\rm thr}}^{{{slS:+.2f}}}$"
            "\n" r"($\sim r_{\rm max}$ of the split)",
            xy=(3e-6, 2.0e2), fontsize=7, color=C_SHOT)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(8e-8, kt.max() * 1.5)
ax.set_ylim(1.0, 8e6)
ax.set_xlabel(r"$\kappa_{\rm threshold}$")
ax.set_ylabel(r"weighted band radius $r$ [comoving kpc]")
ax.grid(False)
ax.legend(fontsize=6, loc="lower left", handlelength=2.0)
ax.set_title(r"transverse footprint of the weak band ($z_s=1$)", fontsize=9)

out = ROOT / "plots" / "weak_band_footprint.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print(f"Wrote {out}")
