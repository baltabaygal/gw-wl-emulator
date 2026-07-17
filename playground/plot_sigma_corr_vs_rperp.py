#!/usr/bin/env python3
"""sigma_corr (weak-arm clustering component) vs bias_Rperp at z_s=1.

Data: playground/sigma_corr_vs_rperp_zs1_ens.npz — 8-seed ensemble from
sweep_sigma_corr_vs_rperp.py, modes main (1..1e4 kpc) + ext (..1e6 kpc);
100k rays/seed/pt, default kappa_thr(<N>=100), clip1 core mask, analytic
shot sigma_W = 1.327e-3 subtracted in variance (signed, no clipping).

Fit (Var-weighted, full range): sigma_corr = sigma0 / (1 + (R/Rc)^p).
Plateau below the P(k) convergence scale; falling branch (asymptote ~1/R)
where the disk window removes transverse power; at large R the weak arm
collapses onto the pure Gaussian shot draw = the no-bias limit (verified
against a bias=False reference: Var_w = shot^2 to 0.002%).
Output: plots/sigma_corr_vs_rperp.png
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

C_CORR, C_W, C_FIT, C_DEF, C_SHOT = "#5b8fbe", "#0f4c81", "#d97706", "#888888", "#444444"
RPERP_DEFAULT = 8441.0

d = np.load(ROOT / "playground" / "sigma_corr_vs_rperp_zs1_ens.npz")
rp, corr, sem, sig_w = d["rperp"], d["corr"], d["sem"], d["sig_w"]
shot = float(d["shot"])
s0, rc, p = d["popt"]
print(f"fit: sigma0={s0:.5f}  Rc={rc:.0f} kpc  p={p:.3f}")

rf = np.geomspace(rp.min(), rp.max(), 700)
fit = s0 / (1 + (rf / rc) ** p)
lab_fit = r"fit $\sigma_0/\left[1+(R_\perp/R_c)^p\right]$"

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.16, top=0.82, wspace=0.30)

# ---- left: lin-lin zoom (R <= 1e4 kpc)
ax = axes[0]
m = rp <= 1.0e4
ax.errorbar(rp[m], corr[m], yerr=sem[m], fmt="o", ms=3.0, lw=0.8, color=C_CORR,
            label="weak (corr.), measured", zorder=5)
ax.plot(rf[rf <= 1.0e4], fit[rf <= 1.0e4], color=C_FIT, lw=1.4, label=lab_fit, zorder=4)
ax.axvline(RPERP_DEFAULT, color=C_DEF, lw=0.9, ls=":", zorder=3)
ax.text(RPERP_DEFAULT * 0.97, 0.0225, "default", fontsize=6, color=C_DEF, ha="right")
ax.set_ylim(0, 0.0265)
ax.set_title("lin-lin (zoom $R_\\perp \\leq 10\\,$Mpc)", fontsize=8)
ax.legend(fontsize=6, loc="lower left", handlelength=1.8)

# ---- right: log-log, full range to 1 Gpc, with the no-bias asymptote
ax = axes[1]
ax.errorbar(rp, corr, yerr=sem, fmt="o", ms=3.0, lw=0.8, color=C_CORR,
            label="weak (corr.)", zorder=5)
ax.plot(rp, sig_w, "s-", ms=2.2, lw=0.7, color=C_W, alpha=0.85,
        label=r"weak total $\sigma_w=\sqrt{\rm shot^2+corr^2}$", zorder=4)
ax.plot(rf, fit, color=C_FIT, lw=1.4, label=lab_fit, zorder=4)
ax.axhline(shot, color=C_SHOT, lw=0.9, ls="--", zorder=3,
           label=r"shot $\sigma_W$ = no-bias limit")
ax.axvline(RPERP_DEFAULT, color=C_DEF, lw=0.9, ls=":", zorder=3,
           label=r"default $R_\perp = 8441$ kpc")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(6e-5, 0.05)
ax.set_title("log-log (full range)", fontsize=8)
ax.legend(fontsize=5.5, loc="lower left", handlelength=1.8)
ax.annotate(rf"$\sigma_0 = {s0:.4f}$" "\n" rf"$R_c = {rc/1e3:.1f}\,$Mpc"
            "\n" rf"$p = {p:.2f}$",
            xy=(0.71, 0.74), xycoords="axes fraction", fontsize=7)

for ax in axes:
    ax.set_xlabel(r"$R_\perp$ [comoving kpc]")
    ax.set_ylabel(r"$\sigma_{\kappa}$")
    ax.grid(False)

fig.suptitle(r"weak-arm clustering component vs pencil radius $R_\perp$  "
             r"($z_s=1$, default $\kappa_{\rm thr}$; bias$=$False ref: "
             r"$\mathrm{Var}_w = \sigma_W^2$ exactly)",
             fontsize=9, y=0.97)

out = ROOT / "plots" / "sigma_corr_vs_rperp.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print(f"Wrote {out}")
