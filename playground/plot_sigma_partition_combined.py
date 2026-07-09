#!/usr/bin/env python3
"""Combined variance-partitioning figure: measured total + measured explicit +
analytic background vs the host resolution threshold kappa_thr (zs=1, halo-only).

Inputs (all in playground/):
  sigma_total_vs_kthr_zs1.txt     measured sigma_total (production MC, floor-consistent
                                  background injection, filaments/bias/ell/subhalo off)
  sigma_explicit_vs_kthr_zs1.txt  measured sigma_explicit (Poisson MC of kappa>kappa_thr)
  sigmaW_vs_kthr_zs1.txt          analytic sigmakappaW: fixed-fraction vs fixed-absolute floor

Writes plots/sigma_partition_vs_kthr.{png,pdf}.
"""
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED, GREEN = "#2a78d6", "#e34948", "#2e9e62"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

kt, sW_old, sW = np.loadtxt("sigmaW_vs_kthr_zs1.txt", unpack=True)
ktE, nE, sE, rE, _ = np.loadtxt("sigma_explicit_vs_kthr_zs1.txt", unpack=True)
ktT, sT, rT, _ = np.loadtxt("sigma_total_vs_kthr_zs1.txt", unpack=True)
plateau = sW[-1]

fig, ax = plt.subplots(figsize=(8.6, 5.6))

ax.axhline(plateau, color=MUTED, ls=":", lw=1.2, zorder=1)
ax.plot(kt, sW_old, "--", color=RED, lw=1.4, alpha=0.65, zorder=2,
        label=r"analytic background, OLD floor $\kappa_{\rm min}=10^{-3}\kappa_{\rm thr}$ (artifact)")
ax.plot(kt, sW, "-", color=BLUE, lw=1.9, zorder=4,
        label=r"analytic background $\sigma_W$ (fixed $\kappa_{\rm min}=1.28\times10^{-7}$)")
ax.errorbar(ktE, sE, yerr=sE * rE / 2.0, fmt="^", color=GREEN, ms=6, lw=1.2, capsize=2,
            zorder=5, label=r"measured explicit: MC Var$(\Sigma\kappa,\ \kappa>\kappa_{\rm thr})$")
ax.errorbar(ktT, sT, yerr=sT * rT / 2.0, fmt="o", color=INK, ms=5, lw=1.2, capsize=2,
            zorder=6, label=r"measured total: production MC, $10^5$ realizations")

ax.annotate(rf"full Campbell variance $\sigma = {plateau:.5f}$",
            xy=(1.3e-5, plateau), xytext=(1.3e-5, plateau * 1.045), color=INK, fontsize=9.5)

ax.set_xscale("log")
ax.set_xlabel(r"host resolution threshold $\kappa_{\rm thr}$")
ax.set_ylabel(r"$\sigma_\kappa$   ($z_s=1$, halo-only)")
ax.set_title("Convergence variance partitioning vs. host resolution threshold")
ax.set_ylim(-0.0012, 0.0315)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1,
          fontsize=9.5)
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)
plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/sigma_partition_vs_kthr.{ext}", dpi=200)
print("wrote plots/sigma_partition_vs_kthr.{png,pdf}")

sW_on_T = np.interp(np.log(ktT), np.log(kt), sW)
sE_on_T = np.interp(np.log(ktT), np.log(ktE), sE)
quad = np.sqrt(sE_on_T**2 + sW_on_T**2)
print("measured total / plateau:      ", " ".join(f"{v:.3f}" for v in sT / plateau))
print("quadrature(expl+bg) / plateau: ", " ".join(f"{v:.3f}" for v in quad / plateau))
