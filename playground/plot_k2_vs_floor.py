#!/usr/bin/env python3
"""Plot K2 (weak-lens variance) vs the convergence-floor fraction eps_floor.

Reads playground/k2_vs_floor_zs{1,5}.txt (from k2_vs_floor.cpp; columns eps_floor K2).
Left panel: K2/K2(1e-7) — a plateau at small eps_floor means the outward radial
integral is converged. Right panel: the deficit 1 - K2/K2(1e-7) on log-log, showing
the missing variance scales ~linearly with eps_floor (each decade of floor buys a
decade of accuracy). Writes plots/k2_vs_floor.{png,pdf}.
"""
import numpy as np
import matplotlib.pyplot as plt

INK, MUTED, BASE, GRID = "#0b0b0b", "#898781", "#c3c2b7", "#e1e0d9"
BLUE, RED, GREEN = "#2a78d6", "#e34948", "#2e9e62"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": BASE, "font.size": 10.5})

runs = {}
for zs, color in [(1, BLUE), (5, GREEN)]:
    eps, K2 = np.loadtxt(f"k2_vs_floor_zs{zs}.txt", unpack=True)
    o = np.argsort(eps)
    eps, K2 = eps[o], K2[o]
    runs[zs] = (eps, K2 / K2[0], color)   # smallest eps_floor = most converged

fig, (ax, axd) = plt.subplots(1, 2, figsize=(11.6, 4.7))

for zs, (eps, ratio, color) in runs.items():
    ax.plot(eps, ratio, "-o", color=color, ms=4, lw=1.6, zorder=4, label=f"$z_s={zs}$")
ax.axhline(1.0, color=MUTED, ls=":", lw=1.2)

eps1, ratio1, _ = runs[1]
i = int(np.argmin(np.abs(eps1 - 1e-3)))
ax.scatter([eps1[i]], [ratio1[i]], s=130, facecolor="none", edgecolor=RED, lw=2, zorder=5)
ax.annotate(f"default $\\epsilon_{{\\rm floor}}=10^{{-3}}$\nK2 = {ratio1[i]*100:.2f}% of converged",
            xy=(eps1[i], ratio1[i]), xytext=(eps1[i]*30, 0.93), color=RED, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

ax.set_xscale("log"); ax.invert_xaxis()          # small floor (deep integral) on the right
ax.set_xlabel(r"convergence floor  $\epsilon_{\rm floor}=\kappa_{\rm min}/\kappa_{\rm thr}$"
              r"    (deeper integral $\rightarrow$)")
ax.set_ylabel(r"$K_2(\epsilon_{\rm floor})\,/\,K_2(10^{-7})$")
ax.set_title(r"Weak-lens variance $K_2$ converges as the floor is lowered")
ax.set_ylim(0.87, 1.02)
ax.legend(frameon=False, loc="lower left")
ax.grid(color=GRID, lw=.6, which="both"); ax.set_axisbelow(True)

for zs, (eps, ratio, color) in runs.items():
    deficit = 1.0 - ratio
    m = deficit > 0
    axd.plot(eps[m], deficit[m], "-o", color=color, ms=4, lw=1.6, zorder=4, label=f"$z_s={zs}$")
# linear-in-eps guide through the zs=1 default point
axd.plot(eps1, (1.0 - ratio1[i]) * eps1 / 1e-3, ls="--", color=MUTED, lw=1.2,
         label=r"$\propto \epsilon_{\rm floor}$")
axd.scatter([eps1[i]], [1.0 - ratio1[i]], s=130, facecolor="none", edgecolor=RED, lw=2, zorder=5)
axd.set_xscale("log"); axd.set_yscale("log"); axd.invert_xaxis()
axd.set_xlabel(r"$\epsilon_{\rm floor}$    (deeper integral $\rightarrow$)")
axd.set_ylabel(r"missing variance  $1 - K_2(\epsilon_{\rm floor})/K_2(10^{-7})$")
axd.set_title("Missing variance falls one decade per decade of floor")
axd.set_ylim(1e-6, 0.3)
axd.legend(frameon=False, loc="lower left")
axd.grid(color=GRID, lw=.6, which="both"); axd.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../plots/k2_vs_floor.{ext}", dpi=200)
print("wrote plots/k2_vs_floor.{png,pdf}")
for zs, (eps, ratio, _) in runs.items():
    for e in (1e-2, 1e-3, 1e-5):
        j = int(np.argmin(np.abs(eps - e)))
        print(f"zs={zs}  K2({e:.0e})/K2(1e-7) = {ratio[j]:.5f}")
