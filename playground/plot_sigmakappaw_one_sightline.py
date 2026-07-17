#!/usr/bin/env python3
"""Companion figure to plot_sigmakappaw_variance_proof.py: show what "brute force"
means by building ONE sightline out of its individual weak halos.

Reads (from the sigmakappaw_poisson_vs_fixed probe):
  - playground/sigmakappaw_one_sightline_zs1.txt  columns: kappa  z_lens  r_kpc
        every weak halo drawn on realization 0 (the model's own Poisson field)
  - playground/sigmakappaw_sums_poisson_zs1.txt   one total Sum(kappa) per sightline
  - playground/sigmakappaw_poisson_vs_fixed_zs1.log   for K1 (mean)

Panel A: the running total Sum(kappa) as the halos of this one sightline are added
         (halos sorted largest-first), so you see a few near-threshold lenses start
         it and a long tail of many tiny halos fill it in to the sightline's total.
         Points along the top = the individual halo contributions, colored by lens z.
Panel B: the histogram of totals over ALL sightlines, with THIS sightline marked --
         i.e. the object built in panel A is a single sample of the distribution the
         other figure compares against K2.

Writes plots/sigmakappaw_one_sightline.{png,pdf}.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection  # noqa: F401 (kept for parity)

LOG = "playground/sigmakappaw_poisson_vs_fixed_zs1.log"
K1 = float(re.search(r"K1 = ([\d.e+-]+)", open(LOG).read()).group(1))

halos = np.loadtxt("playground/sigmakappaw_one_sightline_zs1.txt")   # (Nh, 3)
kap, zl, r = halos[:, 0], halos[:, 1], halos[:, 2]
tot = kap.sum()
Nh = len(kap)

sums = np.loadtxt("playground/sigmakappaw_sums_poisson_zs1.txt")

# sort halos largest contribution first -> cumulative build-up
order = np.argsort(kap)[::-1]
kap_s, zl_s = kap[order], zl[order]
cum = np.cumsum(kap_s)
rank = np.arange(1, Nh + 1)

# a couple of headline fractions for the annotation
def frac_from_top(f):
    n = max(1, int(f * Nh))
    return cum[n - 1] / tot

top1pct = frac_from_top(0.01)
half_rank = int(np.searchsorted(cum, 0.5 * tot)) + 1   # how many halos make half the total

# ---- style (matches the proof figure) -----------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
BLUE, RED, YELLOW = "#2a78d6", "#e34948", "#eda100"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10.5, "axes.titlesize": 11.5,
})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6))
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.85, wspace=0.26)

# ---- Panel A: build one sightline ---------------------------------------------
axA.plot(rank, cum, color=BLUE, lw=2, zorder=4,
         label="running total  $\\sum\\kappa$  (halos added largest-first)")
sc = axA.scatter(rank, kap_s, c=zl_s, s=7, cmap="viridis", alpha=0.7, zorder=3,
                 label="each halo's own $\\kappa$")
axA.axhline(tot, color=INK2, lw=1.2, ls="--")
axA.text(1.3, tot, f"  this sightline's total  $\\sum\\kappa$ = {tot:.4f}",
         color=INK2, va="bottom", ha="left", fontsize=9.3)
axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlabel("halo number (sorted by contribution)")
axA.set_ylabel(r"$\kappa$   /   running $\sum\kappa$")
axA.set_title(f"A — one sightline = {Nh:,} weak halos, summed ($z_s=1$)", loc="left")
axA.text(0.03, 0.06,
         f"largest single halo = {100*kap_s[0]/tot:.2f}% of total\n"
         f"top 1% of halos ({int(0.01*Nh):,}) make {100*top1pct:.0f}% of $\\sum\\kappa$\n"
         f"it takes {half_rank:,} halos to reach half the total",
         transform=axA.transAxes, fontsize=9, color=INK2, va="bottom",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f9f9f7", ec=BASE, lw=0.8))
axA.legend(frameon=False, fontsize=8.6, loc="upper right")
axA.grid(color=GRID, lw=0.6, alpha=0.7, which="both")
axA.set_axisbelow(True)
cb = fig.colorbar(sc, ax=axA, pad=0.01, fraction=0.05)
cb.set_label("lens redshift $z_l$", fontsize=9)
cb.ax.tick_params(labelsize=8)

# ---- Panel B: this sightline is one sample of the distribution ----------------
d_all = (sums - K1) * 1e3
axB.hist(d_all, bins=60, density=True, color=BLUE, alpha=0.32, edgecolor="none",
         label=f"all {len(sums):,} sightlines\n(each built like panel A)")
axB.axvline((tot - K1) * 1e3, color=RED, lw=2.2, zorder=5)
axB.text((tot - K1) * 1e3, axB.get_ylim()[1] * 0.92,
         "  the panel-A\n  sightline",
         color=RED, va="top", ha="left", fontsize=9.3)
axB.set_xlabel(r"$\sum\kappa_{\rm weak} - \langle\,\cdot\,\rangle$   [$10^{-3}$]")
axB.set_ylabel("probability density")
axB.set_title("B — that sightline is one draw from the histogram", loc="left")
axB.legend(frameon=False, fontsize=8.6, loc="upper left")
axB.grid(color=GRID, lw=0.6, alpha=0.7)
axB.set_axisbelow(True)

fig.suptitle("What the brute force does: throw down every weak halo on a sightline, sum their $\\kappa$, repeat",
             x=0.07, ha="left", fontsize=12, color=INK)

for ext in ("png", "pdf"):
    fig.savefig(f"plots/sigmakappaw_one_sightline.{ext}", dpi=200)
print("wrote plots/sigmakappaw_one_sightline.{png,pdf}")
print(f"sightline: {Nh:,} halos, Sum kappa = {tot:.6f}  (mean K1 = {K1:.6f})")
print(f"largest halo {100*kap_s[0]/tot:.2f}% of total; top 1% -> {100*top1pct:.1f}%; "
      f"half-total at {half_rank:,} halos")
