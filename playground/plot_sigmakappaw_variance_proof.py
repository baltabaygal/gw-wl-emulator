#!/usr/bin/env python3
"""Figure for the sigmakappaW -kappa1^2/Nh question.

Reads the output of playground/sigmakappaw_poisson_vs_fixed.cpp:
  - playground/sigmakappaw_poisson_vs_fixed_zs1.log   (summary numbers)
  - playground/sigmakappaw_sums_{poisson,frozen}_zs1.txt (per-realization sums)

Panel A: the raw MC — histogram of the weak-band kappa sum over realizations
         (Poisson counts, the model's own convention), with the Gaussian the
         code currently uses (sigma_W) vs the Campbell width sqrt(K2).
Panel B: zoom on the subtraction — Var/K2 for the Poisson-count MC and the
         frozen-count MC, against the two analytic predictions.

Writes plots/sigmakappaw_variance_proof.{png,pdf}.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

LOG = "playground/sigmakappaw_poisson_vs_fixed_zs1.log"

# ---- parse the probe's summary ------------------------------------------------
txt = open(LOG).read()

def grab(pattern):
    m = re.search(pattern, txt)
    assert m, pattern
    return float(m.group(1))

n_tot = grab(r"n = ([\d.e+-]+)")
K1 = grab(r"K1 = ([\d.e+-]+)")
K2 = grab(r"K2 = ([\d.e+-]+)")
sigma_code = grab(r"sigmakappaW\(\) = ([\d.e+-]+)")
varP_pred = grab(r"Var_P \(Campbell\)\s+= ([\d.e+-]+)")
varF_pred = grab(r"Var_F \(fixed N\)\s+= ([\d.e+-]+)")

sumsP = np.loadtxt("playground/sigmakappaw_sums_poisson_zs1.txt")
sumsF = np.loadtxt("playground/sigmakappaw_sums_frozen_zs1.txt")
Nreal = len(sumsP)
relerr = np.sqrt(2.0 / (Nreal - 1))

varP, varF = sumsP.var(), sumsF.var()

# ---- style ---------------------------------------------------------------------
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
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.86, wspace=0.25)

# ---- Panel A: the MC itself ----------------------------------------------------
d = (sumsP - K1) * 1e3  # centered, in units of 1e-3
axA.hist(d, bins=60, density=True, color=BLUE, alpha=0.35, edgecolor="none",
         label=f"brute-force MC ({Nreal:,} sightlines,\nPoisson counts, "
               f"$\\langle N\\rangle$={n_tot:,.0f} weak halos each)")
x = np.linspace(d.min(), d.max(), 400)
sig_camp = np.sqrt(K2) * 1e3
sig_code = sigma_code * 1e3
axA.plot(x, np.exp(-x**2 / (2 * sig_camp**2)) / (sig_camp * np.sqrt(2 * np.pi)),
         color=BLUE, lw=2, label=r"Campbell: $\sigma^2=K_2=\int n\,\kappa^2$")
axA.plot(x, np.exp(-x**2 / (2 * sig_code**2)) / (sig_code * np.sqrt(2 * np.pi)),
         color=RED, lw=2, ls="--",
         label=r"code $\sigma_W$: $\pi$ measure, $-\kappa_1^2/N_h$")
axA.set_xlabel(r"$\sum\kappa_{\rm weak} - \langle\,\cdot\,\rangle$   [$10^{-3}$]")
axA.set_ylabel("probability density")
axA.set_title("A — draw the weak-band halos explicitly ($z_s=1$)", loc="left")
axA.legend(frameon=False, fontsize=8.6, loc="upper left")
axA.grid(color=GRID, lw=0.6, alpha=0.7)
axA.set_axisbelow(True)

# ---- Panel B: the subtraction, isolated ----------------------------------------
axB.axhline(1.0, color=BLUE, lw=1.6, ls="--")
axB.axhline(varF_pred / K2, color=INK2, lw=1.6, ls=":")
axB.errorbar([0], [varP / K2], yerr=[varP / K2 * relerr], fmt="o", ms=9,
             color=BLUE, capsize=5, lw=2, zorder=5)
axB.errorbar([1], [varF / K2], yerr=[varF / K2 * relerr], fmt="s", ms=9,
             color=RED, capsize=5, lw=2, zorder=5)
axB.text(0.98, 1.0 + 0.004, r"Campbell  $K_2$  (no subtraction)",
         color=BLUE, ha="right", va="bottom", fontsize=9.5)
axB.text(0.98, varF_pred / K2 - 0.005, r"fixed-count formula  $K_2 - K_1^2/n$",
         color=INK2, ha="right", va="top", fontsize=9.5)
axB.text(0.5, 0.905,
         "code returns the fixed-count value, halved:\n"
         rf"$\sigma_W^2 / K_2 = {sigma_code**2 / K2:.3f}"
         rf" \approx \frac{{1}}{{2}}\,(K_2 - K_1^2/n)/K_2$",
         ha="center", va="bottom", fontsize=9.5, color=INK2,
         bbox=dict(boxstyle="round,pad=0.45", fc="#f9f9f7", ec=BASE, lw=0.8))
axB.set_xticks([0, 1])
axB.set_xticklabels(["counts fluctuate\nPoisson($\\lambda_c$) per cell\n(the model, lensing.cpp:538)",
                     "count frozen\nexactly $n$ halos, always\n(the formula's assumption)"])
axB.set_xlim(-0.55, 1.55)
axB.set_ylim(0.90, 1.06)
axB.set_ylabel(r"Var$\left(\sum\kappa_{\rm weak}\right)\; /\; K_2$")
axB.set_title("B — same draw, only the count statistics changed", loc="left")
axB.grid(color=GRID, lw=0.6, alpha=0.7, axis="y")
axB.set_axisbelow(True)

fig.suptitle(r"Which variance does the weak-lens Gaussian need?"
             r"   ($K_1,K_2$: band integrals with the $2\pi r^2\,\mathrm{d}\ln r$ measure)",
             x=0.07, ha="left", fontsize=12, color=INK)

for ext in ("png", "pdf"):
    fig.savefig(f"plots/sigmakappaw_variance_proof.{ext}", dpi=200)
print("wrote plots/sigmakappaw_variance_proof.{png,pdf}")
print(f"Var_P/K2 = {varP/K2:.4f} ± {varP/K2*relerr:.4f}   (pred 1.0000)")
print(f"Var_F/K2 = {varF/K2:.4f} ± {varF/K2*relerr:.4f}   (pred {varF_pred/K2:.4f})")
print(f"code sigma_W^2/K2 = {sigma_code**2/K2:.4f}")
