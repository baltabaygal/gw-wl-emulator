#!/usr/bin/env python3
"""Split-quality knob: fraction of core Campbell cumulants below kappa_thr (z_s=1).

Data: playground/weak_band_cumulants_zs1.txt (weak_band_cumulants.cpp probe):
band k_n = int n kappa^n dA dchi over [kappa_min, kappa_thr], normalized by
the kappa_thr = 1 row (the certified kappa<1 core). f2 = shot-variance share
(exactly compensated by the weak arm — shown for context); f3, f4 = the
Poisson skewness/kurtosis the Gaussianized weak arm DISCARDS = the split
error budget. Top axis: cost <N>(kappa_thr) from nexp_vs_kthr_zs1.txt.
Output: plots/weak_band_cumulant_fractions.png
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

C2, C3, C4, C_DEF = "#5b8fbe", "#d97706", "#b5443c", "#888888"
KTHR_DEFAULT = 1.277748e-4
KMIN = 1.278e-7

d = np.loadtxt(ROOT / "playground" / "weak_band_cumulants_zs1.txt")
kt = d[:, 0]
iref = np.argmin(np.abs(kt - 1.0))
f2 = d[:, 2] / d[iref, 2]
f3 = d[:, 3] / d[iref, 3]
f4 = d[:, 4] / d[iref, 4]

nx = np.loadtxt(ROOT / "playground" / "nexp_vs_kthr_zs1.txt")
lnN = lambda k: np.interp(np.log(k), np.log(nx[:, 0]), np.log(nx[:, 1]))

m = kt <= 1.0
idef = np.argmin(np.abs(kt - KTHR_DEFAULT))
w = (kt >= 1e-6) & (kt <= 1e-3)
sl = [np.polyfit(np.log(kt[w]), np.log(f[w]), 1)[0] for f in (f2, f3, f4)]
print(f"slopes (1e-6..1e-3): f2 {sl[0]:+.2f}  f3 {sl[1]:+.2f}  f4 {sl[2]:+.2f}")
print(f"at default kthr: f2={f2[idef]:.3e}  f3={f3[idef]:.3e}  f4={f4[idef]:.3e}")

fig, ax = plt.subplots(figsize=(4.4, 3.3))
fig.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.80)

ax.plot(kt[m], f2[m], "o-", ms=2.6, lw=1.3, color=C2,
        label=rf"$f_2$ shot var. (compensated), $\propto\kappa_{{\rm thr}}^{{{sl[0]:.1f}}}$")
ax.plot(kt[m], f3[m], "s-", ms=2.6, lw=1.3, color=C3,
        label=rf"$f_3$ skewness discarded, $\propto\kappa_{{\rm thr}}^{{{sl[1]:.1f}}}$")
ax.plot(kt[m], f4[m], "^-", ms=2.6, lw=1.3, color=C4,
        label=rf"$f_4$ kurtosis discarded, $\propto\kappa_{{\rm thr}}^{{{sl[2]:.1f}}}$")

ax.axhline(1e-2, color="#444444", lw=0.8, ls="--", alpha=0.7)
ax.text(2.2e-7, 2.2e-3, r"1\% of core cumulant" if mpl.rcParams["text.usetex"]
        else "1% of core cumulant", fontsize=6, color="#444444")
ax.axvline(KTHR_DEFAULT, color=C_DEF, lw=0.9, ls=":")
ax.axvline(KMIN, color=C_DEF, lw=0.9, ls="-.", alpha=0.8)
ax.text(KTHR_DEFAULT * 1.25, 2e-14, r"default $\kappa_{\rm thr}$",
        fontsize=6, color=C_DEF, rotation=90, va="bottom")
ax.text(KMIN * 1.25, 2e-14, r"floor $\kappa_{\rm min}$", fontsize=6, color=C_DEF,
        rotation=90, va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(8e-8, 1.5)
ax.set_ylim(1e-14, 3.0)
ax.set_xlabel(r"$\kappa_{\rm threshold}$")
ax.set_ylabel(r"band fraction of core ($\kappa<1$) cumulant")
ax.grid(False)
ax.legend(fontsize=6, loc="lower right", handlelength=2.0)

# cost axis: <N> at matching kappa_thr positions
top = ax.twiny()
top.set_xscale("log")
top.set_xlim(ax.get_xlim())
tick_kt = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
top.set_xticks(tick_kt)
top.set_xticklabels([f"{np.exp(lnN(k)):.0e}".replace("e+0", "e").replace("e-0", "e-")
                     for k in tick_kt], fontsize=5.5)
top.set_xlabel(r"cost: $\langle N\rangle$ halos per ray", fontsize=7, labelpad=5)
top.tick_params(length=2)

ax.set_title(r"split-quality knob: non-Gaussianity handed to the weak arm ($z_s=1$)",
             fontsize=8, pad=28)

out = ROOT / "plots" / "weak_band_cumulant_fractions.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print(f"Wrote {out}")
