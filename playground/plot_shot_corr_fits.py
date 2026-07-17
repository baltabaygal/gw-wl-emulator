#!/usr/bin/env python3
"""
Companion analysis to paper_prod/scripts/plot_sigma_partition_vs_kthr_bias.py:
isolate the two weak-arm components and test the conjectured scalings.

  shot : Var_shot(k_thr)  = A * (k_thr - k_min)          [linear in k_thr]
  corr : Var_corr(k_thr)  = B * ln^2(k_thr / k_min)      [sigma linear in ln k]

shot = analytic Campbell sigma_W (exact, what production injects);
corr = sqrt(Var_w,clip1 - shot^2) on the kappa_tot<=1 core mask (8 seeds),
both from data/sigma_partition_vs_kthr_bias_z1.npz, model domain k >= k_min only.

Layout: 2x2 — rows = (shot, corr), cols = (lin-lin, log-log); fits solid inside
the fit window, dashed extrapolation outside. Output: plots/shot_corr_kthr_fits.png
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

KMIN = 1.278e-7          # absolute background floor 1e-3 * kappa_thr(N=100, zs=1)
SHOT_FIT_KMAX = 1e-3     # linear regime, before the threshold enters the HMF knee
CORR_FIT_KMAX = 3e-2     # before the corr plateau (~0.1)
C_SHOT, C_CORR, C_FIT = "#0f4c81", "#5b8fbe", "#d97706"

d = np.load(ROOT / "data" / "sigma_partition_vs_kthr_bias_z1.npz")
kthr = d["kthr"]
shot_map = {round(4 * np.log10(k)): s
            for k, s in zip(d["kthr_analytic"], d["sigW_shot_analytic"])}
shot = np.array([shot_map[round(4 * np.log10(k))] for k in kthr])
var_w_seed = d["clip1_per_seed"][:, :, 0]              # (nseed, nkt)
corr_seed = np.sqrt(np.maximum(var_w_seed - shot**2, 0.0))
corr = np.sqrt(np.maximum(d["var_w_clip1"] - shot**2, 0.0))
corr_sem = corr_seed.std(0, ddof=1) / np.sqrt(corr_seed.shape[0])

core = kthr >= KMIN
k, sh, co, co_sem = kthr[core], shot[core], corr[core], corr_sem[core]

# ---- fits (variance space for shot, sigma-vs-ln for corr) ----
w = k <= SHOT_FIT_KMAX
x, y = k[w] - KMIN, sh[w] ** 2
A = (x * y).sum() / (x * x).sum()
res_sh = np.abs(np.sqrt(A * x) / sh[w] - 1)

w = k <= CORR_FIT_KMAX
x, y = np.log(k[w] / KMIN), co[w]
sqrtB = (x * y).sum() / (x * x).sum()
res_co = np.abs(sqrtB * x / y - 1)

print(f"shot: Var = A(k-kmin), A = {A:.4e}  (window k<={SHOT_FIT_KMAX:.0e}; "
      f"rms dev in sigma {np.sqrt((res_sh**2).mean())*100:.1f}%, max {res_sh.max()*100:.1f}%)")
print(f"corr: sigma = sqrt(B) ln(k/kmin), sqrt(B) = {sqrtB:.4e}, B = {sqrtB**2:.3e}  "
      f"(window k<={CORR_FIT_KMAX:.0e}; rms {np.sqrt((res_co**2).mean())*100:.1f}%, "
      f"max {res_co.max()*100:.1f}%)")

kf = np.geomspace(KMIN * 1.0001, k.max(), 600)
fit_sh = np.sqrt(A * (kf - KMIN))
fit_co = sqrtB * np.log(kf / KMIN)

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.86,
                    hspace=0.42, wspace=0.30)

lab_sh = r"fit $\sigma^2 = A\,(\kappa_{\rm thr}-\kappa_{\rm min})$"
lab_co = r"fit $\sigma = \sqrt{B}\,\ln(\kappa_{\rm thr}/\kappa_{\rm min})$"

def panel(ax, comp, sem, fit, color, dlab, flab, fit_kmax, loglog, xmax=None):
    m = np.ones_like(k, bool) if xmax is None else (k <= xmax)
    ax.errorbar(k[m], comp[m], yerr=(None if sem is None else sem[m]),
                fmt="o", ms=2.6, lw=0.8, color=color, label=dlab, zorder=5)
    win = kf <= fit_kmax
    fm = np.ones_like(kf, bool) if xmax is None else (kf <= xmax)
    ax.plot(kf[win & fm], fit[win & fm], color=C_FIT, lw=1.4, label=flab, zorder=4)
    ax.plot(kf[~win & fm], fit[~win & fm], color=C_FIT, lw=1.1, ls="--",
            alpha=0.7, zorder=4)
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa_{\rm threshold}$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")
    ax.grid(False)

# shot row (analytic curve: no error bars)
panel(axes[0, 0], sh, None, fit_sh, C_SHOT, "weak (shot)", lab_sh,
      SHOT_FIT_KMAX, loglog=False, xmax=1e-2)
axes[0, 0].set_ylim(0, 1.15 * sh[k <= 1e-2].max())
panel(axes[0, 1], sh, None, fit_sh, C_SHOT, "weak (shot)", lab_sh,
      SHOT_FIT_KMAX, loglog=True)
axes[0, 1].set_ylim(0.5 * sh.min(), 3 * sh.max())
axes[0, 0].set_title("shot — lin-lin (zoom)", fontsize=8)
axes[0, 1].set_title("shot — log-log", fontsize=8)

# corr row (measured: SEM error bars)
panel(axes[1, 0], co, co_sem, fit_co, C_CORR, "weak (corr.)", lab_co,
      CORR_FIT_KMAX, loglog=False, xmax=3e-2)
axes[1, 0].set_ylim(0, 1.15 * co[k <= 3e-2].max())
panel(axes[1, 1], co, co_sem, fit_co, C_CORR, "weak (corr.)", lab_co,
      CORR_FIT_KMAX, loglog=True)
axes[1, 1].set_ylim(0.5 * co[co > 0].min(), 3 * co.max())
axes[1, 0].set_title("corr. — lin-lin (zoom)", fontsize=8)
axes[1, 1].set_title("corr. — log-log", fontsize=8)

for ax in axes.ravel():
    ax.legend(fontsize=6, loc="best", handlelength=1.8)

axes[0, 1].annotate(rf"$A = {A:.3e}$", xy=(0.05, 0.86), xycoords="axes fraction", fontsize=7)
axes[1, 1].annotate(rf"$\sqrt{{B}} = {sqrtB:.3e}$" "\n" rf"$\kappa_{{\rm min}} = 1.278\times10^{{-7}}$",
                    xy=(0.05, 0.80), xycoords="axes fraction", fontsize=7)

fig.suptitle(r"weak-arm components vs $\kappa_{\rm thr}$ ($z_s=1$, bias_model=1 + bias_weak)"
             "\n" r"shot: $\sigma^2 \propto \kappa_{\rm thr}$;"
             r"  corr.: $\sigma^2 \propto \ln^2(\kappa_{\rm thr}/\kappa_{\rm min})$",
             fontsize=9, y=0.98)

out = ROOT / "plots" / "shot_corr_kthr_fits.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print(f"Wrote {out}")
