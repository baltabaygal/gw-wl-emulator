"""M_b vs typical lens mass per z_l bin — companion figure to mb_validity_map.py.

For each lens-redshift bin z_l (default grid, fixed-<N>=100 rule, halo only),
collapse the mass axis and plot two curves per z_s panel:
  - <M>   : weighted geometric mean lens mass of the halos in that z_l bin
  - <M_b> : same-weighted geometric mean of the bias mass scale M_b of the bin
with two weightings: barN (number of lens encounters, solid) and the linearized
bias clustering-variance contribution (barN*kbar*sigma_b)^2 (dashed — the cells
that actually drive the bias layer). Shaded band = barN-weighted q10-q90 of M_b.
The PBS claim is "M_b >> M": the vertical gap between same-style curves.

Run: /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/mb_profile_plot.py
Output: plots/mb_profile.png
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
from bias_field_prototype import Cosmo  # noqa: E402

ZS_LIST = [0.2, 1.0, 5.0, 10.0]
NHALOS = 100

C_MB, C_M = "#4269d0", "#e05c1f"      # blue = M_b, orange = M (CVD-safe pair)


def wquant(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(qs, cw, x)


def profile(C, zs):
    kt = C.find_kappathr(zs, NHALOS)
    T = C.cell_tables(zs, kt)
    M = C.Mlist[T["jM"]]
    wN = T["barN"]
    wV = (T["barN"] * T["kbar"] * T["sigma_old"]) ** 2
    rows = []
    for jz in np.unique(T["jz"]):
        s = T["jz"] == jz
        if wN[s].sum() <= 0:
            continue
        def gmean(x, w):
            return 10 ** (np.sum(w * np.log10(x)) / np.sum(w))
        lo, hi = 10 ** wquant(np.log10(T["Mb"][s]), wN[s], [0.10, 0.90])
        rows.append((C.zlist[jz],
                     gmean(M[s], wN[s]), gmean(T["Mb"][s], wN[s]),
                     gmean(M[s], wV[s]) if wV[s].sum() > 0 else np.nan,
                     gmean(T["Mb"][s], wV[s]) if wV[s].sum() > 0 else np.nan,
                     lo, hi,
                     wV[s].sum()))
    return kt, np.array(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=True,
                             constrained_layout=True)
    C = Cosmo()
    for ax, zs in zip(axes.ravel(), ZS_LIST):
        kt, P = profile(C, zs)
        zl, Mn, Mbn, Mv, Mbv, lo, hi, wv = P.T
        ax.fill_between(zl, lo, hi, color=C_MB, alpha=0.15, lw=0,
                        step="mid")
        ax.plot(zl, Mbn, color=C_MB, lw=1.2, marker="o", ms=3.2,
                label=r"$\langle M_b\rangle$ (per-encounter)")
        ax.plot(zl, Mn, color=C_M, lw=1.2, marker="o", ms=3.2,
                label=r"$\langle M\rangle$ lens (per-encounter)")
        ax.plot(zl, Mbv, color=C_MB, lw=1.6, ls="--",
                label=r"$\langle M_b\rangle$ (bias-var.-weighted)")
        ax.plot(zl, Mv, color=C_M, lw=1.6, ls="--",
                label=r"$\langle M\rangle$ (bias-var.-weighted)")
        # where the bias-variance weight lives along z_l (90% interval)
        cw = np.cumsum(wv) / wv.sum()
        z5, z95 = np.interp([0.05, 0.95], cw, zl)
        ax.axvspan(z5, z95, color="0.5", alpha=0.10, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$z_s = {zs:g}$   ($\\kappa_{{\\rm thr}}$ = {kt:.1e})",
                     fontsize=11)
        ax.grid(alpha=0.25, which="major")
        # annotate the ratio at the weight core
        jc = np.argmin(np.abs(cw - 0.5))
        ax.annotate(f"$M_b/M \\approx {Mbv[jc]/Mv[jc]:.1f}$ at weight core",
                    xy=(zl[jc], Mbv[jc]), xytext=(0.03, 0.95),
                    textcoords="axes fraction", fontsize=10, va="top")
    for ax in axes[-1]:
        ax.set_xlabel(r"lens redshift bin $z_l$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mass [$M_\odot$]")
    axes[0, 0].legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Bias mass scale $M_b$ vs typical lens mass $M$ per $z_l$ bin "
                 "(default grid, halo only)\nband: q10–q90 of $M_b$ in the bin; "
                 "grey span: 90% of the bias clustering-variance weight",
                 fontsize=12)
    out = REPO / "plots" / "mb_profile.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
