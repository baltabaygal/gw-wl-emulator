#!/usr/bin/env python3
"""
Paper figure (clustering subsection): dependence of the magnification PDF on the
smoothing scale R_s.

Backs the draft's "We show later how changing $R_s$ impacts our results".
Plots the clustering contribution to the width of P(ln mu),
Var(ln mu; R_s) / Var(ln mu; no clustering), against R_s, for three source
redshifts, with the production value R_s = 20 Mpc marked.

Arms are the FULL production configuration (spherical top-hat, bias_window = 1,
WITH the conditional sub-threshold arm of Sec. II.A.3, bias_weak = True) — the
counts-only arms understate the effect by roughly a factor two at R_s = 20 Mpc
and must not be quoted here (see data/results/bias_window/rs_dependence.md).

Data: data/results/bias_window/summary.npz, produced by
  python3 scripts/convergence/bias_window_scan.py mc -j 8
  python3 scripts/convergence/bias_window_scan.py report
(16 shards x 15k = 240k realizations/arm, kappa_anchor=1, fixed-<N>=100,
subhalo off; Var on a 10-sd median clip — raw Var is dominated by rare
kappa >~ 1 rays.)

Run:
  python3 paper_prod/scripts/plot_fig_clustering_rs.py
Outputs: plots/fig_clustering_rs.{pdf,png}
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper_prod" / "scripts"))

from paper_prod.plot_style import apply_style
from plot_fig_subhalo_population import guard_broken_latex

SUMMARY = REPO / "data" / "results" / "bias_window" / "summary.npz"
RS_GRID = [5.0, 10.0, 20.0, 40.0, 80.0]        # Mpc
RS_PROD = 20.0                                  # Mpc, production smoothing scale
ZS_LIST = [0.5, 1.0, 5.0]

# z_s is ORDINAL, so a single-hue sequential ramp (light -> dark) rather than
# categorical hues: it reads as "deeper source" at a glance and survives B/W
# print. Built around the paper's primary blue. Validated with the dataviz
# checker: CVD separation dE 16.3, normal-vision 16.5, contrast >= 3:1 on every
# step (thin lines need it). Line styles carry the same ordering redundantly.
ZS_STYLE = {0.5: ("#5590C4", ":", 0.9, "o"),
            1.0: ("#0C5DA5", "--", 1.0, "s"),
            5.0: ("#062A5C", "-", 1.1, "^")}


def main():
    apply_style()
    guard_broken_latex()
    import matplotlib.pyplot as plt

    if not SUMMARY.exists():
        sys.exit(f"missing {SUMMARY} — run bias_window_scan.py mc, then report")
    d = np.load(SUMMARY, allow_pickle=True)

    fig, ax = plt.subplots(figsize=(3.37, 2.7))
    fig.subplots_adjust(left=0.175, right=0.965, bottom=0.175, top=0.965)

    for zs in ZS_LIST:
        nb = float(d[f"z{zs:g}_nobias_var_clip"])
        ratio = np.array([float(d[f"z{zs:g}_rw{int(R):02d}_var_clip"]) / nb
                          for R in RS_GRID])
        col, ls, lw, mk = ZS_STYLE[zs]
        ax.plot(RS_GRID, ratio, ls=ls, lw=lw, color=col, marker=mk, ms=3.0,
                label=rf"$z_s = {zs:g}$")
        print(f"z_s = {zs:g}: " + "  ".join(
            f"R_s={R:.0f}: {v:.3f}" for R, v in zip(RS_GRID, ratio)))

    ax.axvline(RS_PROD, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.axhline(1.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax.annotate(r"$R_s = 20\,{\rm Mpc}$", xy=(RS_PROD, 1.86),
                xytext=(-3, 0), textcoords="offset points",
                rotation=90, ha="right", va="top", fontsize=6, color="0.4")
    ax.set_xscale("log")
    ax.set_xlim(4.2, 95.0)
    ax.set_ylim(0.97, 2.06)
    ax.set_xticks([5, 10, 20, 40, 80])
    ax.get_xaxis().set_major_formatter(
        __import__("matplotlib").ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"$R_s\ [{\rm Mpc}]$")
    ax.set_ylabel(r"${\rm Var}(\ln\mu)\,/\,{\rm Var}(\ln\mu)|_{\rm no\ clust.}$")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    for ext in ("pdf", "png"):
        fig.savefig(REPO / "plots" / f"fig_clustering_rs.{ext}", dpi=300)
    print("wrote plots/fig_clustering_rs.{pdf,png}")


if __name__ == "__main__":
    main()
