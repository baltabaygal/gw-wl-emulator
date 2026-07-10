#!/usr/bin/env python3
"""Figure for the subhalo-on threshold-rule JSD test.

Reads cached samples + summary from data/results/kappathr_subhalo_jsd<tag>/
(written by scripts/subhalo_gate/kappathr_subhalo_jsd.py) and renders
plots/kappathr_subhalo_jsd<tag>.{png,pdf}:

  A: floor-subtracted JSD per rule (z=1 vs z=10, subhalo on vs off)
  B: P(ln mu) at z=10, subhalo on — rules vs truth
  C: P(mu) log-log tail at z=10, subhalo on
  D: tail quantile ratios to truth (q99/q99.9/q99.99, z=10 on) with 95% CIs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

INK, MUTED, BASE, GRID = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
COLORS = {"flat_1em4": "#2e8b57", "flat_1em3": "#d94b3d", "fixedN": "#2a78d6"}
LABELS = {"flat_1em4": r"flat $\kappa_{\rm thr}=10^{-4}$",
          "flat_1em3": r"flat $\kappa_{\rm thr}=10^{-3}$",
          "fixedN": r"fixed $\langle N\rangle=100$"}
RULES = ("flat_1em4", "flat_1em3", "fixedN")


def density(x, edges):
    c, _ = np.histogram(x, bins=edges)
    w = np.diff(edges)
    return c / (c.sum() * w)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d = REPO / "data" / "results" / f"kappathr_subhalo_jsd{args.tag}"
    res = json.loads((d / "summary.json").read_text())

    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": BASE, "font.size": 10.5})
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    axA, axB, axC, axD = axes

    # ---- A: floor-subtracted JSD --------------------------------------------
    groups = [g for g in ("z1_on", "z1_off", "z10_on", "z10_off") if g in res]
    xg = np.arange(len(groups))
    width = 0.26
    for i, r in enumerate(RULES):
        vals = [res[g]["rules"][r]["jsd_excess"] for g in groups]
        floors = [res[g]["rules"][r]["floor_pred"] for g in groups]
        axA.bar(xg + (i - 1) * width, vals, width, color=COLORS[r],
                label=LABELS[r], zorder=3)
        axA.plot(xg + (i - 1) * width, floors, "_", color=INK, ms=10, zorder=4)
    axA.set_xticks(xg)
    axA.set_xticklabels([g.replace("_", "\nsubhalo ") for g in groups])
    axA.set_yscale("log")
    axA.set_ylabel("JSD excess above floor [nats]")
    axA.set_title("A. Distribution gap to $\\kappa_{\\rm thr}=3{\\times}10^{-5}$ truth\n"
                  "(dash = finite-sample floor)")
    axA.grid(color=GRID, lw=0.7, axis="y", which="both")
    axA.set_axisbelow(True)
    axA.legend(frameon=False, fontsize=8.6)

    # ---- B/C: z=10 subhalo-on PDFs ------------------------------------------
    gk = "z10_on"
    smp = {r: np.load(d / f"z10_{'on'}_{r}.npy") for r in RULES}
    truth = np.concatenate([np.load(d / "z10_on_truthA.npy"),
                            np.load(d / "z10_on_truthB.npy")])
    edges = np.asarray(res[gk]["edges"])
    xc = 0.5 * (edges[:-1] + edges[1:])
    axB.plot(xc, density(truth, edges), color=INK, lw=2.4, label="truth (3e-5)")
    for r in RULES:
        axB.plot(xc, density(smp[r], edges), color=COLORS[r], lw=1.5, label=LABELS[r])
    axB.set_yscale("log")
    axB.set_xlabel(r"$\ln\mu$"); axB.set_ylabel(r"$dP/d\ln\mu$")
    axB.set_title("B. $z_s=10$, subhalo ON: $P(\\ln\\mu)$")
    axB.grid(color=GRID, lw=0.7); axB.set_axisbelow(True)
    axB.legend(frameon=False, fontsize=8.6)

    mu_edges = np.geomspace(1.5, max(np.exp(truth.max()), 100.0), 80)
    mc = np.sqrt(mu_edges[:-1] * mu_edges[1:])
    axC.plot(mc, density(np.exp(truth), mu_edges), color=INK, lw=2.4)
    for r in RULES:
        axC.plot(mc, density(np.exp(smp[r]), mu_edges), color=COLORS[r], lw=1.5)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel(r"$\mu$"); axC.set_ylabel(r"$dP/d\mu$")
    axC.set_title("C. $z_s=10$, subhalo ON: magnification tail")
    axC.grid(color=GRID, lw=0.7, which="both"); axC.set_axisbelow(True)

    # ---- D: tail quantile ratios (z=10 on) ----------------------------------
    qkeys = ("q99", "q999", "q9999")
    qlabs = ("$q_{99}$", "$q_{99.9}$", "$q_{99.99}$")
    xq = np.arange(len(qkeys))
    tq = {k: res[gk]["rules"]["truth"]["tails"][k] for k in qkeys}
    for i, r in enumerate(RULES):
        rq = res[gk]["rules"][r]["tails"]
        ratio = np.array([rq[k][1] / tq[k][1] for k in qkeys])
        lo = np.array([rq[k][0] / tq[k][2] for k in qkeys])
        hi = np.array([rq[k][2] / tq[k][0] for k in qkeys])
        axD.errorbar(xq + (i - 1) * 0.15, ratio, yerr=[ratio - lo, hi - ratio],
                     fmt="o", color=COLORS[r], ms=6, lw=1.4, capsize=3)
    axD.axhline(1.0, color=INK, lw=1.2, ls="--")
    axD.set_xticks(xq); axD.set_xticklabels(qlabs)
    axD.set_ylabel("quantile / truth quantile")
    axD.set_title("D. $z_s=10$, subhalo ON: tail vs truth\n(bars = conservative 95% CI)")
    axD.grid(color=GRID, lw=0.7, axis="y"); axD.set_axisbelow(True)

    fig.suptitle("Threshold rule with substructure: subhalo_model=3, factor=1e-5; "
                 "truth = flat $\\kappa_{\\rm thr}=3{\\times}10^{-5}$ (split halves)",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / f"kappathr_subhalo_jsd{args.tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
