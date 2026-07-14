#!/usr/bin/env python3
"""Figure for the subhalo_factor PDF-level acceptance test (model 3 vs brute).

Reads cached samples + summary from data/results/subhalo_factor_jsd<tag>/
(written by scripts/convergence/subhalo_factor_jsd.py) and renders
plots/subhalo_factor_jsd<tag>.{png,pdf}:

  A: floor-subtracted JSD per config (z=1 vs z=5)
  B: P(ln mu) at z=5 — factor arms vs brute truth
  C: P(mu) log-log tail at z=5
  D: tail quantile ratios to truth (q99/q99.9/q99.99, z=5) with 95% CIs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

INK, MUTED, BASE, GRID = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
COLORS = {"f1em5": "#2a78d6", "f1em3": "#2e8b57", "f1em2": "#d94b3d",
          "f1em1": "#b8860b", "m1_f1em2": "#8a5fbf"}
LABELS = {"f1em5": r"m3, factor $10^{-5}$ (default)",
          "f1em3": r"m3, factor $10^{-3}$",
          "f1em2": r"m3, factor $10^{-2}$ (target)",
          "f1em1": r"m3, factor $10^{-1}$",
          "m1_f1em2": r"m1, factor $10^{-2}$ (no Wsub)"}
CONFIGS = ("f1em5", "f1em3", "f1em2", "f1em1", "m1_f1em2")


def density(x, edges):
    """Counts normalized by the FULL sample size (panel C plots a tail slice;
    per-curve renormalization over the slice would distort config-vs-truth
    offsets by each config's tail fraction)."""
    c, _ = np.histogram(x, bins=edges)
    w = np.diff(edges)
    return c / (x.size * w) if x.size else np.zeros_like(w, float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d = REPO / "data" / "results" / f"subhalo_factor_jsd{args.tag}"
    res = json.loads((d / "summary.json").read_text())

    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                         "axes.edgecolor": BASE, "font.size": 10.5})
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    axA, axB, axC, axD = axes

    # ---- A: floor-subtracted JSD --------------------------------------------
    groups = list(res)
    if not groups:
        raise SystemExit(f"no groups in {d / 'summary.json'} — run the sampler first")
    xg = np.arange(len(groups))
    width = 0.15
    for i, c in enumerate(CONFIGS):
        vals = [res[g]["configs"][c]["jsd_excess"] for g in groups]
        floors = [res[g]["configs"][c]["floor_pred"] for g in groups]
        axA.bar(xg + (i - 2) * width, vals, width, color=COLORS[c],
                label=LABELS[c], zorder=3)
        axA.plot(xg + (i - 2) * width, floors, "_", color=INK, ms=9, zorder=4)
    axA.set_xticks(xg)
    axA.set_xticklabels([f"$z_s={g[1:]}$" for g in groups])
    axA.set_yscale("log")
    axA.set_ylabel("JSD excess above floor [nats]")
    axA.set_title("A. Distribution gap to brute truth\n(dash = finite-sample floor)")
    axA.grid(color=GRID, lw=0.7, axis="y", which="both")
    axA.set_axisbelow(True)
    axA.legend(frameon=False, fontsize=8.2)

    # ---- B/C/D: focus group = z5 (z=1 tails all agree) ----------------------
    gk = "z5" if "z5" in res else groups[-1]
    glab = gk.replace("z", "$z_s{=}$", 1)
    smp = {c: np.load(d / f"{gk}_{c}.npy") for c in CONFIGS}
    truth = np.concatenate([np.load(d / f"{gk}_truthA.npy"),
                            np.load(d / f"{gk}_truthB.npy")])
    edges = np.asarray(res[gk]["edges"])
    xc = 0.5 * (edges[:-1] + edges[1:])
    axB.plot(xc, density(truth, edges), color=INK, lw=2.4, label="truth (brute)")
    for c in CONFIGS:
        axB.plot(xc, density(smp[c], edges), color=COLORS[c], lw=1.3,
                 ls="--" if c == "m1_f1em2" else "-", label=LABELS[c])
    axB.set_yscale("log")
    axB.set_xlabel(r"$\ln\mu$"); axB.set_ylabel(r"$dP/d\ln\mu$")
    axB.set_title(f"B. {glab}: $P(\\ln\\mu)$")
    axB.grid(color=GRID, lw=0.7); axB.set_axisbelow(True)
    axB.legend(frameon=False, fontsize=8.2)

    mu_edges = np.geomspace(1.5, max(np.exp(truth.max()), 100.0), 80)
    mc = np.sqrt(mu_edges[:-1] * mu_edges[1:])
    axC.plot(mc, density(np.exp(truth), mu_edges), color=INK, lw=2.4)
    for c in CONFIGS:
        axC.plot(mc, density(np.exp(smp[c]), mu_edges), color=COLORS[c], lw=1.3,
                 ls="--" if c == "m1_f1em2" else "-")
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xlabel(r"$\mu$"); axC.set_ylabel(r"$dP/d\mu$")
    axC.set_title(f"C. {glab}: magnification tail")
    axC.grid(color=GRID, lw=0.7, which="both"); axC.set_axisbelow(True)

    # ---- D: tail quantile ratios (focus group) ------------------------------
    qkeys = ("q99", "q999", "q9999")
    qlabs = ("$q_{99}$", "$q_{99.9}$", "$q_{99.99}$")
    xq = np.arange(len(qkeys))
    tq = {k: res[gk]["configs"]["truth"]["tails"][k] for k in qkeys}
    for i, c in enumerate(CONFIGS):
        rq = res[gk]["configs"][c]["tails"]
        ratio = np.array([rq[k][1] / tq[k][1] for k in qkeys])
        lo = np.array([rq[k][0] / tq[k][2] for k in qkeys])
        hi = np.array([rq[k][2] / tq[k][0] for k in qkeys])
        axD.errorbar(xq + (i - 2) * 0.12, ratio, yerr=[ratio - lo, hi - ratio],
                     fmt="o", color=COLORS[c], ms=5, lw=1.3, capsize=3)
    axD.axhline(1.0, color=INK, lw=1.2, ls="--")
    axD.set_xticks(xq); axD.set_xticklabels(qlabs)
    axD.set_ylabel("quantile / truth quantile")
    axD.set_title(f"D. {glab}: tail vs truth\n(bars = conservative 95% CI)")
    axD.grid(color=GRID, lw=0.7, axis="y"); axD.set_axisbelow(True)

    fig.suptitle("subhalo_factor PDF-level acceptance: model 3 vs brute truth "
                 "(fixed-$\\langle N\\rangle{=}100$ rule, m_floor=$10^7$)",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    out = REPO / "plots" / f"subhalo_factor_jsd{args.tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
