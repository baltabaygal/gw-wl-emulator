#!/usr/bin/env python3
"""Decision plot: fixed explicit-halo count versus fixed kappa threshold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "data" / "results" / "kappathr_pdf_acceptance_summary.json"
OUT = ROOT / "plots" / "kappathr_rule_decision"
ZS = (0.2, 1.0, 10.0)
RULES = ("flat_1e-4", "flat_1e-3")
LABELS = {"flat_1e-4": "flat kappa = 1e-4", "flat_1e-3": "flat kappa = 1e-3"}

# Live simulator values from get_expected_halo_count at the acceptance cosmology.
EXPECTED_COUNTS = {
    "adaptive": (100.0, 100.0, 100.0),
    "flat_1e-4": (5.9161944098, 129.8910852338, 1848.7281819322),
    "flat_1e-3": (0.4415073719, 10.8365571237, 145.7844993294),
}

TOKENS = {
    "surface": "#FCFCFD", "panel": "#FFFFFF", "ink": "#1F2430",
    "muted": "#6F768A", "grid": "#E6E8F0", "axis": "#D7DBE7",
}
COLORS = {"flat_1e-4": "#F0986E", "flat_1e-3": "#A3BEFA"}
EDGES = {"flat_1e-4": "#804126", "flat_1e-3": "#2E4780"}
MARKERS = {"flat_1e-4": "o", "flat_1e-3": "s"}


def style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": TOKENS["surface"], "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"], "axes.labelcolor": TOKENS["ink"],
        "grid.color": TOKENS["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
    })


def dot_series(ax, x: np.ndarray, y: np.ndarray, rule: str, label: str | None = None) -> None:
    sns.scatterplot(x=x, y=y, ax=ax, s=64, marker=MARKERS[rule],
                    color=COLORS[rule], edgecolor=EDGES[rule], linewidth=1.0,
                    label=label, zorder=3)


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    style()
    x = np.arange(len(ZS), dtype=float)
    offsets = {"flat_1e-4": -0.08, "flat_1e-3": 0.08}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    ax_kl, ax_tail, ax_count, ax_runtime = axes.flat

    for rule in RULES:
        kl = np.array([data["results"][str(z)]["rules"][rule]["kl_to_adaptive"] for z in ZS])
        dot_series(ax_kl, x + offsets[rule], kl, rule, LABELS[rule])
    ax_kl.axhline(1.0e-3, color=TOKENS["ink"], ls="--", lw=1.0, label="acceptance limit")
    ax_kl.set_yscale("log")
    ax_kl.set_ylabel("KL(flat || adaptive)")
    ax_kl.set_title("A. Distribution gap")
    ax_kl.legend(frameon=False, fontsize=8.5, loc="upper center")

    for rule in RULES:
        ratios, lows, highs = [], [], []
        for z in ZS:
            reference = data["results"][str(z)]["rules"]["adaptive"]["tail"]["q99"]
            candidate = data["results"][str(z)]["rules"][rule]["tail"]["q99"]
            ratio = candidate["value"] / reference["value"]
            conservative_lo = candidate["ci95"][0] / reference["ci95"][1]
            conservative_hi = candidate["ci95"][1] / reference["ci95"][0]
            ratios.append(ratio)
            lows.append(ratio - conservative_lo)
            highs.append(conservative_hi - ratio)
        xp = x + offsets[rule]
        ax_tail.vlines(xp, np.array(ratios) - lows, np.array(ratios) + highs,
                       color=EDGES[rule], lw=1.1, zorder=2)
        dot_series(ax_tail, xp, np.array(ratios), rule)
    ax_tail.axhline(1.0, color=TOKENS["ink"], ls="--", lw=1.0)
    ax_tail.set_ylabel("q99(mu) / adaptive")
    ax_tail.set_title("B. Science-bearing tail (95% CI)")

    for rule in ("adaptive",) + RULES:
        if rule == "adaptive":
            sns.scatterplot(x=x, y=EXPECTED_COUNTS[rule], ax=ax_count, s=55,
                            marker="D", color="#C5CAD3", edgecolor="#464C55",
                            linewidth=1.0, label="adaptive N = 100", zorder=3)
        else:
            dot_series(ax_count, x + offsets[rule], np.array(EXPECTED_COUNTS[rule]), rule, LABELS[rule])
    ax_count.set_yscale("log")
    ax_count.set_ylabel("Expected explicit halos / sightline")
    ax_count.set_title("C. Computational load driver")
    ax_count.legend(frameon=False, fontsize=8.3, loc="upper left")

    adaptive_runtime = np.array([
        data["results"][str(z)]["rules"]["adaptive"]["runtime_seconds"] for z in ZS])
    for rule in RULES:
        runtime = np.array([
            data["results"][str(z)]["rules"][rule]["runtime_seconds"] for z in ZS])
        dot_series(ax_runtime, x + offsets[rule], runtime / adaptive_runtime, rule, LABELS[rule])
    ax_runtime.axhline(1.0, color=TOKENS["ink"], ls="--", lw=1.0, label="adaptive runtime")
    ax_runtime.set_ylabel("Wall time / adaptive")
    ax_runtime.set_title("D. Measured production-run cost")
    ax_runtime.legend(frameon=False, fontsize=8.5, loc="upper left")

    for ax in axes.flat:
        ax.set_xticks(x, ("0.2", "1", "10"))
        ax.set_xlabel("Source redshift z_s")
        ax.grid(axis="x", visible=False)
        ax.tick_params(colors=TOKENS["ink"])

    fig.subplots_adjust(top=0.82, hspace=0.42, wspace=0.28, left=0.09, right=0.98, bottom=0.09)
    fig.text(0.09, 0.965, "Adaptive N = 100 is the safer production rule today",
             ha="left", va="top", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(0.09, 0.925,
             "Subhalo off; seed 123; 1,000,000 draws at z_s=0.2/1 and 300,000 at z_s=10. "
             "Flat thresholds fail redshift-robust PDF fidelity and do not deliver a uniform speed advantage.",
             ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])
    fig.text(0.09, 0.025,
             "Source: data/results/kappathr_pdf_acceptance_summary.json. Runtime is single-run wall time; "
             "explicit counts are simulator expectations.",
             ha="left", va="bottom", fontsize=8, color=TOKENS["muted"])

    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None,
                    facecolor=TOKENS["surface"])
    plt.close(fig)
    print(f"wrote {OUT}.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
