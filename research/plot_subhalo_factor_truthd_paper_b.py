"""
Paper-style panel-B comparison using the paired playground backend.

This keeps the styling and general framing of scripts/plot_subhalo_factor_paper.py
but replaces the zs=1 production backend with the paired single-host playground scan:

  sigma_sub^2(kappa) = Var(kappa) - Var(kappa_nosub)

The output overlays:
  1. brute-force reference band from the playground scan
  2. proxy-r curve (current geometry proxy)
  3. truth-d curve (corrected clump-ray geometry)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
DATA = ROOT / "playground" / "subhalo_factor_paired_area_scan_smooth.json"
OUT_BASE = ROOT / "research" / "subhalo_factor_truthd_paper_b"
ADOPT = 1.0e-5

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "cm",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})


def log_format(x, pos):
    val = np.log10(x)
    if np.isclose(val, np.round(val)):
        exponent = int(np.round(val))
        if exponent == 0:
            return "1"
        return rf"$10^{{{exponent}}}$"
    return f"{x:g}"


def main() -> None:
    rows = json.loads(DATA.read_text())
    rows = sorted(rows, key=lambda row: row["subhalo_factor"])

    factors = np.array([row["subhalo_factor"] for row in rows], dtype=float)
    ex_brute = np.array([row["excess_brute"] for row in rows], dtype=float)
    ex_proxy = np.array([row["excess_proxy"] for row in rows], dtype=float)
    ex_truth = np.array([row["excess_truth"] for row in rows], dtype=float)

    brute_ref = float(np.mean(ex_brute))
    brute_scatter = float(np.std(ex_brute, ddof=1)) if len(ex_brute) > 1 else 0.0
    host_mass = rows[0]["host_mass"]
    z_lens = rows[0]["z_lens"]
    z_source = rows[0]["z_source"]
    nrays = rows[0]["nrays"]
    nseeds = rows[0]["nseeds"]

    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    ax.axhspan(brute_ref - 2.0 * brute_scatter, brute_ref + 2.0 * brute_scatter, color="0.85", zorder=0)
    ax.axhline(
        brute_ref,
        color="0.45",
        ls="--",
        lw=1.0,
        label=r"brute force ($\pm 2\sigma$ band)",
    )
    ax.errorbar(
        factors,
        ex_proxy,
        yerr=None,
        marker="o",
        ms=3.4,
        lw=1.2,
        color="#3b5bdb",
        zorder=3,
        label=r"proxy-$r$",
    )
    ax.errorbar(
        factors,
        ex_truth,
        yerr=None,
        marker="o",
        ms=3.4,
        lw=1.2,
        color="#d9480f",
        zorder=4,
        label=r"truth-$d$",
    )
    ax.axvline(ADOPT, color="#c92a2a", ls=":", lw=1.3)
    ax.text(
        ADOPT * 1.65,
        brute_ref * 0.05,
        r"adopted $10^{-5}$",
        color="#c92a2a",
        fontsize=8.5,
        rotation=90,
        va="bottom",
        ha="left",
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(log_format))
    ax.set_xlabel(r"$\epsilon_{\rm sub}$")
    ax.set_ylabel(r"$\sigma^2_{\rm sub}(\kappa)$")
    ax.set_title(
        rf"Single-host paired scan: $M_h={host_mass:.0e}$, $z_l={z_lens:g}$, $z_s={z_source:g}$",
        fontsize=9.5,
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {OUT_BASE.with_suffix('.png')}")
    print(f"saved {OUT_BASE.with_suffix('.pdf')}")
    print(
        f"backend: paired playground scan with nrays={nrays}, nseeds={nseeds}, "
        f"M_host={host_mass:.3e}, z_l={z_lens:g}, z_s={z_source:g}"
    )


if __name__ == "__main__":
    main()
