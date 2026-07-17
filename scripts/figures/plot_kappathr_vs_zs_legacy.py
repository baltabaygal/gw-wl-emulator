#!/usr/bin/env python3
"""Plot the legacy kappa_thr(z_s) curve for the fixed-<N> host-halo rule.

This uses the live pybind binding `gwlensing.get_kappa_threshold`, which solves
NhfNFW(kappa_thr; z_s) = Nhalos with the same C++ logic as the production code's
legacy path (`kappathr_flat <= 0`).

Writes:
  - plots/kappathr_vs_zs_legacy.csv
  - plots/kappathr_vs_zs_legacy.png
  - plots/kappathr_vs_zs_legacy.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhalos", type=int, default=100,
                        help="Fixed expected number of resolved host halos.")
    parser.add_argument("--zmin", type=float, default=0.1,
                        help="Minimum source redshift.")
    parser.add_argument("--zmax", type=float, default=10.0,
                        help="Maximum source redshift.")
    parser.add_argument("--num", type=int, default=40,
                        help="Number of log-spaced z_s samples.")
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-m", type=float, default=0.315, dest="omega_m")
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--as", type=float, default=-1.0, dest="As")
    parser.add_argument("--omega-b", type=float, default=0.0493, dest="omega_b")
    parser.add_argument("--zeq", type=float, default=3402.0)
    parser.add_argument("--ns", type=float, default=0.965)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "build"))

    import gwlensing  # pylint: disable=import-error

    zs = np.geomspace(args.zmin, args.zmax, args.num)
    kthr = np.array([
        gwlensing.get_kappa_threshold(
            float(z), args.h, args.omega_m, args.sigma8, args.nhalos,
            args.As, args.omega_b, args.zeq, args.ns
        )
        for z in zs
    ])

    plot_dir = repo / "plots"
    plot_dir.mkdir(exist_ok=True)
    csv_path = plot_dir / "kappathr_vs_zs_legacy.csv"

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["zs", "kappa_thr", "Nhalos"])
        for z, k in zip(zs, kthr):
            writer.writerow([f"{z:.12g}", f"{k:.12g}", args.nhalos])

    ink, muted, base, grid = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
    blue, red = "#2a78d6", "#d94b3d"
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": base,
        "font.size": 10.5,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(zs, kthr, color=blue, lw=2.2,
            label=rf"legacy solve: $\langle N \rangle = {args.nhalos}$")

    anchor_zs = np.array([0.2, 1.0, 10.0])
    anchor_k = np.array([
        gwlensing.get_kappa_threshold(
            float(z), args.h, args.omega_m, args.sigma8, args.nhalos,
            args.As, args.omega_b, args.zeq, args.ns
        )
        for z in anchor_zs
    ])
    ax.scatter(anchor_zs, anchor_k, color=red, s=24, zorder=3)
    for z, k in zip(anchor_zs, anchor_k):
        ax.annotate(f"z={z:g}\n{k:.2e}", (z, k),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=muted)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"source redshift $z_s$")
    ax.set_ylabel(r"legacy threshold $\kappa_{\rm thr}$")
    ax.set_title(r"$\kappa_{\rm thr}(z_s)$ with fixed resolved-halo count")
    ax.grid(color=grid, lw=0.7, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    subtitle = (
        rf"$h={args.h}, \Omega_M={args.omega_m}, \sigma_8={args.sigma8}, "
        rf"N_{{\rm halos}}={args.nhalos}$"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", color=ink, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    for ext in ("png", "pdf"):
        fig.savefig(plot_dir / f"kappathr_vs_zs_legacy.{ext}", dpi=220)

    print(f"wrote {csv_path}")
    print(f"wrote {plot_dir / 'kappathr_vs_zs_legacy.png'}")
    print(f"wrote {plot_dir / 'kappathr_vs_zs_legacy.pdf'}")


if __name__ == "__main__":
    main()
