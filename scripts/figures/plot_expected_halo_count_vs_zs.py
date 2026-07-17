#!/usr/bin/env python3
"""Plot expected explicit host-halo count vs source redshift for fixed kappa_thr.

Uses the live pybind binding `gwlensing.get_expected_halo_count`, i.e. the same
C++ `NhfNFW(z_s, kappa_thr)` used by the simulator.

Writes:
  - plots/expected_halo_count_vs_zs_kthr_{tag}.csv
  - plots/expected_halo_count_vs_zs_kthr_{tag}.png
  - plots/expected_halo_count_vs_zs_kthr_{tag}.pdf
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
    parser.add_argument("--kappathr", type=float, default=1.0e-4,
                        help="Fixed explicit-halo threshold.")
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


def tagify_kappa(kappa: float) -> str:
    sci = f"{kappa:.0e}"
    return sci.replace("-", "m").replace("+", "").replace(".", "p")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "build"))

    import gwlensing  # pylint: disable=import-error

    zs = np.geomspace(args.zmin, args.zmax, args.num)
    nh = np.array([
        gwlensing.get_expected_halo_count(
            float(z), args.h, args.omega_m, args.sigma8, args.kappathr,
            args.As, args.omega_b, args.zeq, args.ns
        )
        for z in zs
    ])

    plot_dir = repo / "plots"
    plot_dir.mkdir(exist_ok=True)
    tag = tagify_kappa(args.kappathr)
    csv_path = plot_dir / f"expected_halo_count_vs_zs_kthr_{tag}.csv"
    png_path = plot_dir / f"expected_halo_count_vs_zs_kthr_{tag}.png"
    pdf_path = plot_dir / f"expected_halo_count_vs_zs_kthr_{tag}.pdf"

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["zs", "expected_halo_count", "kappa_thr"])
        for z, n in zip(zs, nh):
            writer.writerow([f"{z:.12g}", f"{n:.12g}", f"{args.kappathr:.12g}"])

    ink, muted, base, grid = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
    blue, red = "#2a78d6", "#d94b3d"
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": base,
        "font.size": 10.5,
    })

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(zs, nh, color=blue, lw=2.2,
            label=rf"fixed $\kappa_{{\rm thr}}={args.kappathr:.0e}$")

    anchor_zs = np.array([0.2, 0.5, 1.0, 2.0, 10.0])
    anchor_nh = np.array([
        gwlensing.get_expected_halo_count(
            float(z), args.h, args.omega_m, args.sigma8, args.kappathr,
            args.As, args.omega_b, args.zeq, args.ns
        )
        for z in anchor_zs
    ])
    ax.scatter(anchor_zs, anchor_nh, color=red, s=24, zorder=3)
    for z, n in zip(anchor_zs, anchor_nh):
        ax.annotate(f"z={z:g}\n{n:.2g}", (z, n),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, color=muted)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"source redshift $z_s$")
    ax.set_ylabel(r"expected explicit host halos $\langle N \rangle$")
    ax.set_title(r"$\langle N \rangle(z_s)$ for fixed explicit-halo threshold")
    ax.grid(color=grid, lw=0.7, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.3, loc="upper left")
    fig.text(
        0.5, 0.01,
        rf"$h={args.h}, \Omega_M={args.omega_m}, \sigma_8={args.sigma8}, \kappa_{{\rm thr}}={args.kappathr:.0e}$",
        ha="center", color=ink, fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=220)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
