"""
Sweep subhalo_factor for the playground d-gate prototype and plot retained kappa^2.

This is a single-host, single-ray diagnostic. It does not replace the full line-of-sight
Var(kappa), but it is the cleanest way to visualize how the current r-based proxy and the
corrected d-based gate behave as subhalo_factor changes on matched brute realizations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from playground.subhalo_factor_dgate_prototype import run_case  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--r-ray", type=float, default=300.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=[
            1e-5,
            3.1622776601683795e-5,
            1e-4,
            3.1622776601683795e-4,
            1e-3,
            3.1622776601683795e-3,
            1e-2,
            3.1622776601683795e-2,
            1e-1,
            3.1622776601683795e-1,
            1.0,
            3.1622776601683795,
            10.0,
            31.622776601683793,
            100.0,
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "playground" / "subhalo_factor_dgate_scan_r300.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "playground" / "subhalo_factor_dgate_scan_r300.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for factor in args.factors:
        summary, _ = run_case(
            host_mass=args.host_mass,
            z_lens=args.z_lens,
            z_source=args.z_source,
            r_ray=args.r_ray,
            subhalo_factor=float(factor),
            kappa_thr_host=args.kappa_thr_host,
            m_floor=args.m_floor,
            seeds=args.seeds,
        )
        rows.append(summary)
        print(
            f"factor={factor:.4g} "
            f"k2_brute={summary.get('k2_brute_mean', np.nan):.6e} "
            f"k2_proxy={summary.get('k2_proxy_mean', np.nan):.6e} "
            f"k2_truth={summary.get('k2_truth_mean', np.nan):.6e}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    factors = np.array([row["subhalo_factor"] for row in rows], dtype=float)
    k2_brute = np.array([row.get("k2_brute_mean", np.nan) for row in rows], dtype=float)
    k2_proxy = np.array([row.get("k2_proxy_mean", np.nan) for row in rows], dtype=float)
    k2_truth = np.array([row.get("k2_truth_mean", np.nan) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(factors, k2_brute, marker="o", lw=2.0, color="0.45", label="brute")
    ax.plot(factors, k2_proxy, marker="o", lw=2.0, color="#2563eb", label="proxy-r")
    ax.plot(factors, k2_truth, marker="o", lw=2.0, color="#dc2626", label="truth-d")
    ax.set_xscale("log")
    ax.set_xlabel("subhalo_factor")
    ax.set_ylabel("matched-realization $\\sum \\kappa^2$")
    ax.set_title("Single-host variance proxy vs subhalo_factor")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.suptitle(
        f"M_host={args.host_mass:.1e} Msun, z_l={args.z_lens:g}, z_s={args.z_source:g}, "
        f"r_ray={args.r_ray:.0f} kpc, seeds={len(args.seeds)}"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
