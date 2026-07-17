#!/usr/bin/env python3
"""Sweep legacy-vs-flat PDF differences over source redshift.

This compares:
  - legacy fixed-<N> threshold (kappathr_flat = -1)
  - flat fixed threshold (user-chosen kappathr_flat)

using the live pybind sampler, and records summary diagnostics versus z_s.

Outputs:
  - data/results/kappathr_pdf_gap_sweep_{tag}.csv
  - plots/kappathr_pdf_gap_sweep_{tag}.png
  - plots/kappathr_pdf_gap_sweep_{tag}.pdf
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
    parser.add_argument("--flat-kthr", type=float, default=1.0e-4, dest="flat_kthr")
    parser.add_argument("--zmin", type=float, default=0.2)
    parser.add_argument("--zmax", type=float, default=10.0)
    parser.add_argument("--num", type=int, default=10,
                        help="Number of log-spaced z_s values in the sweep.")
    parser.add_argument("--nsamples", type=int, default=100000,
                        help="Samples per model at each z_s.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-m", type=float, default=0.315, dest="omega_m")
    parser.add_argument("--sigma8", type=float, default=0.811)
    return parser.parse_args()


def density_hist(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(samples, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=float)
    return counts / (total * widths)


def js_divergence(p: np.ndarray, q: np.ndarray, widths: np.ndarray) -> float:
    eps = 1.0e-300
    pn = np.clip(p, eps, None)
    qn = np.clip(q, eps, None)
    pn = pn / np.sum(pn * widths)
    qn = qn / np.sum(qn * widths)
    m = 0.5 * (pn + qn)
    kl_pm = np.sum(pn * np.log(pn / m) * widths)
    kl_qm = np.sum(qn * np.log(qn / m) * widths)
    return 0.5 * (kl_pm + kl_qm)


def tagify_kappa(kappa: float) -> str:
    sci = f"{kappa:.0e}"
    return sci.replace("-", "m").replace("+", "").replace(".", "p")


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "build"))

    import gwlensing  # pylint: disable=import-error

    zs = np.geomspace(args.zmin, args.zmax, args.num)
    rows = []

    for i, z in enumerate(zs):
        seed = args.seed + i
        run_common = dict(
            z=float(z),
            h=args.h,
            OmegaM=args.omega_m,
            sigma8=args.sigma8,
            nsamples=args.nsamples,
            seed=seed,
            strict_weak_lensing=False,
            subhalo=False,
        )
        legacy = gwlensing.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=-1.0)
        flat = gwlensing.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=args.flat_kthr)

        lnmu_legacy = np.asarray(legacy["lnmu"])
        lnmu_flat = np.asarray(flat["lnmu"])
        lo = min(np.quantile(lnmu_legacy, 1e-4), np.quantile(lnmu_flat, 1e-4))
        hi = max(np.quantile(lnmu_legacy, 1 - 1e-4), np.quantile(lnmu_flat, 1 - 1e-4))
        pad = 0.08 * (hi - lo)
        edges = np.linspace(lo - pad, hi + pad, 121)
        widths = np.diff(edges)
        pdf_legacy = density_hist(lnmu_legacy, edges)
        pdf_flat = density_hist(lnmu_flat, edges)
        jsd = js_divergence(pdf_legacy, pdf_flat, widths)
        legacy_kthr = gwlensing.get_kappa_threshold(float(z), args.h, args.omega_m, args.sigma8, 100)
        expected_flat_n = gwlensing.get_expected_halo_count(float(z), args.h, args.omega_m, args.sigma8, args.flat_kthr)
        expected_legacy_n = 100.0

        rows.append({
            "z": float(z),
            "legacy_kthr": float(legacy_kthr),
            "flat_kthr": float(args.flat_kthr),
            "js_divergence": float(jsd),
            "legacy_lnmu_std": float(np.std(lnmu_legacy)),
            "flat_lnmu_std": float(np.std(lnmu_flat)),
            "legacy_q999": float(np.quantile(lnmu_legacy, 0.999)),
            "flat_q999": float(np.quantile(lnmu_flat, 0.999)),
            "legacy_invalid": int(dict(legacy["invalid_stats"])["invalid_samples"]),
            "flat_invalid": int(dict(flat["invalid_stats"])["invalid_samples"]),
            "expected_legacy_n": expected_legacy_n,
            "expected_flat_n": float(expected_flat_n),
        })
        print(f"z={z:.3g} JSD={jsd:.3e} legacy_kthr={legacy_kthr:.3e} flat_N={expected_flat_n:.3g}")

    results_dir = repo / "data" / "results"
    plots_dir = repo / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    tag = tagify_kappa(args.flat_kthr)
    csv_path = results_dir / f"kappathr_pdf_gap_sweep_{tag}.csv"
    png_path = plots_dir / f"kappathr_pdf_gap_sweep_{tag}.png"
    pdf_path = plots_dir / f"kappathr_pdf_gap_sweep_{tag}.pdf"

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    z = np.array([r["z"] for r in rows])
    legacy_kthr = np.array([r["legacy_kthr"] for r in rows])
    jsd = np.array([r["js_divergence"] for r in rows])
    nflat = np.array([r["expected_flat_n"] for r in rows])
    std_legacy = np.array([r["legacy_lnmu_std"] for r in rows])
    std_flat = np.array([r["flat_lnmu_std"] for r in rows])

    ink, muted, base, grid = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
    blue, red, green = "#2a78d6", "#d94b3d", "#2f8f5b"
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": base,
        "font.size": 10.5,
    })

    fig, axs = plt.subplots(3, 1, figsize=(8.3, 9.4), sharex=True)

    axs[0].plot(z, legacy_kthr, color=blue, lw=2.0, label=r"legacy solved $\kappa_{\rm thr}(z_s)$")
    axs[0].axhline(args.flat_kthr, color=red, lw=1.5, ls="--",
                   label=rf"flat $\kappa_{{\rm thr}}={args.flat_kthr:.0e}$")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$\kappa_{\rm thr}$")
    axs[0].legend(frameon=False, fontsize=9.3, loc="upper left")
    axs[0].grid(color=grid, lw=0.7, which="both")

    axs[1].plot(z, nflat, color=green, lw=2.0, label=r"flat expected $\langle N \rangle$")
    axs[1].axhline(100.0, color=blue, lw=1.5, ls="--", label=r"legacy $\langle N \rangle = 100$")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"explicit halos $\langle N \rangle$")
    axs[1].legend(frameon=False, fontsize=9.3, loc="upper left")
    axs[1].grid(color=grid, lw=0.7, which="both")

    axs[2].plot(z, jsd, color=ink, lw=2.0, label="JSD(flat, legacy)")
    axs[2].plot(z, std_flat / std_legacy, color=red, lw=1.5, ls="--",
                label=r"$\sigma_{\ln\mu}^{\rm flat} / \sigma_{\ln\mu}^{\rm legacy}$")
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel(r"source redshift $z_s$")
    axs[2].set_ylabel("PDF gap / width ratio")
    axs[2].legend(frameon=False, fontsize=9.3, loc="upper right")
    axs[2].grid(color=grid, lw=0.7, which="both")

    fig.suptitle(rf"Legacy vs flat-threshold PDF sweep ($\kappa_{{\rm thr}}={args.flat_kthr:.0e}$)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=220)

    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
