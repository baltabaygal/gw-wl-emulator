#!/usr/bin/env python3
"""Compare ln(mu) PDFs for legacy-vs-flat kappa-threshold rules.

By default this isolates the host-threshold effect:
  - legacy fixed-<N> rule: kappathr_flat = -1
  - current flat rule:     kappathr_flat = 1e-3
with subhalos disabled.

Outputs:
  - plots/kappathr_pdf_compare_z{z}.png
  - plots/kappathr_pdf_compare_z{z}.pdf
  - plots/kappathr_pdf_compare_z{z}_semilogy.png
  - plots/kappathr_pdf_compare_z{z}_semilogy.pdf
  - plots/kappathr_pdf_compare_z{z}_loglog.png
  - plots/kappathr_pdf_compare_z{z}_loglog.pdf
  - data/results/kappathr_pdf_compare_z{z}.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=0.5, help="Source redshift.")
    parser.add_argument("--nsamples", type=int, default=200000, help="Samples per model.")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for both runs.")
    parser.add_argument("--nbins", type=int, default=120, help="Histogram bins in ln(mu).")
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-m", type=float, default=0.315, dest="omega_m")
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--flat-kthr", type=float, default=1.0e-3, dest="flat_kthr")
    return parser.parse_args()


def density_hist(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(samples, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=float)
    return counts / (total * widths)


def finite_positive_floor(*arrays: np.ndarray) -> float:
    positives = [a[np.isfinite(a) & (a > 0.0)] for a in arrays]
    positives = [a for a in positives if a.size > 0]
    if not positives:
        return 1.0e-12
    return float(np.min(np.concatenate(positives)))


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


def summarize(name: str, lnmu: np.ndarray, invalid_stats: dict) -> dict:
    mu = np.exp(lnmu)
    out = {
        "name": name,
        "n_valid": int(lnmu.size),
        "lnmu_mean": float(np.mean(lnmu)),
        "lnmu_std": float(np.std(lnmu)),
        "mu_mean": float(np.mean(mu)),
        "mu_std": float(np.std(mu)),
        "q001": float(np.quantile(lnmu, 0.001)),
        "q01": float(np.quantile(lnmu, 0.01)),
        "q99": float(np.quantile(lnmu, 0.99)),
        "q999": float(np.quantile(lnmu, 0.999)),
        "invalid_stats": invalid_stats,
    }
    return out


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "build"))

    import gwlensing as gw  # pylint: disable=import-error

    run_common = dict(
        z=args.z,
        h=args.h,
        OmegaM=args.omega_m,
        sigma8=args.sigma8,
        nsamples=args.nsamples,
        seed=args.seed,
        strict_weak_lensing=False,
        subhalo=False,
    )

    legacy = gw.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=-1.0)
    flat = gw.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=args.flat_kthr)

    lnmu_legacy = np.asarray(legacy["lnmu"])
    lnmu_flat = np.asarray(flat["lnmu"])

    lo = min(np.quantile(lnmu_legacy, 1e-4), np.quantile(lnmu_flat, 1e-4))
    hi = max(np.quantile(lnmu_legacy, 1 - 1e-4), np.quantile(lnmu_flat, 1 - 1e-4))
    pad = 0.08 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    pdf_legacy = density_hist(lnmu_legacy, edges)
    pdf_flat = density_hist(lnmu_flat, edges)
    ratio = np.divide(pdf_flat, pdf_legacy, out=np.full_like(pdf_flat, np.nan), where=pdf_legacy > 0.0)
    jsd = js_divergence(pdf_legacy, pdf_flat, widths)

    mu_legacy = np.exp(lnmu_legacy)
    mu_flat = np.exp(lnmu_flat)
    mu_lo = min(np.quantile(mu_legacy, 1e-4), np.quantile(mu_flat, 1e-4))
    mu_hi = max(np.quantile(mu_legacy, 1 - 1e-4), np.quantile(mu_flat, 1 - 1e-4))
    mu_edges = np.geomspace(max(mu_lo * 0.9, 1.0e-6), mu_hi * 1.1, args.nbins + 1)
    mu_centers = np.sqrt(mu_edges[:-1] * mu_edges[1:])
    mu_pdf_legacy = density_hist(mu_legacy, mu_edges)
    mu_pdf_flat = density_hist(mu_flat, mu_edges)
    mu_ratio = np.divide(
        mu_pdf_flat, mu_pdf_legacy,
        out=np.full_like(mu_pdf_flat, np.nan),
        where=mu_pdf_legacy > 0.0,
    )

    legacy_kthr = gw.get_kappa_threshold(args.z, args.h, args.omega_m, args.sigma8, 100)

    results_dir = repo / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = repo / "plots"
    plots_dir.mkdir(exist_ok=True)

    tag = f"z{str(args.z).replace('.', 'p')}"
    npz_path = results_dir / f"kappathr_pdf_compare_{tag}.npz"
    png_path = plots_dir / f"kappathr_pdf_compare_{tag}.png"
    pdf_path = plots_dir / f"kappathr_pdf_compare_{tag}.pdf"
    semilogy_png_path = plots_dir / f"kappathr_pdf_compare_{tag}_semilogy.png"
    semilogy_pdf_path = plots_dir / f"kappathr_pdf_compare_{tag}_semilogy.pdf"
    loglog_png_path = plots_dir / f"kappathr_pdf_compare_{tag}_loglog.png"
    loglog_pdf_path = plots_dir / f"kappathr_pdf_compare_{tag}_loglog.pdf"

    summary = {
        "z": args.z,
        "nsamples": args.nsamples,
        "seed": args.seed,
        "legacy_kappathr": float(legacy_kthr),
        "flat_kappathr": float(args.flat_kthr),
        "js_divergence": float(jsd),
        "legacy": summarize("legacy", lnmu_legacy, dict(legacy["invalid_stats"])),
        "flat": summarize("flat", lnmu_flat, dict(flat["invalid_stats"])),
    }

    np.savez(
        npz_path,
        lnmu_legacy=lnmu_legacy,
        lnmu_flat=lnmu_flat,
        mu_legacy=mu_legacy,
        mu_flat=mu_flat,
        edges=edges,
        centers=centers,
        pdf_legacy=pdf_legacy,
        pdf_flat=pdf_flat,
        ratio=ratio,
        mu_edges=mu_edges,
        mu_centers=mu_centers,
        mu_pdf_legacy=mu_pdf_legacy,
        mu_pdf_flat=mu_pdf_flat,
        mu_ratio=mu_ratio,
        summary_json=np.array(json.dumps(summary)),
    )

    ink, muted, base, grid = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
    blue, red = "#2a78d6", "#d94b3d"
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": base,
        "font.size": 10.5,
    })

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.08},
    )

    ax.plot(centers, pdf_legacy, color=blue, lw=2.0,
            label=rf"legacy fixed-$\langle N \rangle$: $\kappa_{{\rm thr}}={legacy_kthr:.2e}$")
    ax.plot(centers, pdf_flat, color=red, lw=2.0,
            label=rf"flat threshold: $\kappa_{{\rm thr}}={args.flat_kthr:.2e}$")
    ax.set_ylabel(r"$dP/d\ln\mu$")
    ax.set_title(rf"$P(\ln\mu)$ comparison at $z_s={args.z:g}$")
    ax.grid(color=grid, lw=0.7, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.3, loc="upper right")
    ax.text(
        0.02, 0.04,
        rf"JSD = {jsd:.3e}" "\n"
        rf"$N_{{\rm samp}} = {args.nsamples:,}$, same seed, subhalo off",
        transform=ax.transAxes, ha="left", va="bottom", color=muted, fontsize=9,
    )

    axr.plot(centers, ratio, color=ink, lw=1.7)
    axr.axhline(1.0, color=muted, lw=1.0, ls="--")
    axr.set_xlabel(r"$\ln\mu$")
    axr.set_ylabel("flat / legacy")
    axr.grid(color=grid, lw=0.7, which="both")
    axr.set_axisbelow(True)

    fig.tight_layout()
    for path in (png_path, pdf_path):
        fig.savefig(path, dpi=220)

    mu_floor = 0.8 * finite_positive_floor(mu_pdf_legacy, mu_pdf_flat)

    fig2, (ax2, ax2r) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.08},
    )
    ax2.semilogy(mu_centers, np.maximum(mu_pdf_legacy, mu_floor), color=blue, lw=2.0,
                 label=rf"legacy fixed-$\langle N \rangle$: $\kappa_{{\rm thr}}={legacy_kthr:.2e}$")
    ax2.semilogy(mu_centers, np.maximum(mu_pdf_flat, mu_floor), color=red, lw=2.0,
                 label=rf"flat threshold: $\kappa_{{\rm thr}}={args.flat_kthr:.2e}$")
    ax2.set_ylabel(r"$dP/d\mu$")
    ax2.set_title(rf"$P(\mu)$ comparison at $z_s={args.z:g}$ (log y, linear x)")
    ax2.grid(color=grid, lw=0.7, which="both")
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9.3, loc="upper right")
    ax2.text(
        0.02, 0.04,
        rf"JSD in $\ln\mu$ bins = {jsd:.3e}" "\n"
        rf"$N_{{\rm samp}} = {args.nsamples:,}$, same seed, subhalo off",
        transform=ax2.transAxes, ha="left", va="bottom", color=muted, fontsize=9,
    )
    ax2r.plot(mu_centers, mu_ratio, color=ink, lw=1.7)
    ax2r.axhline(1.0, color=muted, lw=1.0, ls="--")
    ax2r.set_xlabel(r"$\mu$")
    ax2r.set_ylabel("flat / legacy")
    ax2r.grid(color=grid, lw=0.7, which="both")
    ax2r.set_axisbelow(True)
    fig2.tight_layout()
    for path in (semilogy_png_path, semilogy_pdf_path):
        fig2.savefig(path, dpi=220)

    fig3, (ax3, ax3r) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.3], "hspace": 0.08},
    )
    ax3.loglog(mu_centers, np.maximum(mu_pdf_legacy, mu_floor), color=blue, lw=2.0,
               label=rf"legacy fixed-$\langle N \rangle$: $\kappa_{{\rm thr}}={legacy_kthr:.2e}$")
    ax3.loglog(mu_centers, np.maximum(mu_pdf_flat, mu_floor), color=red, lw=2.0,
               label=rf"flat threshold: $\kappa_{{\rm thr}}={args.flat_kthr:.2e}$")
    ax3.set_ylabel(r"$dP/d\mu$")
    ax3.set_title(rf"$P(\mu)$ comparison at $z_s={args.z:g}$ (log-log)")
    ax3.grid(color=grid, lw=0.7, which="both")
    ax3.set_axisbelow(True)
    ax3.legend(frameon=False, fontsize=9.3, loc="upper right")
    ax3.text(
        0.02, 0.04,
        rf"JSD in $\ln\mu$ bins = {jsd:.3e}" "\n"
        rf"$N_{{\rm samp}} = {args.nsamples:,}$, same seed, subhalo off",
        transform=ax3.transAxes, ha="left", va="bottom", color=muted, fontsize=9,
    )
    ax3r.semilogx(mu_centers, mu_ratio, color=ink, lw=1.7)
    ax3r.axhline(1.0, color=muted, lw=1.0, ls="--")
    ax3r.set_xlabel(r"$\mu$")
    ax3r.set_ylabel("flat / legacy")
    ax3r.grid(color=grid, lw=0.7, which="both")
    ax3r.set_axisbelow(True)
    fig3.tight_layout()
    for path in (loglog_png_path, loglog_pdf_path):
        fig3.savefig(path, dpi=220)

    print(json.dumps(summary, indent=2))
    print(f"wrote {npz_path}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {semilogy_png_path}")
    print(f"wrote {semilogy_pdf_path}")
    print(f"wrote {loglog_png_path}")
    print(f"wrote {loglog_pdf_path}")


if __name__ == "__main__":
    main()
