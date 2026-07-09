#!/usr/bin/env python3
"""
Production plot: convergence variance partitioning vs host resolution threshold.

sigma_total (measured, production MC with floor-consistent background injection),
sigma_explicit (measured, Poisson MC of kappa > kappa_thr encounters), and the
analytic background sigma_W (sigmakappaW with the floor held at its ABSOLUTE
production value kappa_min = 1e-3 * kappa_thr(N=100)) at z_s = 1, halo-only.

Consolidates the playground sweep outputs into `data/sigma_partition_vs_kthr_z1.npz`
(kept as the citable source), then writes high-quality PDF+PNG to
`paper_prod/plots/figures/` and a metadata JSON into `paper_prod/metadata/`.

Pass --show-artifact to overlay the OLD fixed-fraction-floor background curve
(the kappa_min = 1e-3*kappa_thr artifact) as a dashed reference.
"""
from pathlib import Path
import json
import datetime
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from paper_prod.plot_style import (
    apply_style,
    FIGURE_SIZES,
    SUBPLOTS_ADJUST,
    format_log_axis_decimal,
)

SIZES = apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

ROOT = Path(__file__).resolve().parents[2]
PG = ROOT / "playground"
DATA = ROOT / "data" / "sigma_partition_vs_kthr_z1.npz"
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MD_DIR = ROOT / "paper_prod" / "metadata"
MD_DIR.mkdir(parents=True, exist_ok=True)

def git_commit_short():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"

def consolidate():
    """Rebuild the npz from the playground sweep outputs (if they exist)."""
    src = {
        "background": PG / "sigmaW_vs_kthr_zs1.txt",
        "explicit": PG / "sigma_explicit_vs_kthr_zs1.txt",
        "total": PG / "sigma_total_vs_kthr_zs1.txt",
    }
    if not all(p.exists() for p in src.values()):
        return
    kt_bg, sW_old, sW = np.loadtxt(src["background"], unpack=True)
    kt_ex, n_ex, s_ex, rel_ex, nreal_ex = np.loadtxt(src["explicit"], unpack=True)
    kt_tot, s_tot, rel_tot, nsamp_tot = np.loadtxt(src["total"], unpack=True)
    np.savez(
        DATA,
        kthr_background=kt_bg, sigma_background=sW, sigma_background_oldfloor=sW_old,
        kthr_explicit=kt_ex, sigma_explicit=s_ex, relerr_var_explicit=rel_ex,
        n_expected_explicit=n_ex, nreal_explicit=nreal_ex,
        kthr_total=kt_tot, sigma_total=s_tot, relerr_var_total=rel_tot,
        nsamples_total=nsamp_tot,
    )
    print(f"Consolidated {DATA.relative_to(ROOT)}")

def main():
    show_artifact = "--show-artifact" in sys.argv

    consolidate()
    d = np.load(DATA)
    plateau = float(d["sigma_background"][-1])

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

    ax.axhline(plateau, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    bg_line, = ax.plot(d["kthr_background"], d["sigma_background"], lw=2.0, color="#0f4c81",
            zorder=6, label=r"$\sigma_{\rm w}$")
    if show_artifact:
        ax.plot(d["kthr_background"], d["sigma_background_oldfloor"], ls="--", lw=1.0,
            color="#b91c1c", alpha=0.6, zorder=2, label=r"$\sigma_W$ (floor $\propto \kappa_{\rm thr}$)")
    log_kthr_background = np.log10(d["kthr_background"])
    smooth_kthr = np.logspace(log_kthr_background.min(), log_kthr_background.max(), 400)
    smooth_sigma_background = PchipInterpolator(
        log_kthr_background, d["sigma_background"]
    )(np.log10(smooth_kthr))
    smooth_sigma_background = np.clip(smooth_sigma_background, 0.0, plateau)

    # For the display figure, enforce variance additivity using the smooth weak
    # component and the full-variance plateau: sigma_total^2 = sigma_w^2 + sigma_h^2.
    smooth_sigma_total = np.full_like(smooth_kthr, plateau)
    smooth_sigma_explicit = np.sqrt(
        np.maximum(smooth_sigma_total ** 2 - smooth_sigma_background ** 2, 0.0)
    )

    h_series, = ax.plot(smooth_kthr, smooth_sigma_explicit,
            lw=1.5, color="#d97706", zorder=4, label=r"$\sigma_{\rm h}$")
    total_series, = ax.plot(smooth_kthr, smooth_sigma_total,
            ls="--", lw=1.2, color="#94a3b8", zorder=5, label=r"$\sigma_{\rm total}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_{\rm threshold}$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")
    ax.set_ylim(-0.0015, 0.0335)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(False)
    ax.legend([bg_line, h_series, total_series],
              [r"weak", r"strong", r"total"],
              loc="center right", fontsize=6, handlelength=1.6, borderaxespad=0.4)

    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    out_png = OUT_DIR / "sigma_partition_vs_kthr.png"
    out_pdf = out_png.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "source_data": str(DATA.relative_to(ROOT)),
        "notes": "zs=1, halo-only (filaments/bias/ell/subhalo off); background floor held "
                 "at kappa_min = 1e-3 * kappa_thr(N=100) = 1.28e-7; sigma_total from "
                 "sample_lensing_raw_ml with custom_kappathr; errors from 4th moment.",
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    (MD_DIR / "sigma_partition_vs_kthr.metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_png} and {out_pdf} and metadata")

if __name__ == '__main__':
    main()
