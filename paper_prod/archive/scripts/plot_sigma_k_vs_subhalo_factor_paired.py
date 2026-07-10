#!/usr/bin/env python3
"""
Production plot: sigma_k vs subhalo_factor (paired excess)
Loads `data/variance_sweep_data_z1.npz` and `data/variance_sweep_data_z2.npz` and writes high-quality PDF+PNG to
`paper_prod/plots/figures/` and a metadata JSON into `paper_prod/metadata/`.
"""
from pathlib import Path
import json
import datetime
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from paper_prod.plot_style import (
    apply_style,
    FIGURE_SIZES,
    SUBPLOTS_ADJUST,
    format_log_axis_decimal,
)

# Apply paper style
SIZES = apply_style()

ROOT = Path(__file__).resolve().parents[2]
DATA_Z1 = ROOT / "data" / "variance_sweep_data_z1.npz"
DATA_Z2 = ROOT / "data" / "variance_sweep_data_z2.npz"
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

def main():
    # Load z=1 data
    z1_npz = np.load(DATA_Z1)
    factors_z1 = z1_npz["factors"]
    var_z1 = z1_npz["var_on_paired"]
    sigma_z1 = np.sqrt(np.asarray(var_z1, dtype=float))

    # Load z=2 data
    z2_npz = np.load(DATA_Z2)
    factors_z2 = z2_npz["factors"]
    var_z2 = z2_npz["var_on_paired"]
    sigma_z2 = np.sqrt(np.asarray(var_z2, dtype=float))

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

    # Plot both curves
    ax.plot(factors_z1, sigma_z1, marker="o", lw=1.5, ms=4, color="#1a365d", label=r"$z_s = 1.0$")
    ax.plot(factors_z2, sigma_z2, marker="s", lw=1.5, ms=4, color="#d97706", label=r"$z_s = 2.0$")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon_{\rm sub}\ \ (\kappa_{\rm thr,clump}/\kappa_{\rm thr,host})$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")

    # Limit the number of ticks on y-axis to prevent crowding
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    ax.grid(False)
    ax.legend(loc="lower left")

    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    out_png = OUT_DIR / "sigma_k_vs_subhalo_factor_paired.png"
    out_pdf = out_png.with_suffix('.pdf')

    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    # Write metadata
    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "source_data": [str(DATA_Z1.relative_to(ROOT)), str(DATA_Z2.relative_to(ROOT))],
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    meta_path = MD_DIR / "sigma_k_vs_subhalo_factor_paired.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_png} and {out_pdf} and metadata")

if __name__ == '__main__':
    main()
