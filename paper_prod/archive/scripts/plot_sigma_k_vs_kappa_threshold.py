#!/usr/bin/env python3
"""
Production plot: sigma_k vs kappa_threshold (replot from cached data)
Loads `data/variance_sweep_data_z1.npz` and writes high-quality PDF+PNG to
`paper_prod/plots/figures/` and a metadata JSON into `paper_prod/metadata/`.
"""
from pathlib import Path
import json
import datetime
import numpy as np
from paper_prod.plot_style import apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST

# Apply paper style
SIZES = apply_style()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "variance_sweep_data_z1.npz"
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MD = ROOT / "paper_prod" / "metadata"
MD.mkdir(parents=True, exist_ok=True)

def load_data(path: Path):
    npz = np.load(path)
    factors = npz["factors"]
    var_on_paired = npz["var_on_paired"]
    return factors, var_on_paired

def plot(factors, var_on_paired, out_png, out_pdf):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from paper_prod.plot_style import format_log_axis_decimal

    sigma = np.sqrt(np.asarray(var_on_paired, dtype=float))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])
    ax.plot(factors, sigma, marker="o", lw=1.5, ms=4, color="#0b233f")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_{\rm threshold}$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")

    # Limit the number of ticks on y-axis to prevent crowding
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))

    # No top title, no background grid — keep axes clean for manuscript
    ax.grid(False)

    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

def write_metadata(out_png: Path):
    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_rev_short(),
        "source_data": str(DATA.relative_to(ROOT)),
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_png.with_suffix('.pdf').relative_to(ROOT)),
    }
    md_path = MD / (out_png.stem + ".metadata.json")
    md_path.write_text(json.dumps(meta, indent=2))

def _git_rev_short():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"

def main():
    factors, var_on_paired = load_data(DATA)
    out_png = OUT_DIR / "sigma_k_vs_kappa_threshold_measured.png"
    out_pdf = out_png.with_suffix('.pdf')
    plot(factors, var_on_paired, out_png, out_pdf)
    write_metadata(out_png)
    print(f"Wrote {out_png} and {out_pdf} and metadata")

if __name__ == '__main__':
    main()
