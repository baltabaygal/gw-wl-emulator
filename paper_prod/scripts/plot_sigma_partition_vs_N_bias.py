#!/usr/bin/env python3
"""
Production plot: the bias-era variance partition drawn against the expected
explicit-halo count <N> instead of kappa_thr — <N> = NhfNFW(zs, kappa_thr) is
monotone decreasing in kappa_thr (<N> = 100 at the production threshold by
construction), so this is a change of x-variable of
plot_sigma_partition_vs_kthr_bias.py: weak falls / strong rises with <N>,
sigma_total stays flat.

Reads the consolidated npz written by plot_sigma_partition_vs_kthr_bias.py
(which must be run first — it also stores the <N>(kappa_thr) mapping from
playground/nexp_vs_kthr_zs1.txt <- playground/compute_nexp_vs_kthr.py).
Points with <N> < XMIN are dropped (kappa_thr >~ 0.5, where <N> collapses
super-exponentially — the weak arm is already at its plateau there).

Writes paper_prod/plots/figures/sigma_partition_vs_N_bias.{png,pdf} + metadata.
"""
from pathlib import Path
import json
import datetime
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_prod.plot_style import (  # noqa: E402
    apply_style,
    FIGURE_SIZES,
    SUBPLOTS_ADJUST,
    format_log_axis_decimal,
)

SIZES = apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

DATA = ROOT / "data" / "sigma_partition_vs_kthr_bias_z1.npz"
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
MD_DIR = ROOT / "paper_prod" / "metadata"

ESTIMATOR = "clip1"   # shared-mask core estimator (kappa_tot <= 1)
XMIN = 1.0e-4         # smallest <N> shown (weak arm at plateau below this)
KMIN = 1.278e-7       # absolute background floor 1e-3 * kappa_thr(N=100, zs=1)


def git_commit_short():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def main():
    d = np.load(DATA)
    if "nexp" not in d.files:
        raise SystemExit("npz lacks the <N> mapping — run plot_sigma_partition_vs_kthr_bias.py "
                         "(after playground/compute_nexp_vs_kthr.py) first")
    kthr = d["kthr"]
    nexp = d["nexp"]
    sig_w = np.sqrt(d[f"var_w_{ESTIMATOR}"])
    sig_s = np.sqrt(d[f"var_s_{ESTIMATOR}"])
    sig_t = np.sqrt(d[f"var_t_{ESTIMATOR}"])

    shot_map = {round(4.0 * np.log10(k)): s
                for k, s in zip(d["kthr_analytic"], d["sigW_shot_analytic"])}
    sig_w_shot = np.array([shot_map[round(4.0 * np.log10(k))] for k in kthr])
    sig_w_corr = np.sqrt(np.maximum(sig_w**2 - sig_w_shot**2, 0.0))

    core = kthr >= KMIN
    plateau = float(sig_t[core].mean())
    flat = (sig_t[core].max() - sig_t[core].min()) / plateau

    # x-axis: <N>, increasing = decreasing kappa_thr; drop the collapsed tail
    # AND the sub-floor points (kappa_thr < kappa_min = the model's fixed halo
    # budget — the weak arm is empty there and the total genuinely grows; see
    # plot_sigma_partition_vs_kthr_bias.py). <N>(kappa_min) ~ 1.1e5.
    m = np.isfinite(nexp) & (nexp >= XMIN) & core
    order = np.argsort(nexp[m])
    N = nexp[m][order]
    W, WSH, WCO, S, T = (a[m][order] for a in (sig_w, sig_w_shot, sig_w_corr, sig_s, sig_t))
    n_kmin = float(np.exp(np.interp(np.log(KMIN), np.log(kthr[np.argsort(kthr)]),
                                    np.log(nexp[np.argsort(kthr)]))))

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

    ax.axhline(plateau, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    ax.axvline(100.0, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    ax.text(100.0 * 1.5, 1.16 * plateau, r"$\langle N\rangle = 100$", fontsize=6,
            color="#64748b", ha="left", va="top")

    w_line, = ax.plot(N, W, lw=2.0, color="#0f4c81", zorder=6)
    wsh_line, = ax.plot(N, WSH, lw=1.0, ls=(0, (1, 1.2)), color="#0f4c81",
                        alpha=0.85, zorder=3)
    wco_line, = ax.plot(N, WCO, lw=1.0, ls=(0, (4, 1.5, 1, 1.5)),
                        color="#5b8fbe", zorder=3)
    s_line, = ax.plot(N, S, lw=1.5, color="#d97706", zorder=4)
    t_line, = ax.plot(N, T, ls="--", lw=1.2, color="#94a3b8", zorder=7)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\langle N \rangle$")
    ax.set_ylabel(r"$\sigma_{\kappa}$")
    ax.set_ylim(-0.05 * plateau, 1.23 * plateau)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(False)
    ax.legend([w_line, wsh_line, wco_line, s_line, t_line],
              [r"weak", r"weak (shot)", r"weak (corr.)", r"strong", r"total"],
              loc="center right", fontsize=6, handlelength=2.0, borderaxespad=0.8,
              labelspacing=0.35)

    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    out_png = OUT_DIR / "sigma_partition_vs_N_bias.png"
    out_pdf = out_png.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "source_data": str(DATA.relative_to(ROOT)),
        "estimator": ESTIMATOR,
        "plateau_sigma": plateau,
        "flatness_relrange_sigma_total_core": float(flat),
        "xmin_nexp": XMIN,
        "nexp_at_kappa_min": n_kmin,
        "notes": "Same data and estimator as sigma_partition_vs_kthr_bias (see its "
                 "metadata); x-axis changed to <N> = NhfNFW(zs=1, kappa_thr) from "
                 "playground/compute_nexp_vs_kthr.py (<N>=100 at the production "
                 "threshold; <N> ~ 0.013/kappa_thr in the low-threshold regime). "
                 "Points with <N> < %g dropped (kappa_thr >~ 0.5: <N> collapses "
                 "super-exponentially, weak arm already at plateau), and the plot ends "
                 "at the model domain boundary <N>(kappa_min) ~ %.2g — beyond it "
                 "(kappa_thr < kappa_min) the weak arm is empty and the explicit arm "
                 "re-includes the sub-floor band no arm ever covered, so the total "
                 "genuinely grows (sub-floor points kept in the npz)." % (XMIN, n_kmin),
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    (MD_DIR / "sigma_partition_vs_N_bias.metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"plateau sigma_total = {plateau:.6f}   core flatness = {flat:.4f}   "
          f"<N>(kappa_min) = {n_kmin:.3g}")
    print(f"Wrote {out_png} and {out_pdf} and metadata")


if __name__ == '__main__':
    main()
