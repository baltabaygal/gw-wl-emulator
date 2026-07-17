#!/usr/bin/env python3
"""Legacy-vs-flat kappa-threshold PDF comparison across many source redshifts.

Isolates the host-threshold effect (subhalos off):
  - legacy fixed-<N> rule: kappathr_flat = -1
  - flat rule:             kappathr_flat = FLAT_KTHR (default 1e-4)

Produces three 6-panel (2x3) figures over z_s = 0.2, 0.5, 1.0, 2.0, 5.0, 10.0:
  - linlin  : dP/dln(mu) vs ln(mu)
  - loglin  : dP/dmu vs mu   (log y, linear x)
  - loglog  : dP/dmu vs mu   (log y, log x)

Outputs:
  plots/kappathr_pdf_grid_1em04_linlin.{png,pdf}
  plots/kappathr_pdf_grid_1em04_loglin.{png,pdf}
  plots/kappathr_pdf_grid_1em04_loglog.{png,pdf}
  data/results/kappathr_pdf_grid_1em04.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zs", type=float, nargs="+",
                   default=[0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    p.add_argument("--nsamples", type=int, default=200000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--nbins", type=int, default=120)
    p.add_argument("--h", type=float, default=0.674)
    p.add_argument("--omega-m", type=float, default=0.315, dest="omega_m")
    p.add_argument("--sigma8", type=float, default=0.811)
    p.add_argument("--flat-kthr", type=float, default=1.0e-4, dest="flat_kthr")
    return p.parse_args()


def density_hist(samples: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(samples, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(widths, dtype=float)
    return counts / (total * widths)


def finite_positive_floor(*arrays: np.ndarray) -> float:
    pos = [a[np.isfinite(a) & (a > 0.0)] for a in arrays]
    pos = [a for a in pos if a.size > 0]
    if not pos:
        return 1.0e-12
    return float(np.min(np.concatenate(pos)))


def js_divergence(p: np.ndarray, q: np.ndarray, widths: np.ndarray) -> float:
    eps = 1.0e-300
    pn = np.clip(p, eps, None); pn = pn / np.sum(pn * widths)
    qn = np.clip(q, eps, None); qn = qn / np.sum(qn * widths)
    m = 0.5 * (pn + qn)
    return 0.5 * (np.sum(pn * np.log(pn / m) * widths)
                  + np.sum(qn * np.log(qn / m) * widths))


def compute_one(gw, z, args) -> dict:
    run_common = dict(
        z=z, h=args.h, OmegaM=args.omega_m, sigma8=args.sigma8,
        nsamples=args.nsamples, seed=args.seed,
        strict_weak_lensing=False, subhalo=False,
    )
    legacy = gw.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=-1.0)
    flat = gw.sample_lnmu_ml_with_diagnostics(**run_common, kappathr_flat=args.flat_kthr)

    lnmu_l = np.asarray(legacy["lnmu"])
    lnmu_f = np.asarray(flat["lnmu"])

    lo = min(np.quantile(lnmu_l, 1e-4), np.quantile(lnmu_f, 1e-4))
    hi = max(np.quantile(lnmu_l, 1 - 1e-4), np.quantile(lnmu_f, 1 - 1e-4))
    pad = 0.08 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, args.nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    pdf_l = density_hist(lnmu_l, edges)
    pdf_f = density_hist(lnmu_f, edges)
    jsd = js_divergence(pdf_l, pdf_f, widths)

    mu_l = np.exp(lnmu_l); mu_f = np.exp(lnmu_f)
    mu_lo = min(np.quantile(mu_l, 1e-4), np.quantile(mu_f, 1e-4))
    mu_hi = max(np.quantile(mu_l, 1 - 1e-4), np.quantile(mu_f, 1 - 1e-4))
    mu_edges = np.geomspace(max(mu_lo * 0.9, 1.0e-6), mu_hi * 1.1, args.nbins + 1)
    mu_centers = np.sqrt(mu_edges[:-1] * mu_edges[1:])
    mu_pdf_l = density_hist(mu_l, mu_edges)
    mu_pdf_f = density_hist(mu_f, mu_edges)

    legacy_kthr = gw.get_kappa_threshold(z, args.h, args.omega_m, args.sigma8, 100)

    return dict(
        z=z, jsd=float(jsd), legacy_kthr=float(legacy_kthr),
        centers=centers, pdf_l=pdf_l, pdf_f=pdf_f,
        mu_centers=mu_centers, mu_pdf_l=mu_pdf_l, mu_pdf_f=mu_pdf_f,
    )


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "build"))
    import gwlensing as gw  # pylint: disable=import-error

    data = []
    for z in args.zs:
        print(f"[compute] z_s = {z} ...", flush=True)
        data.append(compute_one(gw, z, args))

    # ---- style ----
    ink, muted, base, grid = "#0b0b0b", "#6d6a61", "#c9c5b8", "#e6e2d7"
    blue, red = "#2a78d6", "#d94b3d"
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": base, "font.size": 10.5,
    })

    nz = len(data)
    ncol = 3
    nrow = int(np.ceil(nz / ncol))

    leg_label = rf"legacy fixed-$\langle N \rangle$"
    flat_label = rf"flat $\kappa_{{\rm thr}}={args.flat_kthr:.0e}$"

    def make_fig(kind: str, title: str):
        fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 8.2))
        axes = np.atleast_1d(axes).ravel()
        for ax, d in zip(axes, data):
            if kind == "linlin":
                ax.plot(d["centers"], d["pdf_l"], color=blue, lw=1.9)
                ax.plot(d["centers"], d["pdf_f"], color=red, lw=1.9)
                ax.set_xlabel(r"$\ln\mu$")
                ax.set_ylabel(r"$dP/d\ln\mu$")
            else:
                floor = 0.8 * finite_positive_floor(d["mu_pdf_l"], d["mu_pdf_f"])
                yl = np.maximum(d["mu_pdf_l"], floor)
                yf = np.maximum(d["mu_pdf_f"], floor)
                ax.plot(d["mu_centers"], yl, color=blue, lw=1.9)
                ax.plot(d["mu_centers"], yf, color=red, lw=1.9)
                ax.set_yscale("log")
                if kind == "loglog":
                    ax.set_xscale("log")
                ax.set_xlabel(r"$\mu$")
                ax.set_ylabel(r"$dP/d\mu$")
            ax.set_title(rf"$z_s={d['z']:g}$   (legacy $\kappa_{{\rm thr}}={d['legacy_kthr']:.2e}$)",
                         fontsize=10)
            ax.grid(color=grid, lw=0.7, which="both")
            ax.set_axisbelow(True)
            ax.text(0.03, 0.95, rf"JSD = {d['jsd']:.2e}", transform=ax.transAxes,
                    ha="left", va="top", color=muted, fontsize=8.6)
        for ax in axes[nz:]:
            ax.axis("off")

        # single shared legend
        handles = [plt.Line2D([], [], color=blue, lw=2.2, label=leg_label),
                   plt.Line2D([], [], color=red, lw=2.2, label=flat_label)]
        fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
                   fontsize=11, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, y=1.035, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        return fig

    plots_dir = repo / "plots"; plots_dir.mkdir(exist_ok=True)
    results_dir = repo / "data" / "results"; results_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("linlin", rf"$P(\ln\mu)$: legacy vs flat $\kappa_{{\rm thr}}={args.flat_kthr:.0e}$ (subhalo off, $N={args.nsamples:,}$)"),
        ("loglin", rf"$P(\mu)$ log-lin: legacy vs flat $\kappa_{{\rm thr}}={args.flat_kthr:.0e}$ (subhalo off, $N={args.nsamples:,}$)"),
        ("loglog", rf"$P(\mu)$ log-log: legacy vs flat $\kappa_{{\rm thr}}={args.flat_kthr:.0e}$ (subhalo off, $N={args.nsamples:,}$)"),
    ]
    for kind, title in specs:
        fig = make_fig(kind, title)
        png = plots_dir / f"kappathr_pdf_grid_1em04_{kind}.png"
        pdf = plots_dir / f"kappathr_pdf_grid_1em04_{kind}.pdf"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")
        print(f"wrote {pdf}")

    # save arrays
    save = {"zs": np.array(args.zs), "flat_kthr": args.flat_kthr,
            "nsamples": args.nsamples, "seed": args.seed}
    for d in data:
        t = f"z{str(d['z']).replace('.', 'p')}"
        for k in ("centers", "pdf_l", "pdf_f", "mu_centers", "mu_pdf_l", "mu_pdf_f"):
            save[f"{t}_{k}"] = d[k]
        save[f"{t}_jsd"] = d["jsd"]
        save[f"{t}_legacy_kthr"] = d["legacy_kthr"]
    npz = results_dir / "kappathr_pdf_grid_1em04.npz"
    np.savez(npz, **save)
    print(f"wrote {npz}")
    print(json.dumps({str(d["z"]): {"jsd": d["jsd"], "legacy_kthr": d["legacy_kthr"]}
                      for d in data}, indent=2))


if __name__ == "__main__":
    main()
