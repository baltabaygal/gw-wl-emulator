#!/usr/bin/env python3
"""
Production plot: per-host SUBSTRUCTURE variance partitioning vs subhalo_factor.

The subhalo analog of plot_sigma_partition_vs_kthr.py: the resolved-clump
("strong") and unresolved analytic Wsub ("weak") variance components of one
host's substructure convergence, as the resolution gate kappa_thr_clump =
f_subhalo * kappa_thr moves. By the Poisson restriction theorem the two add in
quadrature to the factor-independent total (subhalo_model 3, exact partition;
docs/subhalo/wsub_gaussian_term_derivation.md).

Curves are the exact Campbell integrals over the encounter ensemble of the
fiducial host (M = 1e13 Msun, z_l = 0.5, z_s = 1; area-weighted impact
parameter y within the host encounter disc, production flat kappa_thr = 1e-3):

  sigma_strong^2(f) = < C2_resolved(f, y) >_y     (discrete clump shot noise)
  sigma_weak^2(f)   = < sigma_unres^2(f, y) >_y   (the model-3 Gaussian term)
  sigma_total       = sqrt(strong^2 + weak^2)     (flat plateau, any f)

The quadrature machinery is the one validated against the stratified clump MC
(playground/analytic/wsub_partition_proof.py, wsub_mc_partition.py) and against
the C++ production run (model 3 flat at brute over factor 1e-5..1).

Writes data/sigma_partition_vs_subhalo_factor_z1.npz (citable source),
paper_prod/plots/figures/sigma_partition_vs_subhalo_factor.{png,pdf} and a
metadata JSON.
"""
from pathlib import Path
import datetime
import json
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper_prod.plot_style import (  # noqa: E402
    apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST, format_log_axis_decimal,
)
from scripts.subhalo_gate.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX,
    fg_kappa, gamma_norm, n_tau, nfw_params, sigma_crit,
)
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.analytic.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile,
)

SIZES = apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

DATA = ROOT / "data" / "sigma_partition_vs_subhalo_factor_z1.npz"
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
MD_DIR = ROOT / "paper_prod" / "metadata"

HOST_MASS, Z_LENS, Z_SOURCE = 1.0e13, 0.5, 1.0
KAPPA_THR_HOST = 1.0e-3          # production flat threshold (2026-07-09)
M_FLOOR = 1.0e7


def git_commit_short():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def compute(factors, n_mass=100, n_y=200, n_d=240, n_theta=256):
    """Exact Campbell partition curves for the fiducial host."""
    rs_h, rhos_h, c_host, r200_h = nfw_params(HOST_MASS, Z_LENS)
    rmax = host_rmax(HOST_MASS, Z_LENS, Z_SOURCE, KAPPA_THR_HOST)
    nt, _ = n_tau(HOST_MASS, Z_LENS)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(Z_SOURCE, Z_LENS)

    lpsi = np.linspace(np.log(M_FLOOR / HOST_MASS), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    mass = psi * HOST_MASS
    dN = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)       # dN/dlnpsi

    d_grid = np.logspace(-3, np.log10(rmax + r200_h), n_d)
    rs_c, rhos_c, _, _ = nfw_params(mass, Z_LENS)
    kappa0_c = rs_c * rhos_c / sigmac
    k1 = 2.0 * kappa0_c[:, None] * fg_kappa(np.maximum(d_grid[None, :] / rs_c[:, None], 1e-12))

    sigma_interp, _ = projected_profile(c_host, r200_h)
    y_grid = np.logspace(np.log10(0.05), np.log10(rmax), n_y)
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)
    fd = f_dy * (d_grid * np.gradient(np.log(d_grid)))[None, :]
    J2 = np.einsum("yd,md->my", fd, k1**2, optimize=True)

    a_y = 2.0 * y_grid**2 * np.gradient(np.log(y_grid)) / rmax**2
    a_y /= a_y.sum()

    # gate variable: clump kappa at HOST-CENTER distance y (== reach(m) >= y)
    k_at_y = 2.0 * kappa0_c[:, None] * fg_kappa(np.maximum(y_grid[None, :] / rs_c[:, None], 1e-12))

    C2_b = np.sum(a_y * np.trapezoid(dN[:, None] * J2, lpsi, axis=0))
    s_weak = np.empty(len(factors))
    for i, f in enumerate(factors):
        unres = k_at_y < f * KAPPA_THR_HOST
        s2U = np.trapezoid(dN[:, None] * J2 * unres, lpsi, axis=0)
        s_weak[i] = np.sqrt(np.sum(a_y * s2U))
    s_total = np.sqrt(C2_b)
    s_strong = np.sqrt(np.maximum(C2_b - s_weak**2, 0.0))
    return s_weak, s_strong, s_total


def main():
    factors = np.logspace(-5, 3, 65)
    s_weak, s_strong, s_total = compute(factors)
    np.savez(DATA, subhalo_factor=factors, sigma_weak=s_weak,
             sigma_strong=s_strong, sigma_total=float(s_total),
             host_mass=HOST_MASS, z_lens=Z_LENS, z_source=Z_SOURCE,
             kappa_thr_host=KAPPA_THR_HOST, m_floor=M_FLOOR)
    print(f"Consolidated {DATA.relative_to(ROOT)}  (plateau sigma = {s_total:.4g})")

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])

    ax.axhline(s_total, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    weak_line, = ax.plot(factors, s_weak, lw=2.0, color="#0f4c81", zorder=6)
    strong_line, = ax.plot(factors, s_strong, lw=1.5, color="#d97706", zorder=4)
    total_line, = ax.plot(factors, np.full_like(factors, s_total),
                          ls="--", lw=1.2, color="#94a3b8", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$f_{\rm subhalo} = \kappa^{\rm clump}_{\rm thr} / \kappa_{\rm thr}$")
    ax.set_ylabel(r"$\sigma_{\kappa,\,{\rm sub}}$")
    ax.set_ylim(-0.05 * s_total, 1.22 * s_total)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.grid(False)
    ax.legend([weak_line, strong_line, total_line],
              [r"weak", r"strong", r"total"],
              loc="center right", fontsize=6, handlelength=1.6, borderaxespad=0.4)
    try:
        format_log_axis_decimal(ax, axis='x')
    except Exception:
        pass

    out_png = OUT_DIR / "sigma_partition_vs_subhalo_factor.png"
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "source_data": str(DATA.relative_to(ROOT)),
        "notes": "Per-host substructure variance partition for the fiducial host "
                 "M=1e13, z_l=0.5, z_s=1, kappa_thr=1e-3 (flat production default), "
                 "m_floor=1e7. Exact Campbell integrals over the encounter ensemble "
                 "(area-weighted y in [0.05 kpc, rmax]); strong = resolved clump shot "
                 "noise, weak = model-3 unresolved Gaussian sigma; quadrature-exact "
                 "partition (Poisson restriction theorem), machinery validated in "
                 "playground/analytic/wsub_partition_proof.py + wsub_mc_partition.py "
                 "and against C++ subhalo_model=3.",
        "output_png": str(out_png.relative_to(ROOT)),
        "output_pdf": str(out_pdf.relative_to(ROOT)),
    }
    (MD_DIR / "sigma_partition_vs_subhalo_factor.metadata.json").write_text(
        json.dumps(meta, indent=2))
    print(f"Wrote {out_png} and {out_pdf} and metadata")


if __name__ == "__main__":
    main()
