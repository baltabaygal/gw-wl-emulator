"""
Plot: total mean host+substructure convergence (k_total = reduced host + resolved
clumps + kappa_u mean) vs. the absolute clump-mass floor psi_min = m_floor/M.

Chat context (2026-07-22, mass-conserving-carve follow-up): mu_u/M_u/sigma_u^2
individually swing tens of percent with psi_min, but k_total is protected by the
carve's exact mass-conservation identity (M_h + Sum m_i + M_u = M for any psi_min)
combined with kappa's near-linearity in mass -- what survives is a small, nearly
psi_min-independent residual from the host (NFW cusp) vs. subhalo (radially-biased,
B(x)-depleted) profile-shape mismatch. This script renders that curve directly,
using a quadrature floor (m_deep) far below every psi_min tested, to avoid the
grid-clipping artifact found in the first pass (playground/analytic/
mfloor_sensitivity_scan.py used a hardcoded m_deep=1e3 that silently flattened
anything below m_floor~1e3).

Physics functions/geometry kernels: same validated machinery as
wsub_partition_proof.py (checked against the C++ to <~1.6%,
docs/subhalo/subhalo_combining.md).

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/plot_ktotal_vs_psimin.py
Writes plot_ktotal_vs_psimin.{png,json} next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_gate.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX, nfw_params, n_tau, gamma_norm, sigma_crit, fg_kappa,
)
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.analytic.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile,
)


def area_weighted_mean(y_grid, rmax_host, values):
    q_y = 2.0 * y_grid / rmax_host**2
    wt_y = y_grid * np.gradient(np.log(y_grid))
    ay = q_y * wt_y
    return float(np.sum(ay * values) / np.sum(ay))


def run(host_mass, z_lens, z_source, kappa_thr_host, m_floors, m_deep,
        n_mass=160, n_y=160, n_d=240, n_theta=256):
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0_host = rs_host * rhos_host / sigmac

    psi_deep = m_deep / host_mass
    lpsi = np.linspace(np.log(psi_deep), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    mass = psi * host_mass
    dN_dlnpsi = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)

    d_grid = np.logspace(-3, np.log10(rmax_host + r200_host), n_d)
    k1 = np.empty((n_mass, n_d))
    for i, m in enumerate(mass):
        rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
        kappa0 = rs_i * rhos_i / sigmac
        k1[i] = 2.0 * kappa0 * fg_kappa(np.maximum(d_grid / rs_i, 1.0e-12))

    sigma_interp, _ = projected_profile(c_host, r200_host)
    y_grid = np.logspace(-1, np.log10(rmax_host), n_y)
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)
    wt_d = np.gradient(np.log(d_grid))
    fd = f_dy * (d_grid * wt_d)[None, :]
    kappa_ns = 2.0 * kappa0_host * fg_kappa(np.maximum(y_grid / rs_host, 1.0e-12))
    J1 = np.einsum("yd,md->my", fd, k1, optimize=True)

    def mu_of(weight_my):
        return np.trapezoid(dN_dlnpsi[:, None] * J1 * weight_my, lpsi, axis=0)

    def frac_of(weight_my):
        return np.trapezoid((dN_dlnpsi * psi)[:, None] * weight_my, lpsi, axis=0)

    def host_kappa_reduced(frac_res):
        out = np.empty_like(y_grid)
        for fr in np.unique(np.round(frac_res, 6)):
            sel = np.round(frac_res, 6) == fr
            m_eff = max((1.0 - min(fr, 0.95)) * host_mass, 1.0)
            rs_e, rhos_e, _, _ = nfw_params(m_eff, z_lens)
            k0 = rs_e * rhos_e / sigmac
            out[sel] = 2.0 * k0 * fg_kappa(np.maximum(y_grid[sel] / rs_e, 1.0e-12))
        return out

    ones = np.ones((n_mass, n_y))
    k_ns_mean = area_weighted_mean(y_grid, rmax_host, kappa_ns)

    rows = []
    for m_floor in m_floors:
        psi_floor = m_floor / host_mass
        above = psi >= psi_floor
        w = ones * above[:, None]
        mu_b = mu_of(w)
        frac_b = frac_of(w)
        host_full = host_kappa_reduced(frac_b)
        total_y = host_full + mu_b
        rows.append({
            "m_floor": float(m_floor),
            "psi_min": float(psi_floor),
            "f_b": float(np.mean(frac_b)),
            "mu_b_mean": area_weighted_mean(y_grid, rmax_host, mu_b),
            "k_total_mean": area_weighted_mean(y_grid, rmax_host, total_y),
            "closure": area_weighted_mean(y_grid, rmax_host, total_y / kappa_ns),
        })

    return {"k_ns_mean": k_ns_mean, "host_mass": host_mass, "z_lens": z_lens,
            "z_source": z_source, "m_deep": m_deep}, rows


def main() -> None:
    host_mass, z_lens, z_source = 1.0e14, 0.5, 1.0
    kappa_thr_host = 1.276e-4
    m_deep = 1.0e-3   # far below every m_floor tested; avoids the earlier grid-clip artifact
    production_m_floor = 1.0e7

    m_floors = np.logspace(-3, 9, 37)

    meta, rows = run(host_mass, z_lens, z_source, kappa_thr_host, m_floors, m_deep)

    psi_min = np.array([r["psi_min"] for r in rows])
    k_total = np.array([r["k_total_mean"] for r in rows])
    closure = np.array([r["closure"] for r in rows])
    k_ns = meta["k_ns_mean"]
    psi_prod = production_m_floor / host_mass

    out_stem = ROOT / "playground" / "analytic" / "plot_ktotal_vs_psimin"
    out_stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    ax = axes[0]
    ax.plot(psi_min, k_total, "o-", color="#2563eb", ms=4, label=r"$k_{\rm total}$ (host+clumps+$\kappa_u$ mean)")
    ax.axhline(k_ns, color="0.4", ls="--", lw=1.2, label=r"unperturbed host $\kappa_{\rm ns}$ (no split)")
    ax.axvline(psi_prod, color="#dc2626", ls=":", lw=1.3, label=r"production $\psi_{\rm min}$ ($m_{\rm floor}=10^7\,M_\odot$)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\psi_{\rm min} = m_{\rm floor}/M$")
    ax.set_ylabel(r"area-weighted mean $\kappa$")
    ax.set_title(r"total mean $\kappa$ vs. absolute clump-mass floor")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(psi_min, closure, "o-", color="#059669", ms=4, label=r"$k_{\rm total}/\kappa_{\rm ns}$")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.2)
    ax.axvline(psi_prod, color="#dc2626", ls=":", lw=1.3, label="production default")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\psi_{\rm min} = m_{\rm floor}/M$")
    ax.set_ylabel(r"closure ratio $k_{\rm total}/\kappa_{\rm ns}$")
    ax.set_title("mass-conservation closure (residual = host/subhalo profile-shape mismatch)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    fig.suptitle(f"M={host_mass:.0e}, $z_l$={z_lens}, $z_s$={z_source}, "
                 f"subhalo model 3 carve (scheme A)", y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=180, facecolor="white", bbox_inches="tight")
    print(f"saved {out_stem}.png / .json")
    print(f"k_ns (unperturbed host) = {k_ns:.5e}")
    print(f"closure at production psi_min ({psi_prod:.2e}): "
          f"{np.interp(np.log(psi_prod), np.log(psi_min), closure):.5f}")


if __name__ == "__main__":
    main()
