"""
kappa_total vs psi_min for the "resolved only, no kappa_u" scheme (chat 2026-07-23):
here psi_min IS the resolved floor itself (no separate coarse/deep split) -- everything
with psi >= psi_min is drawn as an explicit clump and contributes to Sum_i kappa_i; the
host is reduced by exactly that resolved bound fraction; nothing analytic fills in
psi < psi_min (no kappa_u, no M_u). Swept over psi_min = 1e-5 .. 1e2 (>1 means nothing
is resolved at all, since psi_max = 1).

kappa_host(psi_min)     = host reduced by f_res(psi_min) = F(psi_min, psi_max)
kappa_resolved(psi_min) = mean Sum_i kappa_NFW(m_i), all clumps psi in [psi_min, psi_max]
kappa_total             = kappa_host + kappa_resolved   (the two ingredients, summed)
kappa_ns                = unperturbed host at full M (no split at all) -- reference only

Same validated Campbell-moment machinery as wsub_partition_proof.py / plot_ktotal_vs_psimin.py
(checked against the C++ to <~1.6%, docs/subhalo/subhalo_combining.md).

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/plot_ktotal_resolved_only.py
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


def run(host_mass, z_lens, z_source, kappa_thr_host, psi_mins,
        n_mass=200, n_y=160, n_d=240, n_theta=256):
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0_host = rs_host * rhos_host / sigmac

    # deep enough for the smallest psi_min tested; ALPHA=-0.82 mass integral converges anyway
    psi_deep = min(psi_mins) * 1e-2
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
    for psi_min in psi_mins:
        if psi_min >= PSI_MAX:
            # nothing resolved: host stays at full mass, no clumps at all
            k_host_y = kappa_ns.copy()
            mu_y = np.zeros_like(y_grid)
        else:
            above = psi >= psi_min
            w = ones * above[:, None]
            mu_y = mu_of(w)
            frac_y = frac_of(w)
            k_host_y = host_kappa_reduced(frac_y)
        total_y = k_host_y + mu_y
        rows.append({
            "psi_min": float(psi_min),
            "kappa_host": area_weighted_mean(y_grid, rmax_host, k_host_y),
            "kappa_resolved": area_weighted_mean(y_grid, rmax_host, mu_y),
            "kappa_total": area_weighted_mean(y_grid, rmax_host, total_y),
        })

    return {"k_ns_mean": k_ns_mean, "host_mass": host_mass, "z_lens": z_lens, "z_source": z_source}, rows


def main() -> None:
    host_mass, z_lens, z_source = 1.0e14, 0.5, 1.0
    kappa_thr_host = 1.276e-4
    psi_mins = np.logspace(-8, 2, 55)

    meta, rows = run(host_mass, z_lens, z_source, kappa_thr_host, psi_mins)

    psi_min_arr = np.array([r["psi_min"] for r in rows])
    k_host = np.array([r["kappa_host"] for r in rows])
    k_res = np.array([r["kappa_resolved"] for r in rows])
    k_total = np.array([r["kappa_total"] for r in rows])
    k_ns = meta["k_ns_mean"]

    out_stem = ROOT / "playground" / "analytic" / "plot_ktotal_resolved_only"
    out_stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.0))
    ax.plot(psi_min_arr, k_total, "o-", color="#2563eb", ms=4, lw=2.2,
            label=r"$\kappa_{\rm total}$ = host + resolved (no $\kappa_u$)")
    ax.plot(psi_min_arr, k_host, "s--", color="#d97706", ms=4, lw=1.6,
            label=r"$\kappa_{\rm host}$ (reduced by resolved fraction)")
    ax.plot(psi_min_arr, k_res, "^--", color="#7c3aed", ms=4, lw=1.6,
            label=r"$\Sigma_i\,\kappa_i$ (resolved clumps, mean)")
    ax.axhline(k_ns, color="0.4", ls=":", lw=1.3, label=r"unperturbed host $\kappa_{\rm ns}$ (no split at all)")
    ax.axvline(1.0, color="0.6", ls="-", lw=0.8)
    ax.text(1.05, ax.get_ylim()[1]*0.02 + k_ns, r"$\psi_{\max}=1$", fontsize=8, color="0.4")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\psi_{\rm min}$  (resolved floor; everything below stays in the host, no correction)")
    ax.set_ylabel(r"area-weighted mean $\kappa$")
    ax.set_title(f"resolved-only scheme (no $\\kappa_u$): M={host_mass:.0e}, "
                 f"$z_l$={z_lens}, $z_s$={z_source}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=180, facecolor="white", bbox_inches="tight")
    print(f"saved {out_stem}.png / .json")
    for r in rows[::4]:
        print(f"psi_min={r['psi_min']:.2e}  host={r['kappa_host']:.4e}  "
              f"resolved={r['kappa_resolved']:.4e}  total={r['kappa_total']:.4e}")
    print(f"k_ns (unperturbed) = {k_ns:.4e}")


if __name__ == "__main__":
    main()
