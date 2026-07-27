"""
sigma_kappa vs psi_min for the "resolved only, no kappa_u" scheme -- companion to
plot_ktotal_resolved_only.py (which shows the MEAN <kappa> vs psi_min for the same
scheme). Here psi_min IS the resolved floor itself: everything with psi >= psi_min is
an explicit clump contributing to Sum_i kappa_i, the host is reduced by exactly the
resolved bound fraction, and nothing analytic fills in psi < psi_min (no kappa_u, no
M_u) -- i.e. this is subhalo_model=4's scheme (chat 2026-07-23).

Extends the exact-quadrature Campbell-moment machinery to the SECOND moment: for a
Poisson clump field with intensity dN/dlnpsi, Var(Sum_i g(x_i)) = integral of
intensity(x) * g(x)^2 dx (Campbell's theorem) -- no MC needed, same distance_kernel/
projected_profile projection used for the mean, just with the per-clump kernel k1
squared before the same LOS projection. This is the same machinery/host as
plot_ktotal_resolved_only.py, extended with:
  sigma_sub(psi_min)  = sqrt(Var(Sum_i kappa_i))              (Campbell 2nd moment)
  sigma_host(psi_min) = |dkappa_NFW/dM| * sqrt(Var(Sum_i m_i)) (carve response, matches
                         the first-order treatment in
                         paper_prod/scripts/plot_fig_subhalo_sigma_decomposition.py)
  sigma_tot(psi_min)  = sqrt(Var_host + Var_sub + 2*Cov(host,sub))

All three area-weighted over the ray impact parameter y (same aperture average used
for the mean), so each psi_min gives one number per curve.

Top axis: psi_min translated to an equivalent kappa scale -- the peak (r=0) NFW
convergence of a single clump sitting exactly at the resolved floor, 2*kappa0(psi_min*M).
This answers "what does psi_min mean in kappa units" directly (interpolated off the same
psi_min grid, so no separate root-find).

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/plot_sigma_vs_psimin.py
Writes plot_sigma_vs_psimin.{png,json} next to this file.
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
    kappa_nfw,
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
    k2 = k1**2

    sigma_interp, _ = projected_profile(c_host, r200_host)
    y_grid = np.logspace(-1, np.log10(rmax_host), n_y)
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)
    wt_d = np.gradient(np.log(d_grid))
    fd = f_dy * (d_grid * wt_d)[None, :]
    kappa_ns = 2.0 * kappa0_host * fg_kappa(np.maximum(y_grid / rs_host, 1.0e-12))
    J1 = np.einsum("yd,md->my", fd, k1, optimize=True)   # <kappa_c | y> per mass bin
    J2 = np.einsum("yd,md->my", fd, k2, optimize=True)   # <kappa_c^2 | y> per mass bin

    def mu_of(weight_my):
        return np.trapezoid(dN_dlnpsi[:, None] * J1 * weight_my, lpsi, axis=0)

    def var_of(weight_my):
        # Campbell 2nd moment: Var(Sum kappa_i | y) = int dN/dlnpsi * <kappa_c^2|y> dlnpsi
        return np.trapezoid(dN_dlnpsi[:, None] * J2 * weight_my, lpsi, axis=0)

    def covmk_of(weight_my):
        # Cov(Sum m_i, Sum kappa_i | y) = int dN/dlnpsi * m * <kappa_c|y> dlnpsi
        return np.trapezoid((dN_dlnpsi * mass)[:, None] * J1 * weight_my, lpsi, axis=0)

    def frac_of(weight_my):
        return np.trapezoid((dN_dlnpsi * psi)[:, None] * weight_my, lpsi, axis=0)

    def varm_of(above):
        # Var(Sum m_i) -- position-independent, scalar
        return float(np.trapezoid(dN_dlnpsi * mass**2 * above, lpsi))

    def host_kappa_reduced(frac_res):
        out = np.empty_like(y_grid)
        for fr in np.unique(np.round(frac_res, 6)):
            sel = np.round(frac_res, 6) == fr
            m_eff = max((1.0 - min(fr, 0.95)) * host_mass, 1.0)
            rs_e, rhos_e, _, _ = nfw_params(m_eff, z_lens)
            k0 = rs_e * rhos_e / sigmac
            out[sel] = 2.0 * k0 * fg_kappa(np.maximum(y_grid[sel] / rs_e, 1.0e-12))
        return out

    def host_dkdM(frac_res):
        # d(kappa_NFW)/dM at the reduced host mass, evaluated over the full y_grid
        m_eff = max((1.0 - min(frac_res, 0.95)) * host_mass, 1.0)
        dM = 1e-3 * m_eff
        rs_p, rhos_p, _, _ = nfw_params(m_eff + dM, z_lens)
        rs_m, rhos_m, _, _ = nfw_params(m_eff - dM, z_lens)
        kp = 2.0 * (rs_p * rhos_p / sigmac) * fg_kappa(np.maximum(y_grid / rs_p, 1.0e-12))
        km = 2.0 * (rs_m * rhos_m / sigmac) * fg_kappa(np.maximum(y_grid / rs_m, 1.0e-12))
        return (kp - km) / (2.0 * dM)

    ones = np.ones((n_mass, n_y))
    k_ns_mean = area_weighted_mean(y_grid, rmax_host, kappa_ns)

    rows = []
    for psi_min in psi_mins:
        if psi_min >= PSI_MAX:
            # nothing resolved: host stays at full mass, no clumps, no scatter at all
            rows.append({
                "psi_min": float(psi_min), "frac_res": 0.0,
                "kappa_total": k_ns_mean,
                "sigma_host": 0.0, "sigma_sub": 0.0, "sigma_total": 0.0,
            })
            continue

        above = psi >= psi_min
        w = ones * above[:, None]
        mu_y = mu_of(w)
        var_y = var_of(w)
        frac_y = frac_of(w)
        frac_res = float(np.mean(frac_y))
        varm = varm_of(above)
        covmk_y = covmk_of(w)
        k_host_y = host_kappa_reduced(frac_y)
        dkdM_y = host_dkdM(frac_res)

        var_host_y = (dkdM_y**2) * varm
        cov_hk_y = -dkdM_y * covmk_y
        total_mean_y = k_host_y + mu_y

        var_host_aw = area_weighted_mean(y_grid, rmax_host, var_host_y)
        var_sub_aw = area_weighted_mean(y_grid, rmax_host, var_y)
        cov_hk_aw = area_weighted_mean(y_grid, rmax_host, cov_hk_y)
        var_tot_aw = var_host_aw + var_sub_aw + 2.0 * cov_hk_aw

        rows.append({
            "psi_min": float(psi_min),
            "frac_res": frac_res,
            "kappa_total": area_weighted_mean(y_grid, rmax_host, total_mean_y),
            "sigma_host": float(np.sqrt(max(var_host_aw, 0.0))),
            "sigma_sub": float(np.sqrt(max(var_sub_aw, 0.0))),
            "sigma_total": float(np.sqrt(max(var_tot_aw, 0.0))),
        })

    return {"k_ns_mean": k_ns_mean, "host_mass": host_mass, "z_lens": z_lens,
            "z_source": z_source}, rows


def main() -> None:
    host_mass, z_lens, z_source = 1.0e14, 0.5, 1.0
    kappa_thr_host = 1.276e-4
    production_m_floor = 1.0e7
    psi_mins = np.logspace(-8, 2, 55)
    psi_prod = production_m_floor / host_mass

    meta, rows = run(host_mass, z_lens, z_source, kappa_thr_host, psi_mins)

    psi_min_arr = np.array([r["psi_min"] for r in rows])
    sig_host = np.array([r["sigma_host"] for r in rows])
    sig_sub = np.array([r["sigma_sub"] for r in rows])
    sig_tot = np.array([r["sigma_total"] for r in rows])

    # top axis: psi_min -> peak (r=0) convergence of a single clump at that mass,
    # interpolated off the same grid used for the sweep (monotonic in psi_min).
    kappa_peak = np.array([
        kappa_nfw(max(p, 1e-300) * host_mass, z_lens, z_source, 0.0) for p in psi_min_arr
    ])

    def fwd(x):
        xx = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.full_like(xx, np.nan)
        pos = xx > 0
        out[pos] = 10 ** np.interp(np.log10(xx[pos]), np.log10(psi_min_arr), np.log10(kappa_peak))
        return out

    def inv(x):
        xx = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.full_like(xx, np.nan)
        pos = xx > 0
        out[pos] = 10 ** np.interp(np.log10(xx[pos]), np.log10(kappa_peak), np.log10(psi_min_arr))
        return out

    out_stem = ROOT / "playground" / "analytic" / "plot_sigma_vs_psimin"
    out_stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.2))
    ax.plot(psi_min_arr, sig_tot, "o-", color="k", ms=4, lw=2.2,
            label=r"$\sigma_\kappa$ (total, resolved-only, no $\kappa_u$)")
    ax.plot(psi_min_arr, sig_host, "s--", color="#d97706", ms=4, lw=1.6,
            label=r"$\sigma_{\kappa,\rm host}$ (carve response)")
    ax.plot(psi_min_arr, sig_sub, "^--", color="#7c3aed", ms=4, lw=1.6,
            label=r"$\sigma_{\kappa,\rm sub}$ (resolved clumps)")
    ax.axvline(psi_prod, color="#dc2626", ls=":", lw=1.3,
               label=r"production $\psi_{\rm min}$ ($m_{\rm floor}=10^7\,M_\odot$)")
    ax.axvline(1.0, color="0.6", ls="-", lw=0.8)
    ax.text(1.05, sig_tot.max() * 0.5, r"$\psi_{\max}=1$", fontsize=8, color="0.4")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\psi_{\rm min}$  (resolved floor; everything below stays in the host, no correction)")
    ax.set_ylabel(r"$\sigma_\kappa$  (area-weighted over the aperture)")
    ax.set_title(f"resolved-only scheme (no $\\kappa_u$): $\\sigma_\\kappa$ vs. floor, "
                 f"M={host_mass:.0e}, $z_l$={z_lens}, $z_s$={z_source}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="lower left")

    ax2 = ax.secondary_xaxis("top", functions=(fwd, inv))
    ax2.set_xlabel(r"equivalent peak clump convergence $2\kappa_{0,\rm clump}(\psi_{\rm min})$ at $r=0$")

    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=180, facecolor="white", bbox_inches="tight")
    print(f"saved {out_stem}.png / .json")
    for r in rows[::4]:
        print(f"psi_min={r['psi_min']:.2e}  frac_res={r['frac_res']:.4f}  "
              f"sig_host={r['sigma_host']:.3e}  sig_sub={r['sigma_sub']:.3e}  "
              f"sig_tot={r['sigma_total']:.3e}")
    i_prod = int(np.argmin(np.abs(psi_min_arr - psi_prod)))
    print(f"\nat production psi_min={psi_prod:.2e}: sig_tot={sig_tot[i_prod]:.4e} "
          f"(host={sig_host[i_prod]:.4e}, sub={sig_sub[i_prod]:.4e})")


if __name__ == "__main__":
    main()
