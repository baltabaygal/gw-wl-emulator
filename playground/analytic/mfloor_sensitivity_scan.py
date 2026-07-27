"""
m_floor sensitivity scan: does the unresolved-band Gaussian term (kappa_u, its mean
mu_u and Campbell second moment C2_u = sigma_u^2) actually depend on where the
absolute clump-mass floor m_floor is set -- and does that dependence survive in
the TOTAL host+substructure mean once the smooth host is carved by the same
extended bound fraction f_b(m_floor)?

Question this answers (chat 2026-07-22, mass-conserving-carve follow-up): the
analytic ~psi_min^{1+alpha}=psi_min^0.18 argument said mu_u/M_u should be strongly
m_floor-sensitive (same slow power as the bound-mass fraction itself) while
C2_u/sigma_u^2 should be much better protected (~psi_min^{2+alpha}=psi_min^1.18).
It also predicted the TOTAL mean kappa_halo (reduced host + resolved clumps +
kappa_u mean) should be *partially* protected against m_floor by the mass-
conservation identity, but not exactly, because subhalos trace a radially-biased
profile (depleted near the host center) rather than the host's own NFW cusp.

This reuses the validated Campbell-moment machinery from wsub_partition_proof.py
(itself checked against the C++ to <~1.6%, docs/subhalo/subhalo_combining.md) at a
FIXED production subhalo_factor=1e-2, scanning only m_floor.

Run (test env not required beyond numpy/scipy/matplotlib):
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/mfloor_sensitivity_scan.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from playground.analytic.wsub_partition_proof import run  # noqa: E402


def area_weighted_mean(y_grid, rmax_host, values):
    q_y = 2.0 * y_grid / rmax_host**2
    wt_y = y_grid * np.gradient(np.log(y_grid))
    ay = q_y * wt_y
    return float(np.sum(ay * values) / np.sum(ay))


def scan(host_mass, z_lens, z_source, kappa_thr_host, subhalo_factor, m_floors):
    rows = []
    for m_floor in m_floors:
        # reuse run() at a single factor to get mu_b/C2_b (full brute set, down to
        # m_floor) and mu_p/C2_p (resolved, at the fixed production factor) --
        # recompute inline since `run` doesn't return the per-y arrays, so we
        # monkey-copy its body's relevant pieces via a light re-import trick:
        # simplest robust path is to call run() and also independently rebuild the
        # y-grid quantities by importing the same helpers it uses.
        from scripts.subhalo_gate.subhalo_factor_proxy_check import (
            ALPHA, BETA, OMEGA, PSI_MAX, build_reach_interpolator, fg_kappa,
            gamma_norm, n_tau, nfw_params, sigma_crit,
        )
        from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax
        from playground.analytic.subhalo_factor_analytic_deficit import (
            distance_kernel, projected_profile,
        )

        rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
        rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
        nt, _ = n_tau(host_mass, z_lens)
        fs = 0.3563 / nt**0.6 - 0.075
        gamma = gamma_norm(fs)
        sigmac = sigma_crit(z_source, z_lens)
        kappa0_host = rs_host * rhos_host / sigmac

        m_deep = 1.0e3
        psi_deep = m_deep / host_mass
        psi_floor = m_floor / host_mass
        n_mass, n_y, n_d, n_theta = 100, 160, 240, 256
        lpsi = np.linspace(np.log(psi_deep), np.log(PSI_MAX), n_mass)
        psi = np.exp(lpsi)
        mass = psi * host_mass
        dN_dlnpsi = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)
        above_floor = psi >= psi_floor

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

        J1 = np.einsum("yd,md->my", fd, k1, optimize=True)
        J2 = np.einsum("yd,md->my", fd, k2, optimize=True)

        def cum(weight_my):
            c1 = np.trapezoid(dN_dlnpsi[:, None] * J1 * weight_my, lpsi, axis=0)
            c2 = np.trapezoid(dN_dlnpsi[:, None] * J2 * weight_my, lpsi, axis=0)
            return c1, c2

        def mass_frac(weight_my):
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
        w_brute = ones * above_floor[:, None]
        mu_b, C2_b = cum(w_brute)
        frac_b = mass_frac(w_brute)

        reach = build_reach_interpolator(
            m_min=m_deep, m_max=PSI_MAX * host_mass, zl=z_lens, zs=z_source,
            kappa_thr=subhalo_factor * kappa_thr_host,
        )(mass)
        keep = (y_grid[None, :] <= reach[:, None]) & above_floor[:, None]
        mu_p, C2_p = cum(keep.astype(float))
        frac_p = mass_frac(keep.astype(float))

        mu_u = mu_b - mu_p
        C2_u = C2_b - C2_p

        host_full = host_kappa_reduced(frac_b)          # carve A: reduced by f_b
        total_mean_y = host_full + mu_b                 # host + (resolved+unresolved) mean
        total_ratio_y = total_mean_y / kappa_ns          # closure vs the unreduced host

        rows.append({
            "m_floor": m_floor,
            "psi_min": psi_floor,
            "f_b": area_weighted_mean(y_grid, rmax_host, frac_b[None, :].repeat(1, 0)[0] * 0 + frac_b) if False else float(np.mean(frac_b)),
            "mu_u_mean": area_weighted_mean(y_grid, rmax_host, mu_u),
            "C2_u_mean": area_weighted_mean(y_grid, rmax_host, C2_u),
            "mu_b_mean": area_weighted_mean(y_grid, rmax_host, mu_b),
            "C2_b_mean": area_weighted_mean(y_grid, rmax_host, C2_b),
            "total_closure_ratio": area_weighted_mean(y_grid, rmax_host, total_ratio_y),
        })
    return rows


def main() -> None:
    host_mass = 1.0e14
    z_lens, z_source = 0.5, 1.0
    kappa_thr_host = 1.276e-4
    subhalo_factor = 1.0e-2   # production default (2026-07-12 flip)
    m_floors = [1.0e5, 1.0e6, 1.0e7, 1.0e8]

    rows = scan(host_mass, z_lens, z_source, kappa_thr_host, subhalo_factor, m_floors)

    ref = rows[2]  # m_floor=1e7 production default, index 2
    print(f"host M={host_mass:.0e}, z_l={z_lens}, z_s={z_source}, factor={subhalo_factor}\n")
    print(f"{'m_floor':>10} {'psi_min':>10} {'mu_u':>12} {'C2_u':>12} "
          f"{'mu_u/ref':>10} {'C2_u/ref':>10} {'closure':>10}")
    for r in rows:
        print(f"{r['m_floor']:10.0e} {r['psi_min']:10.2e} {r['mu_u_mean']:12.4e} "
              f"{r['C2_u_mean']:12.4e} {r['mu_u_mean']/ref['mu_u_mean']:10.3f} "
              f"{r['C2_u_mean']/ref['C2_u_mean']:10.3f} {r['total_closure_ratio']:10.5f}")

    out = ROOT / "playground" / "analytic" / "mfloor_sensitivity_scan.json"
    out.write_text(json.dumps({
        "meta": {"host_mass": host_mass, "z_lens": z_lens, "z_source": z_source,
                 "kappa_thr_host": kappa_thr_host, "subhalo_factor": subhalo_factor},
        "rows": rows,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
