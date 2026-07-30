"""
sigma_kappa vs a PER-SUBHALO CONVERGENCE THRESHOLD kappa_thr,sub, at FIXED psi_min.

Difference from plot_sigma_vs_psimin.py (chat 2026-07-24)
--------------------------------------------------------
There the knob was psi_min itself: each point regenerated the clump population with a
different resolved MASS floor, and the kappa axis was cosmetic (the peak convergence of
the floor-mass clump). Here the population is FROZEN at the production floor
psi_min = m_floor/M -- every subhalo is always sampled -- and a SEPARATE knob
kappa_thr,sub acts on each clump's ACTUAL convergence at the ray,

    kappa_i = kappa_NFW(m_i, d_i),   d_i = |ray - clump| in the lens plane,

dropping the clump from the sum when kappa_i < kappa_thr,sub. This is the same rule the
HOST halos already obey (cpp/subhalo.cpp's r_thr table is exactly "include the object
while its kappa at the ray exceeds kappa_thr"), so kappa_thr,sub is directly comparable
to the host counting threshold kappa_thr ~ 1.3e-4 -- which the psi_min version was not.

The cut does NOT commute with a mass cut: a massive clump far from the ray is dropped
while a light clump the ray nearly hits survives. So the threshold must be applied INSIDE
the clump-ray distance integral, not as a mass mask outside it.

Campbell still applies exactly -- a threshold only restricts the integration domain. With
the Poisson clump intensity dN/dlnpsi and the projected clump-ray separation pdf f(d|y),
every moment below is a 2D quadrature over (lnpsi, d) with the indicator folded in:

    <N_ret | y>       = int dlnpsi dN/dlnpsi   int dd f(d|y) THETA
    <kappa_sub | y>   = int dlnpsi dN/dlnpsi   int dd f(d|y) k1 THETA
    Var(kappa_sub|y)  = int dlnpsi dN/dlnpsi   int dd f(d|y) k1^2 THETA
    <M_ret | y>       = int dlnpsi dN/dlnpsi m int dd f(d|y) THETA
    Var(M_ret | y)    = int dlnpsi dN/dlnpsi m^2 int dd f(d|y) THETA
    Cov(M_ret,k_sub|y)= int dlnpsi dN/dlnpsi m int dd f(d|y) k1 THETA
    THETA = 1[k1(m,d) >= kappa_thr,sub]

Dropped clumps keep their mass in the smooth host (mass-conserving, scheme A): the host
is carved at M - M_ret, so the carve is now y-DEPENDENT (it was a scalar under a mass
cut) and weakens as the threshold rises. sigma_host is the first-order carve response
|dkappa_NFW/dM| * sqrt(Var(M_ret)), matching Fig 4 / plot_sigma_vs_psimin.

Bonus: this axis is free of the cusp ambiguity of the psi_min top axis. We never need a
clump's "peak" (r=0) convergence -- which is log-divergent and grid-defined -- only its
convergence at the actual ray separation.

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/sigma_vs_subkappathr.py
Writes sigma_vs_subkappathr.{png,json} next to this file.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_gate.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX, nfw_params, n_tau, gamma_norm, sigma_crit, fg_kappa,
)
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.analytic.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile, production_profile_params, eta_vir_to_200,
    _nfw_mu,
)
from playground.analytic.plot_sigma_vs_psimin import area_weighted_mean  # noqa: E402


def run_kthr(host_mass, z_lens, z_source, kappa_thr_host, kthr_subs,
             m_floor=1.0e7, n_mass=200, n_y=160, n_d=1200, n_theta=256,
             y_probe_r200=None, virial=True, legacy_profile=False):
    """Sweep the per-subhalo convergence cut at fixed psi_min = m_floor/host_mass.

    Returns (meta, rows); kthr_subs may include 0.0 (= no cut, fully populated baseline).

    The scalars in each row are area-weighted over the whole aperture, so their absolute
    size is set as much by rmax_host as by the substructure. Pass y_probe_r200 (impact
    parameters in units of r200) to also get the y-RESOLVED quantities at those radii,
    which are what can honestly be divided by the smooth-host convergence kappa_ns(y) --
    both then live at the same y. Those land in row["probe"], with the no-substructure
    reference kappa_ns(y_probe) in meta["kappa_nosub_probe"].
    """
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0_host = rs_host * rhos_host / sigmac

    # Virial convention (cpp/subhalo.cpp, subhalo_virial; ON in PRODUCTION_CONFIG):
    # psi is referred to M_vir and the population extends to x = eta = r_vir/r_200.
    # M_vir/M_200 = mu(eta c)/mu(c) since both radii share r_s.
    eta = eta_vir_to_200(c_host, z_lens)
    Mpsi = host_mass * _nfw_mu(eta * c_host) / _nfw_mu(c_host) if virial else host_mass

    # FIXED clump population: psi_min = m_floor / Mpsi, never varied in this sweep
    psi_min = m_floor / Mpsi
    lpsi = np.linspace(np.log(psi_min), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    mass = psi * Mpsi                 # PHYSICAL clump mass -> its lensing profile
    # The carve removes sum m_i / (M_vir/M_200) from M_200, i.e. a fraction sum psi of
    # M_200, so the host mass-response variable is psi * M_200, not psi * M_vir.
    mass_carve = psi * host_mass
    dN_dlnpsi = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)

    # per-clump convergence on the (mass, ray-clump separation) grid
    d_grid = np.logspace(-3, np.log10(rmax_host + r200_host), n_d)
    k1 = np.empty((n_mass, n_d))
    for i, m in enumerate(mass):
        rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
        kappa0 = rs_i * rhos_i / sigmac
        k1[i] = 2.0 * kappa0 * fg_kappa(np.maximum(d_grid / rs_i, 1.0e-12))

    # f(d|y) quadrature weights (rows sum to 1: it is a normalized pdf in d)
    if legacy_profile:
        # pre-2026-07-28 convention, kept ONLY to reproduce the published sizing
        # numbers so the change can be attributed: bias x0 = 0.54 read as r_200 units
        # (the engine calibrates in r_vir and refit it to 0.86), extent x <= 1.
        from playground.analytic.subhalo_factor_analytic_deficit import LEGACY_X0_R200
        x0_p, xmax_p, shp = LEGACY_X0_R200, 1.0, 2.0
    else:
        x0_p, xmax_p = production_profile_params(c_host, z_lens, virial=virial)
        shp = 1.0
    sigma_interp, _ = projected_profile(c_host, r200_host, x0=x0_p, xmax=xmax_p,
                                        shape_exp=shp)
    y_grid = np.logspace(-1, np.log10(rmax_host), n_y)
    if y_probe_r200 is not None:
        # splice the requested radii into the grid so the probe is exact, not nearest-cell
        y_probe = np.clip(np.asarray(y_probe_r200, float) * r200_host,
                          y_grid[0], y_grid[-1])
        y_grid = np.unique(np.concatenate([y_grid, y_probe]))
        probe_idx = np.searchsorted(y_grid, y_probe)
    fd = distance_kernel(y_grid, d_grid, sigma_interp, n_theta) \
        * (d_grid * np.gradient(np.log(d_grid)))[None, :]
    norm_err = float(np.max(np.abs(fd.sum(axis=1) - 1.0)))

    kappa_ns = 2.0 * kappa0_host * fg_kappa(np.maximum(y_grid / rs_host, 1.0e-12))
    k_ns_mean = area_weighted_mean(y_grid, rmax_host, kappa_ns)

    def host_kappa_reduced(frac_res):
        """Smooth-host kappa(y) with the (y-dependent) carved mass fraction removed."""
        out = np.empty_like(y_grid)
        key = np.round(frac_res, 6)
        for fr in np.unique(key):
            sel = key == fr
            m_eff = max((1.0 - min(fr, 0.95)) * host_mass, 1.0)
            rs_e, rhos_e, _, _ = nfw_params(m_eff, z_lens)
            k0 = rs_e * rhos_e / sigmac
            out[sel] = 2.0 * k0 * fg_kappa(np.maximum(y_grid[sel] / rs_e, 1.0e-12))
        return out

    def host_dkdM(frac_res):
        m_eff = max((1.0 - min(frac_res, 0.95)) * host_mass, 1.0)
        dM = 1e-3 * m_eff
        rs_p, rhos_p, _, _ = nfw_params(m_eff + dM, z_lens)
        rs_m, rhos_m, _, _ = nfw_params(m_eff - dM, z_lens)
        kp = 2.0 * (rs_p * rhos_p / sigmac) * fg_kappa(np.maximum(y_grid / rs_p, 1.0e-12))
        km = 2.0 * (rs_m * rhos_m / sigmac) * fg_kappa(np.maximum(y_grid / rs_m, 1.0e-12))
        return (kp - km) / (2.0 * dM)

    rows = []
    for kthr in kthr_subs:
        # THETA folded into the d integral -- the cut is on kappa AT THE RAY, so it is
        # a d-dependent (not mass-only) domain restriction.
        keep = (k1 >= kthr) if kthr > 0.0 else np.ones_like(k1, dtype=bool)
        Jn = (fd @ keep.T).T                      # <THETA | y>            (m, y)
        J1 = (fd @ (k1 * keep).T).T               # <k1 THETA | y>
        J2 = (fd @ (k1 * k1 * keep).T).T          # <k1^2 THETA | y>

        n_ret_y = np.trapezoid(dN_dlnpsi[:, None] * Jn, lpsi, axis=0)
        mu_y = np.trapezoid(dN_dlnpsi[:, None] * J1, lpsi, axis=0)
        var_y = np.trapezoid(dN_dlnpsi[:, None] * J2, lpsi, axis=0)
        frac_y = np.trapezoid((dN_dlnpsi * psi)[:, None] * Jn, lpsi, axis=0)
        varm_y = np.trapezoid((dN_dlnpsi * mass_carve**2)[:, None] * Jn, lpsi, axis=0)
        covmk_y = np.trapezoid((dN_dlnpsi * mass_carve)[:, None] * J1, lpsi, axis=0)

        frac_res = float(np.mean(frac_y))
        k_host_y = host_kappa_reduced(frac_y)
        dkdM_y = host_dkdM(frac_res)

        var_host_y = (dkdM_y**2) * varm_y
        cov_hk_y = -dkdM_y * covmk_y
        var_tot_y = var_host_y + var_y + 2.0 * cov_hk_y

        var_host_aw = area_weighted_mean(y_grid, rmax_host, var_host_y)
        var_sub_aw = area_weighted_mean(y_grid, rmax_host, var_y)
        cov_hk_aw = area_weighted_mean(y_grid, rmax_host, cov_hk_y)
        var_tot_aw = var_host_aw + var_sub_aw + 2.0 * cov_hk_aw

        row = {
            "kappa_thr_sub": float(kthr),
            "n_retained": float(area_weighted_mean(y_grid, rmax_host, n_ret_y)),
            "frac_res": frac_res,
            "kappa_total": area_weighted_mean(y_grid, rmax_host, k_host_y + mu_y),
            "kappa_sub": area_weighted_mean(y_grid, rmax_host, mu_y),
            "sigma_host": float(np.sqrt(max(var_host_aw, 0.0))),
            "sigma_sub": float(np.sqrt(max(var_sub_aw, 0.0))),
            "sigma_total": float(np.sqrt(max(var_tot_aw, 0.0))),
        }
        if y_probe_r200 is not None:
            row["probe"] = {
                "n_retained": n_ret_y[probe_idx].tolist(),
                "kappa_total": (k_host_y + mu_y)[probe_idx].tolist(),
                "kappa_sub": mu_y[probe_idx].tolist(),
                "sigma_host": np.sqrt(np.maximum(var_host_y[probe_idx], 0.0)).tolist(),
                "sigma_sub": np.sqrt(np.maximum(var_y[probe_idx], 0.0)).tolist(),
                "sigma_total": np.sqrt(np.maximum(var_tot_y[probe_idx], 0.0)).tolist(),
            }
        rows.append(row)

    meta = {"k_ns_mean": k_ns_mean, "host_mass": host_mass, "z_lens": z_lens,
            "z_source": z_source, "m_floor": m_floor, "psi_min": psi_min,
            "kappa_thr_host": kappa_thr_host, "rmax_host": rmax_host,
            "r200_host": r200_host, "fd_norm_err": norm_err,
            "n_mass": n_mass, "n_y": n_y, "n_d": n_d}
    if y_probe_r200 is not None:
        meta["y_probe"] = y_grid[probe_idx].tolist()
        meta["y_probe_r200"] = (y_grid[probe_idx] / r200_host).tolist()
        meta["kappa_nosub_probe"] = kappa_ns[probe_idx].tolist()
    return meta, rows


def main() -> None:
    host_mass, z_lens, z_source = 1.0e14, 0.5, 1.0
    kappa_thr_host = 1.276e-4
    kthr = np.concatenate([[0.0], np.logspace(-7, 1.3, 60)])

    meta, rows = run_kthr(host_mass, z_lens, z_source, kappa_thr_host, kthr)
    print(f"f(d|y) normalization max error: {meta['fd_norm_err']:.2e}")

    out_stem = ROOT / "playground" / "analytic" / "sigma_vs_subkappathr"
    out_stem.with_suffix(".json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))

    k = np.array([r["kappa_thr_sub"] for r in rows])
    sig_tot = np.array([r["sigma_total"] for r in rows])
    sig_sub = np.array([r["sigma_sub"] for r in rows])
    sig_host = np.array([r["sigma_host"] for r in rows])
    nret = np.array([r["n_retained"] for r in rows])
    base = sig_tot[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    ax = axes[0]
    ax.plot(k[1:], sig_tot[1:], "o-", color="k", ms=3, lw=2.0, label=r"$\sigma_\kappa$ total")
    ax.plot(k[1:], sig_sub[1:], "^--", color="#7c3aed", ms=3, lw=1.5, label=r"$\sigma_{\kappa,\rm sub}$")
    ax.plot(k[1:], sig_host[1:], "s--", color="#d97706", ms=3, lw=1.5, label=r"$\sigma_{\kappa,\rm host}$ (carve)")
    ax.axhline(base, color="0.6", lw=0.8, ls="-")
    ax.axvline(kappa_thr_host, color="#dc2626", ls=":", lw=1.3, label=r"host $\kappa_{\rm thr}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_{\rm thr,sub}$ (drop clumps whose $\kappa$ at the ray is below this)")
    ax.set_ylabel(r"$\sigma_\kappa$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot(k[1:], nret[1:], "o-", color="#0f766e", ms=3, lw=1.8)
    ax.axvline(kappa_thr_host, color="#dc2626", ls=":", lw=1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\kappa_{\rm thr,sub}$")
    ax.set_ylabel(r"mean number of clumps retained per sightline")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(f"fixed $\\psi_{{\\rm min}}$={meta['psi_min']:.1e}, "
                 f"M={host_mass:.0e}, $z_l$={z_lens}, $z_s$={z_source}")
    fig.tight_layout()
    fig.savefig(str(out_stem) + ".png", dpi=170, facecolor="white", bbox_inches="tight")
    print(f"saved {out_stem}.png / .json")

    print(f"\nbaseline (no cut): sigma_tot={base:.4e} sigma_sub={sig_sub[0]:.4e} "
          f"sigma_host={sig_host[0]:.4e} <N>={nret[0]:.1f}")
    f = sig_tot / base
    for frac in (0.99, 0.95, 0.90, 0.50):
        j = np.where(f[1:] <= frac)[0]
        if len(j):
            i = j[0] + 1
            print(f"  sigma_tot -> {frac:.0%} of baseline at kappa_thr,sub={k[i]:.3g} "
                  f"(<N_ret>={nret[i]:.3g})")
    i_h = int(np.argmin(np.abs(k[1:] - kappa_thr_host))) + 1
    print(f"  at the HOST threshold {kappa_thr_host:.3e}: sigma_tot/base={f[i_h]:.4f}, "
          f"<N_ret>={nret[i_h]:.4g} of {nret[0]:.4g}")


if __name__ == "__main__":
    main()
