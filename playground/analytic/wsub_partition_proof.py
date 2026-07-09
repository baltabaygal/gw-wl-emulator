"""
Analytic proof-of-principle for the unresolved-subhalo Gaussian term ("Wsub").

Proposed scheme, per resolved host encountered at ray impact parameter y:

    kappa_host_event = kappa_smooth((1-f_res(y)) M)                 [as now]
                     + sum_i kappa_resolved_clump_i                 [as now]
                     + dkappa_Wsub,   dkappa_Wsub ~ N(0, sigma_Wsub^2(M, z_l, y))

with sigma_Wsub^2 the Campbell second moment of the clumps DROPPED by the
proxy-r gate (reach(m) < y), integrated over the true clump-ray distance d:

    sigma_Wsub^2(y) = Nbar int dlnpsi w(psi) [1 - keep(psi, y)] int dd f(d|y) kappa_psi(d)^2

Because the clump field is (given y) a marked Poisson process, restriction to
disjoint mass sets gives independent components, so cumulants add EXACTLY:
C_n,brute(y) = C_n,resolved(y) + C_n,unresolved(y) at ANY floor. The Gaussian
surrogate therefore restores the paired kappa^2 excess up to (a) the mean-profile
mismatch (unresolved mean carried by the reduced NFW host instead of the
anti-biased clump profile) and (b) the dropped third-and-higher cumulants.
This script quantifies both, giving:

  1. E_new/E_brute vs subhalo_factor (current proxy deficit vs deficit after
     adding sigma_Wsub^2) -> how far the factor can rise;
  2. the unresolved share of the third cumulant + the per-y skewness of the
     dropped set -> the Gaussianity ceiling (trap 2 of the handoff note);
  3. the sub-m_floor (m < 1e7) variance now recoverable for free.

Physics functions come from scripts/subhalo_gate/subhalo_factor_proxy_check.py
(validated against the C++ to <~1.6%, 2026-07-07); geometry kernels from
playground/analytic/subhalo_factor_analytic_deficit.py (validated against the
stratified MC, docs/subhalo/subhalo_factor_closure.md).

Run (test env not required; numpy/scipy/matplotlib only):
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/wsub_partition_proof.py
Writes wsub_partition_proof.{json,png} next to this file.
"""
from __future__ import annotations

import argparse
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
    ALPHA, BETA, OMEGA, PSI_MAX,
    build_reach_interpolator, fg_kappa, gamma_norm, n_tau, nfw_params, sigma_crit,
)
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402
from playground.analytic.subhalo_factor_analytic_deficit import (  # noqa: E402
    distance_kernel, projected_profile,
)


def run(host_mass, z_lens, z_source, kappa_thr_host, m_floor, factors,
        m_deep=1.0e3, n_mass=100, n_y=160, n_d=240, n_theta=256):
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0_host = rs_host * rhos_host / sigmac

    # mass grid down to m_deep (below the hard m_floor) so the sub-m_floor
    # variance comes out of the same quadrature; the production population is
    # the psi >= m_floor/M part.
    psi_deep = m_deep / host_mass
    psi_floor = m_floor / host_mass
    lpsi = np.linspace(np.log(psi_deep), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    mass = psi * host_mass
    # Campbell intensity in log-psi: dN/dlnpsi = gamma * psi^alpha * exp(-beta psi^omega)
    dN_dlnpsi = gamma * psi**ALPHA * np.exp(-BETA * psi**OMEGA)
    above_floor = psi >= psi_floor

    # clump kappa^n profiles on the d grid
    d_grid = np.logspace(-3, np.log10(rmax_host + r200_host), n_d)
    k1 = np.empty((n_mass, n_d))
    for i, m in enumerate(mass):
        rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
        kappa0 = rs_i * rhos_i / sigmac
        k1[i] = 2.0 * kappa0 * fg_kappa(np.maximum(d_grid / rs_i, 1.0e-12))
    k2, k3 = k1**2, k1**3

    # geometry kernel: f(d|y) from the projected anti-biased profile
    sigma_interp, _ = projected_profile(c_host, r200_host)
    y_grid = np.logspace(-1, np.log10(rmax_host), n_y)
    q_y = 2.0 * y_grid / rmax_host**2
    wt_y = y_grid * np.gradient(np.log(y_grid))
    ay = q_y * wt_y                                    # area-weighted y quadrature
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)
    wt_d = np.gradient(np.log(d_grid))
    fd = f_dy * (d_grid * wt_d)[None, :]               # kernel incl. log-d quadrature

    kappa_ns = 2.0 * kappa0_host * fg_kappa(np.maximum(y_grid / rs_host, 1.0e-12))

    # per-mass position expectations E[kappa^n | psi, y] (n_mass, n_y)
    J1 = np.einsum("yd,md->my", fd, k1, optimize=True)
    J2 = np.einsum("yd,md->my", fd, k2, optimize=True)
    J3 = np.einsum("yd,md->my", fd, k3, optimize=True)

    def cum(weight_my):
        """Campbell cumulants C1..C3(y) for the mass window weight_my (n_mass, n_y)."""
        c1 = np.trapezoid(dN_dlnpsi[:, None] * J1 * weight_my, lpsi, axis=0)
        c2 = np.trapezoid(dN_dlnpsi[:, None] * J2 * weight_my, lpsi, axis=0)
        c3 = np.trapezoid(dN_dlnpsi[:, None] * J3 * weight_my, lpsi, axis=0)
        return c1, c2, c3

    def mass_frac(weight_my):
        """resolved (kept) bound-mass fraction f_res(y) = int dN psi over the window."""
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

    def excess(mu_y, C_y, frac_y):
        """pooled paired kappa^2 excess (same estimator as the deficit script)."""
        delta_y = host_kappa_reduced(frac_y) - kappa_ns
        md = mu_y + delta_y
        e = np.sum(ay * (C_y + md**2 + 2.0 * kappa_ns * md))
        e -= np.sum(ay * md) ** 2 + 2.0 * np.sum(ay * kappa_ns) * np.sum(ay * md)
        return e

    ones = np.ones((n_mass, n_y))
    w_brute = ones * above_floor[:, None]              # production brute: all m >= m_floor
    mu_b, C2_b, C3_b = cum(w_brute)
    frac_b = mass_frac(w_brute)
    E_brute = excess(mu_b, C2_b, frac_b)

    # sub-m_floor bonus: variance in m_deep <= m < m_floor (all of it unresolved today)
    w_deep = ones * (~above_floor)[:, None]
    _, C2_deep, C3_deep = cum(w_deep)
    var_below_floor = np.sum(ay * C2_deep) / np.sum(ay * C2_b)

    rows = []
    for f in factors:
        reach = build_reach_interpolator(
            m_min=m_deep, m_max=PSI_MAX * host_mass, zl=z_lens, zs=z_source,
            kappa_thr=f * kappa_thr_host,
        )(mass)
        keep = (y_grid[None, :] <= reach[:, None]) & above_floor[:, None]

        mu_p, C2_p, C3_p = cum(keep.astype(float))
        frac_p = mass_frac(keep.astype(float))

        # unresolved complement (above m_floor); exact by cumulant additivity
        mu_u = mu_b - mu_p
        C2_u = C2_b - C2_p
        C3_u = C3_b - C3_p

        E_proxy = excess(mu_p, C2_p, frac_p)
        # (a) variance-only fix: keep current host bookkeeping (reduced by f_res),
        #     add zero-mean Gaussian with the dropped conditional variance
        E_new = excess(mu_p, C2_p + C2_u, frac_p)
        # (b) mean-only fix: host reduced by the FULL brute fraction f_b, explicit
        #     deterministic unresolved mean profile mu_u(y); no Gaussian
        E_meanfix = excess(mu_p + mu_u, C2_p, frac_b)
        # (c) both = exact identity with brute (numerical check of the bookkeeping)
        E_full = excess(mu_p + mu_u, C2_p + C2_u, frac_b)

        # Gaussianity diagnostics of the dropped set
        pos = C2_u > 0.0
        g1 = np.where(pos, C3_u / np.maximum(C2_u, 1e-300)**1.5, 0.0)
        g1_area = np.sum(ay * g1) / np.sum(ay)
        g1_max = float(np.max(g1))
        c3_share = np.sum(ay * C3_u) / np.sum(ay * C3_b)
        var_share = np.sum(ay * C2_u) / np.sum(ay * C2_b)

        rows.append({
            "subhalo_factor": float(f),
            "proxy_vs_brute": float(E_proxy / E_brute),
            "varfix_vs_brute": float(E_new / E_brute),
            "meanfix_vs_brute": float(E_meanfix / E_brute),
            "full_vs_brute": float(E_full / E_brute),
            "var_share_unres": float(var_share),
            "c3_share_unres": float(c3_share),
            "skew_unres_area": float(g1_area),
            "skew_unres_max": g1_max,
        })
        print(f"factor={f:9.3g}  proxy={E_proxy/E_brute:6.4f}  "
              f"varfix={E_new/E_brute:6.4f}  meanfix={E_meanfix/E_brute:6.4f}  "
              f"full={E_full/E_brute:8.6f}  varU={var_share:6.4f}  "
              f"c3U={c3_share:6.4f}")

    meta = {
        "host_mass": host_mass, "z_lens": z_lens, "z_source": z_source,
        "kappa_thr_host": kappa_thr_host, "m_floor": m_floor, "m_deep": m_deep,
        "r200_host": r200_host, "rmax_host": rmax_host, "c_host": c_host,
        "fs": fs, "E_brute": float(E_brute),
        "var_below_mfloor_frac": float(var_below_floor),
        "c3_below_mfloor_frac": float(np.sum(ay * C3_deep) / np.sum(ay * C3_b)),
    }
    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--factors", type=float, nargs="+",
                        default=list(np.logspace(-5, 0, 11)))
    parser.add_argument("--out-stem", type=Path,
                        default=ROOT / "playground" / "analytic" / "wsub_partition_proof")
    args = parser.parse_args()

    meta, rows = run(args.host_mass, args.z_lens, args.z_source,
                     args.kappa_thr_host, args.m_floor, args.factors)
    print(f"\nsub-m_floor (m<{args.m_floor:.0e}) variance share: "
          f"{meta['var_below_mfloor_frac']:.2e}  (c3 share {meta['c3_below_mfloor_frac']:.2e})")

    Path(str(args.out_stem) + ".json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2))

    f = np.array([r["subhalo_factor"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    ax.plot(f, [r["proxy_vs_brute"] for r in rows], "o-", color="#2563eb",
            label="current: resolved only")
    ax.plot(f, [r["varfix_vs_brute"] for r in rows], "s-", color="#dc2626",
            label=r"+ Gaussian $\sigma^2_{W,\rm sub}$ only")
    ax.plot(f, [r["meanfix_vs_brute"] for r in rows], "^-", color="#d97706",
            label=r"+ mean profile $\mu_{\rm unres}(y)$ only")
    ax.plot(f, [r["full_vs_brute"] for r in rows], "d-", color="#059669",
            label="both (exact identity)")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.1)
    ax.set_xscale("log"); ax.set_xlabel("subhalo_factor")
    ax.set_ylabel(r"paired $\kappa^2$ excess / brute")
    ax.set_title("variance closure"); ax.grid(alpha=0.3, which="both"); ax.legend()

    ax = axes[1]
    ax.plot(f, [r["c3_share_unres"] for r in rows], "o-", color="#7c3aed",
            label=r"unresolved share of clump $c_3$")
    ax.plot(f, [r["skew_unres_area"] for r in rows], "s-", color="#059669",
            label=r"skewness $\gamma_1$ of dropped set (area-wt)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("subhalo_factor")
    ax.set_title("Gaussianity ceiling"); ax.grid(alpha=0.3, which="both"); ax.legend()

    fig.suptitle(f"Wsub partition proof: M={args.host_mass:.0e}, "
                 f"$z_l$={args.z_lens:g}, $z_s$={args.z_source:g}", y=1.0)
    fig.tight_layout()
    fig.savefig(str(args.out_stem) + ".png", dpi=180, facecolor="white")
    print(f"saved {args.out_stem}.png / .json")


if __name__ == "__main__":
    main()
