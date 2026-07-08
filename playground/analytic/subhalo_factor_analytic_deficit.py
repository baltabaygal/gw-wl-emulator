"""
Analytic (Campbell-integral) prediction of the subhalo_factor gate deficits.

For a single host (M, z_l, z_s) the matched-realization expectation of the
clump kappa^2 sum measured by subhalo_factor_dgate_area_scan.py is exactly

    E_gate = Nbar * int dm p_m(m) int dy q(y) int dd f(d|y) kappa_m(d)^2 * 1_gate

where
  p_m(m)  : SHMF mass pdf  ~ psi^(alpha-1) exp(-beta psi^omega), psi = m/M
  Nbar    : mean clump count above m_floor (power-law normalization, matching
            sample_brute_realization)
  q(y)    : area-weighted ray impact parameter pdf, 2y/rmax^2 on [0, rmax]
  f(d|y)  : pdf of clump-ray distance d given ray at host-center distance y,
            from the projected anti-biased subhalo radial profile
  gates   : brute (all), proxy-r (reach(m) >= y), truth-d (reach(m) >= d)

No Monte Carlo anywhere - straight quadrature. The output is compared against
the matched-realization scans (subhalo_factor_dgate_area_scan / paired_area_scan).

Physics functions come from scripts/subhalo_factor_proxy_check.py, which was
validated against the C++ (kappa/reach/NFW params/SHMF norm within <~1%,
2026-07-07).
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

from scripts.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX,
    build_reach_interpolator, fg_kappa, gamma_norm, n_tau, nfw_params, sigma_crit,
)
from playground.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402


def projected_profile(c_host: float, r200: float, n_grid: int = 2048):
    """Projected (2D) surface pdf sigma(R) of the anti-biased subhalo profile.

    3D radial pdf (per sample_biased_radii): p3(x) ~ x^2/(1+cx)^2 * bias(x),
    x = r3d/r200 on [0,1].  Isotropic projection: R = r3d*sin(theta) with
    cos(theta) uniform, so  p2(R) = int_R^r200 p3(r) * R/(r*sqrt(r^2-R^2)) dr.
    Returns sigma(R) = p2(R)/(2 pi R)  (surface density, normalized so that
    int sigma 2 pi R dR = 1) as an interpolator plus the R grid.
    """
    x = np.linspace(1.0e-6, 1.0, n_grid)
    bias = 1.0 / np.sqrt((x / 0.54) ** (-2.5) + 1.0)
    w3 = x**2 / (1.0 + c_host * x) ** 2 * bias
    w3 /= np.trapezoid(w3, x)                      # p3(x), x = r/r200

    def p3_interp(q: np.ndarray) -> np.ndarray:
        return np.interp(q, x, w3, left=w3[0], right=0.0)

    # p2(R) = int_R^1 p3(r) R/(r sqrt(r^2-R^2)) dr.  The integrand has an
    # integrable 1/sqrt(r-R) singularity at r=R; substitute u = sqrt(r^2-R^2)
    # (r = sqrt(R^2+u^2), dr = u du / r) which removes it exactly:
    # p2(R) = int_0^{sqrt(1-R^2)} p3(sqrt(R^2+u^2)) * R/(R^2+u^2) du
    R = np.linspace(1.0e-6, 1.0 - 1.0e-9, n_grid)  # units of r200
    p2 = np.empty_like(R)
    for i, Ri in enumerate(R):
        umax = np.sqrt(max(1.0 - Ri**2, 0.0))
        if umax <= 0.0:
            p2[i] = 0.0
            continue
        u = np.linspace(0.0, umax, 512)
        r = np.sqrt(Ri**2 + u**2)
        p2[i] = np.trapezoid(p3_interp(r) * Ri / r**2, u)
    p2 /= np.trapezoid(p2, R)                      # p2(R), R in units of r200

    R_kpc = R * r200
    sigma = p2 / (2.0 * np.pi * np.maximum(R_kpc, 1.0e-12)) / r200

    def sigma_interp(q: np.ndarray) -> np.ndarray:
        return np.interp(q, R_kpc, sigma, left=sigma[0], right=0.0)

    return sigma_interp, R_kpc


def distance_kernel(y_grid, d_grid, sigma_interp, n_theta: int = 256):
    """f(d|y) on (len(y), len(d)) grid: f = d * int_0^2pi sigma(sqrt(y^2+d^2-2yd cos t)) dt."""
    theta = np.linspace(0.0, np.pi, n_theta)       # symmetric: integrate half, double
    ct = np.cos(theta)
    f = np.empty((len(y_grid), len(d_grid)))
    for i, y in enumerate(y_grid):
        s = np.sqrt(np.maximum(y**2 + d_grid[:, None]**2 - 2.0 * y * d_grid[:, None] * ct[None, :], 0.0))
        f[i] = d_grid * 2.0 * np.trapezoid(sigma_interp(s), theta, axis=1)
    return f


def run(host_mass, z_lens, z_source, kappa_thr_host, m_floor, factors,
        n_mass=80, n_y=160, n_d=200, n_theta=256, pooled=True):
    """Full analytic paired excess per host, per gate.

    With X(y) = delta_host(y) + S(y) (S = kept clump kappa sum) and the clump
    field Poisson given y:
        E[S|y]   = mu(y),   Var[S|y] = C(y)   (Campbell),
        E[X^2 + 2 kappa_ns X | y] = C + (mu+delta)^2 + 2 kappa_ns (mu+delta).
    Production (compound Poisson over hosts):   E = < C + (mu+d)^2 + 2 k_ns (mu+d) >_y
    Playground pooled-variance version subtracts the pooled means:
        E_pool = <...>_y - <mu+d>_y^2 - 2 <k_ns>_y <mu+d>_y
    """
    rs_host, rhos_host, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, _ = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0_host = rs_host * rhos_host / sigmac

    # SHMF: mean count above m_floor (power law, as in sample_brute_realization),
    # mass pdf ~ psi^(alpha-1) exp(-beta psi^omega) renormalized on [psi_lo, 1]
    psi_lo = m_floor / host_mass
    nbar = gamma / ALPHA * (PSI_MAX**ALPHA - psi_lo**ALPHA)
    lpsi = np.linspace(np.log(psi_lo), np.log(PSI_MAX), n_mass)
    psi = np.exp(lpsi)
    w_m = psi**ALPHA * np.exp(-BETA * psi**OMEGA)          # pdf in log-psi
    w_m /= np.trapezoid(w_m, lpsi)
    mass = psi * host_mass

    # clump kappa / kappa^2 profiles on the d grid
    d_grid = np.logspace(-3, np.log10(rmax_host + r200_host), n_d)
    k1 = np.empty((n_mass, n_d))
    for i, m in enumerate(mass):
        rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
        kappa0 = rs_i * rhos_i / sigmac
        k1[i] = 2.0 * kappa0 * fg_kappa(np.maximum(d_grid / rs_i, 1.0e-12))
    k2 = k1**2

    # geometry kernel
    sigma_interp, _ = projected_profile(c_host, r200_host)
    # log-spaced y: the integrand (kappa_ns * mean-shift terms) peaks at small y,
    # which a linear grid under-resolves; area weight 2y/rmax^2 * dy = 2y^2/rmax^2 dln(y)
    y_grid = np.logspace(-1, np.log10(rmax_host), n_y)
    q_y = 2.0 * y_grid / rmax_host**2
    wt_y = y_grid * np.gradient(np.log(y_grid))
    ay = q_y * wt_y                                         # area-weighted y quadrature
    f_dy = distance_kernel(y_grid, d_grid, sigma_interp, n_theta)   # (n_y, n_d)

    # log-d quadrature: dd = d dln(d)
    wt_d = np.gradient(np.log(d_grid))
    fd = f_dy * (d_grid * wt_d)[None, :]                    # kernel incl. d-quadrature

    kappa_ns = 2.0 * kappa0_host * fg_kappa(np.maximum(y_grid / rs_host, 1.0e-12))

    def host_kappa_reduced(frac_res: np.ndarray) -> np.ndarray:
        """kappa_host((1-f)M) at each y, with NFW params re-fit at the reduced mass."""
        out = np.empty_like(y_grid)
        # group identical fractions to avoid redundant nfw_params calls
        for fr in np.unique(np.round(frac_res, 6)):
            sel = np.round(frac_res, 6) == fr
            m_eff = max((1.0 - min(fr, 0.95)) * host_mass, 1.0)
            rs_e, rhos_e, _, _ = nfw_params(m_eff, z_lens)
            k0 = rs_e * rhos_e / sigmac
            out[sel] = 2.0 * k0 * fg_kappa(np.maximum(y_grid[sel] / rs_e, 1.0e-12))
        return out

    def excess(mu_y, C_y, frac_y):
        delta_y = host_kappa_reduced(frac_y) - kappa_ns
        md = mu_y + delta_y
        e = np.sum(ay * (C_y + md**2 + 2.0 * kappa_ns * md))
        if pooled:
            e -= np.sum(ay * md) ** 2 + 2.0 * np.sum(ay * kappa_ns) * np.sum(ay * md)
        return e

    # ---- brute: keep everything above m_floor
    J1 = np.einsum("yd,md->my", fd, k1, optimize=True)      # (n_mass, n_y): int f k dd
    J2 = np.einsum("yd,md->my", fd, k2, optimize=True)
    mu_b = nbar * np.trapezoid(w_m[:, None] * J1, lpsi, axis=0)
    C_b = nbar * np.trapezoid(w_m[:, None] * J2, lpsi, axis=0)
    # resolved mass fraction: nbar * <m * 1_keep> / M  (all kept for brute)
    frac_b = np.full_like(y_grid, nbar * np.trapezoid(w_m * psi, lpsi))
    E_brute = excess(mu_b, C_b, frac_b)

    rows = []
    for f in factors:
        reach = build_reach_interpolator(
            m_min=m_floor, m_max=PSI_MAX * host_mass, zl=z_lens, zs=z_source,
            kappa_thr=f * kappa_thr_host,
        )(mass)

        # proxy-r: keep clump (any d) iff reach(m) >= y
        keep_y = (y_grid[None, :] <= reach[:, None])        # (n_mass, n_y)
        mu_p = nbar * np.trapezoid(w_m[:, None] * J1 * keep_y, lpsi, axis=0)
        C_p = nbar * np.trapezoid(w_m[:, None] * J2 * keep_y, lpsi, axis=0)
        frac_p = nbar * np.trapezoid((w_m * psi)[:, None] * keep_y, lpsi, axis=0)
        E_proxy = excess(mu_p, C_p, frac_p)

        # truth-d: keep iff d <= reach(m)
        keep_d = (d_grid[None, :] <= reach[:, None])        # (n_mass, n_d)
        J1t = np.einsum("yd,md->my", fd, k1 * keep_d, optimize=True)
        J2t = np.einsum("yd,md->my", fd, k2 * keep_d, optimize=True)
        # kept-probability per (m, y) for the mass fraction
        P_keep = np.einsum("yd,md->my", fd, np.broadcast_to(keep_d, (n_mass, n_d)).astype(float),
                           optimize=True)
        mu_t = nbar * np.trapezoid(w_m[:, None] * J1t, lpsi, axis=0)
        C_t = nbar * np.trapezoid(w_m[:, None] * J2t, lpsi, axis=0)
        frac_t = nbar * np.trapezoid((w_m * psi)[:, None] * P_keep, lpsi, axis=0)
        E_truth = excess(mu_t, C_t, frac_t)

        rows.append({
            "subhalo_factor": float(f),
            "E_brute": float(E_brute),
            "E_proxy": float(E_proxy),
            "E_truth": float(E_truth),
            "proxy_vs_brute": float(E_proxy / E_brute),
            "truth_vs_brute": float(E_truth / E_brute),
        })
        print(f"factor={f:.4g}  proxy/brute={E_proxy/E_brute:.4f}  truth/brute={E_truth/E_brute:.4f}")

    meta = {
        "host_mass": host_mass, "z_lens": z_lens, "z_source": z_source,
        "kappa_thr_host": kappa_thr_host, "m_floor": m_floor,
        "r200_host": r200_host, "rmax_host": rmax_host, "nbar": nbar,
        "c_host": c_host, "fs": fs, "pooled": pooled,
    }
    return meta, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--factors", type=float, nargs="+", default=[
        1e-5, 3.1622776601683795e-5, 1e-4, 3.1622776601683795e-4, 1e-3,
        3.1622776601683795e-3, 1e-2, 3.1622776601683795e-2, 1e-1,
        3.1622776601683795e-1, 1.0, 3.1622776601683795, 10.0,
        31.622776601683793, 100.0,
    ])
    parser.add_argument("--mc-json", type=Path,
                        default=ROOT / "playground" / "subhalo_factor_stratified_mc.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "playground" / "subhalo_factor_analytic_deficit.png")
    parser.add_argument("--json-out", type=Path,
                        default=ROOT / "playground" / "subhalo_factor_analytic_deficit.json")
    args = parser.parse_args()

    meta, rows = run(args.host_mass, args.z_lens, args.z_source,
                     args.kappa_thr_host, args.m_floor, args.factors)
    payload = {"meta": meta, "rows": rows}
    args.json_out.write_text(json.dumps(payload, indent=2))

    factors = np.array([r["subhalo_factor"] for r in rows])
    pvb = np.array([r["proxy_vs_brute"] for r in rows])
    tvb = np.array([r["truth_vs_brute"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.plot(factors, pvb, lw=2.2, color="#2563eb", label="proxy-r (analytic)")
    ax.plot(factors, tvb, lw=2.2, color="#dc2626", label="truth-d (analytic)")

    if args.mc_json.exists():
        mc = json.loads(args.mc_json.read_text())
        if isinstance(mc, dict) and "results" in mc:      # stratified MC format
            res = mc["results"]
            eb = res["brute"]["excess"]
            fmc, pmc, perr, tmc, terr = [], [], [], [], []
            for key, r in res.items():
                if key.startswith("proxy_"):
                    fmc.append(float(key.split("_")[1]))
                    pmc.append(r["excess"] / eb)
                    perr.append(r["err"] / abs(eb))
                    t = res[f"truth_{key.split('_')[1]}"]
                    tmc.append(t["excess"] / eb)
                    terr.append(t["err"] / abs(eb))
            ax.errorbar(fmc, pmc, yerr=perr, fmt="o", ms=6, mfc="none", capsize=3,
                        color="#2563eb", label="proxy-r (stratified MC)")
            ax.errorbar(fmc, tmc, yerr=terr, fmt="s", ms=6, mfc="none", capsize=3,
                        color="#dc2626", label="truth-d (stratified MC)")
        else:
            fmc = np.array([r["subhalo_factor"] for r in mc])
            pmc = np.array([r["excess_proxy"] / r["excess_brute"] for r in mc])
            tmc = np.array([r["excess_truth"] / r["excess_brute"] for r in mc])
            ax.plot(fmc, pmc, "o", ms=6, mfc="none", color="#2563eb", label="proxy-r (paired MC)")
            ax.plot(fmc, tmc, "s", ms=6, mfc="none", color="#dc2626", label="truth-d (paired MC)")

    ax.axhline(1.0, color="0.4", ls="--", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("subhalo_factor")
    ax.set_ylabel("retained fraction of brute")
    ax.set_title(f"Analytic Campbell deficit vs matched-realization MC\n"
                 f"M={args.host_mass:.0e}, z_l={args.z_lens:g}, z_s={args.z_source:g}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, facecolor="white")
    print(f"saved {args.out}")
    print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
