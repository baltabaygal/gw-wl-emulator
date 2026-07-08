"""
Stratified, fully-vectorized MC of the single-host paired excess - the numeric
anchor for subhalo_factor_analytic_deficit.py.

Differences from subhalo_factor_paired_area_scan.py:
  * every ray gets an INDEPENDENT clump catalog (no shared-catalog covariance),
  * y is sampled log-uniformly on [y_min, rmax] with importance weight w ~ y^2
    (the area measure), so the small-y region that dominates the mean-shift
    terms is actually sampled,
  * weighted pooled variances; errors from batch splits.

Observable per gate (brute / proxy-r / truth-d), matched catalogs:
    excess = Var_w(kappa_gate) - Var_w(kappa_nosub)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_factor_proxy_check import (  # noqa: E402
    ALPHA, BETA, OMEGA, PSI_MAX,
    build_reach_interpolator, fg_kappa, gamma_norm, n_tau, nfw_params, sigma_crit,
)
from playground.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402


def weighted_var(v: np.ndarray, w: np.ndarray) -> float:
    m = np.sum(w * v) / np.sum(w)
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--factors", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--nrays", type=int, default=40_000)
    parser.add_argument("--y-min", type=float, default=0.1)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--nbatch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--json-out", type=Path,
                        default=ROOT / "playground" / "subhalo_factor_stratified_mc.json")
    args = parser.parse_args()

    HOST, ZL, ZS = args.host_mass, args.z_lens, args.z_source
    rs_h, rhos_h, c_h, r200_h = nfw_params(HOST, ZL)
    rmax = host_rmax(HOST, ZL, ZS, args.kappa_thr_host)
    nt, _ = n_tau(HOST, ZL)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    sigmac = sigma_crit(ZS, ZL)
    psi_lo = args.m_floor / HOST
    nbar = gamma / ALPHA * (PSI_MAX**ALPHA - psi_lo**ALPHA)
    pa_lo, pa_hi = psi_lo**ALPHA, PSI_MAX**ALPHA

    # interpolation tables: clump/host-effective NFW (rs, kappa0) vs mass
    mgrid = np.logspace(np.log10(args.m_floor) - 0.5, np.log10(HOST) + 0.1, 400)
    rs_t = np.empty_like(mgrid)
    k0_t = np.empty_like(mgrid)
    for i, m in enumerate(mgrid):
        rs_i, rhos_i, _, _ = nfw_params(float(m), ZL)
        rs_t[i] = rs_i
        k0_t[i] = rs_i * rhos_i / sigmac
    lm_t = np.log(mgrid)

    def nfw_of_mass(m: np.ndarray):
        lm = np.log(np.maximum(m, mgrid[0]))
        return (np.exp(np.interp(lm, lm_t, np.log(rs_t))),
                np.exp(np.interp(lm, lm_t, np.log(k0_t))))

    def kappa_host(m_eff: np.ndarray, y: np.ndarray) -> np.ndarray:
        rs_e, k0_e = nfw_of_mass(m_eff)
        return 2.0 * k0_e * fg_kappa(np.maximum(y / rs_e, 1.0e-12))

    # anti-biased 3D radial CDF (same as sample_biased_radii)
    xg = np.linspace(0.0, 1.0, 5000)
    xx = np.maximum(xg, 1.0e-9)
    bias = 1.0 / np.sqrt((xx / 0.54) ** (-2.5) + 1.0)
    w3 = xx**2 / (1.0 + c_h * xx) ** 2 * bias
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w3[1:] + w3[:-1]) * np.diff(xg))])
    cdf /= cdf[-1]

    reach_tabs = {}
    for f in args.factors:
        ri = build_reach_interpolator(m_min=args.m_floor, m_max=PSI_MAX * HOST,
                                      zl=ZL, zs=ZS, kappa_thr=f * args.kappa_thr_host)
        reach_tabs[f] = ri

    rng = np.random.default_rng(args.seed)
    n = args.nrays
    y_all = np.exp(rng.uniform(np.log(args.y_min), np.log(rmax), n))
    w_all = y_all**2                       # area measure vs log-uniform proposal
    kns_all = kappa_host(np.full(n, HOST), y_all)

    gates = ["brute"] + [f"proxy_{f:g}" for f in args.factors] + [f"truth_{f:g}" for f in args.factors]
    kap = {g: np.empty(n) for g in gates}

    t0 = time.time()
    for lo in range(0, n, args.chunk):
        hi = min(lo + args.chunk, n)
        nc = hi - lo
        y = y_all[lo:hi]
        counts = rng.poisson(nbar, nc)
        tot = int(counts.sum())
        ray_idx = np.repeat(np.arange(nc), counts)

        # masses: power-law inverse CDF + exp-cutoff rejection (refill, as playground)
        m_flat = np.empty(tot)
        need = np.arange(tot)
        while len(need):
            u = rng.uniform(0.0, 1.0, len(need))
            cand = (pa_lo + u * (pa_hi - pa_lo)) ** (1.0 / ALPHA)
            ok = rng.uniform(0.0, 1.0, len(need)) <= np.exp(-BETA * cand**OMEGA)
            m_flat[need[ok]] = cand[ok] * HOST
            need = need[~ok]

        # positions: anti-biased 3D radius, isotropic projection
        r3d = np.interp(rng.uniform(0.0, 1.0, tot), cdf, xg) * r200_h
        costh = rng.uniform(-1.0, 1.0, tot)
        phi = rng.uniform(0.0, 2.0 * np.pi, tot)
        r2d = r3d * np.sqrt(1.0 - costh**2)
        cx = r2d * np.cos(phi)
        cy = r2d * np.sin(phi)

        d = np.sqrt((cx - y[ray_idx]) ** 2 + cy**2)
        rs_c, k0_c = nfw_of_mass(m_flat)
        kc = 2.0 * k0_c * fg_kappa(np.maximum(d / rs_c, 1.0e-12))

        def assemble(keep: np.ndarray) -> np.ndarray:
            S = np.bincount(ray_idx, weights=kc * keep, minlength=nc)
            mres = np.bincount(ray_idx, weights=m_flat * keep, minlength=nc)
            frac = np.minimum(mres / HOST, 0.95)
            return kappa_host((1.0 - frac) * HOST, y) + S

        kap["brute"][lo:hi] = assemble(np.ones(tot))
        for f in args.factors:
            reach = reach_tabs[f](m_flat)
            kap[f"proxy_{f:g}"][lo:hi] = assemble((reach >= y[ray_idx]).astype(float))
            kap[f"truth_{f:g}"][lo:hi] = assemble((reach >= d).astype(float))
        if (lo // args.chunk) % 10 == 0:
            print(f"  {hi}/{n} rays  [{time.time()-t0:.0f}s]", flush=True)

    # weighted pooled excess + batch errors
    order = rng.permutation(n)
    batches = np.array_split(order, args.nbatch)
    results = {}
    for g in gates:
        e_full = weighted_var(kap[g], w_all) - weighted_var(kns_all, w_all)
        e_b = [weighted_var(kap[g][b], w_all[b]) - weighted_var(kns_all[b], w_all[b])
               for b in batches]
        results[g] = {"excess": e_full, "err": float(np.std(e_b, ddof=1) / np.sqrt(args.nbatch))}

    eb = results["brute"]["excess"]
    print(f"\nnrays={n}  E_brute = {eb:.4e} +/- {results['brute']['err']:.1e}")
    print(f"{'gate':>14} {'excess':>12} {'err':>10} {'ratio to brute':>15}")
    for g in gates:
        r = results[g]
        print(f"{g:>14} {r['excess']:12.4e} {r['err']:10.2e} {r['excess']/eb:15.4f}")

    payload = {"meta": {"host_mass": HOST, "z_lens": ZL, "z_source": ZS,
                        "kappa_thr_host": args.kappa_thr_host, "m_floor": args.m_floor,
                        "nrays": n, "y_min": args.y_min, "rmax": rmax, "nbar": nbar},
               "results": results}
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
