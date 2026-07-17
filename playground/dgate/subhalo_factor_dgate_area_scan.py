"""
Area-weighted host-ray scan for the subhalo split geometry test.

This script moves one step closer to the production observable than the single-ray
prototype by averaging over many ray-host impact parameters r with the correct area
weighting inside the host's own kappathr disk.

For one host mass/redshift and each subhalo_factor, it estimates

    < sum_i kappa_i^2 >_rays

for three matched-realization gates:
  1. brute    : keep all clumps above m_floor
  2. proxy-r  : production-style gate based on host-center distance r
  3. truth-d  : corrected gate based on actual clump-ray distance d

This is still a per-host diagnostic, not the full line-of-sight Var(kappa), but it
should resemble the production convergence curves more closely than the single-ray test.
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
    PSI_MAX,
    build_reach_interpolator,
    fg_kappa,
    gamma_norm,
    n_tau,
    nfw_params,
    sample_brute_realization,
    sigma_crit,
)


def host_rmax(host_mass: float, z_lens: float, z_source: float, kappa_thr_host: float) -> float:
    rs, rhos, _, _ = nfw_params(host_mass, z_lens)
    kappa0 = rs * rhos / sigma_crit(z_source, z_lens)
    lo, hi = 1.0e-6, 1.0e7
    flo = 2.0 * kappa0 * fg_kappa(np.array([max(lo / rs, 1.0e-12)]))[0] - kappa_thr_host
    fhi = 2.0 * kappa0 * fg_kappa(np.array([max(hi / rs, 1.0e-12)]))[0] - kappa_thr_host
    if flo < 0.0:
        return 0.0
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        fmid = 2.0 * kappa0 * fg_kappa(np.array([max(mid / rs, 1.0e-12)]))[0] - kappa_thr_host
        if fmid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1.0e-8:
            break
    return float(np.sqrt(lo * hi))


def run_one_factor(
    host_mass: float,
    z_lens: float,
    z_source: float,
    subhalo_factor: float,
    kappa_thr_host: float,
    m_floor: float,
    seeds: list[int],
    nrays: int,
    ray_seed: int,
) -> dict[str, float]:
    rs_host, _, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, zf = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    kappa_thr_sub = subhalo_factor * kappa_thr_host
    sigmac = sigma_crit(z_source, z_lens)
    reach_interp = build_reach_interpolator(
        m_min=m_floor,
        m_max=PSI_MAX * host_mass,
        zl=z_lens,
        zs=z_source,
        kappa_thr=kappa_thr_sub,
    )

    ray_rng = np.random.default_rng(ray_seed)
    ray_r = rmax_host * np.sqrt(ray_rng.uniform(0.0, 1.0, nrays))
    ray_phi = ray_rng.uniform(0.0, 2.0 * np.pi, nrays)
    ray_x = ray_r * np.cos(ray_phi)
    ray_y = ray_r * np.sin(ray_phi)

    k2_brute_all = []
    k2_proxy_all = []
    k2_truth_all = []
    n_brute_all = []
    n_proxy_all = []
    n_truth_all = []

    for seed in seeds:
        real = sample_brute_realization(seed, host_mass, z_lens, m_floor, gamma, r200_host, c_host)
        mass = np.asarray(real["mass"], dtype=float)
        x = np.asarray(real["x"], dtype=float)
        y = np.asarray(real["y"], dtype=float)
        if len(mass) == 0:
            continue
        reach = reach_interp(mass)
        rs_clump = np.empty_like(mass)
        kappa0_clump = np.empty_like(mass)
        for i, m in enumerate(mass):
            rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
            rs_clump[i] = rs_i
            kappa0_clump[i] = rs_i * rhos_i / sigmac

        for xr, yr, rr, phir in zip(ray_x, ray_y, ray_r, ray_phi):
            cosp = np.cos(phir)
            sinp = np.sin(phir)
            x_aligned = x * cosp + y * sinp
            y_aligned = -x * sinp + y * cosp
            d = np.sqrt((x_aligned - rr) ** 2 + y_aligned**2)
            xshape = np.maximum(d / rs_clump, 1.0e-12)
            kappa = 2.0 * kappa0_clump * fg_kappa(xshape)
            k2 = kappa**2

            keep_proxy = reach >= rr
            keep_truth = reach >= d

            k2_brute_all.append(float(k2.sum()))
            k2_proxy_all.append(float(k2[keep_proxy].sum()))
            k2_truth_all.append(float(k2[keep_truth].sum()))
            n_brute_all.append(float(len(mass)))
            n_proxy_all.append(float(np.count_nonzero(keep_proxy)))
            n_truth_all.append(float(np.count_nonzero(keep_truth)))

    summary = {
        "host_mass": host_mass,
        "z_lens": z_lens,
        "z_source": z_source,
        "subhalo_factor": subhalo_factor,
        "kappa_thr_host": kappa_thr_host,
        "kappa_thr_sub": kappa_thr_sub,
        "m_floor": m_floor,
        "r200_host": r200_host,
        "rmax_host": rmax_host,
        "nrays": nrays,
        "nseeds": len(seeds),
        "k2_brute_mean": float(np.mean(k2_brute_all)),
        "k2_proxy_mean": float(np.mean(k2_proxy_all)),
        "k2_truth_mean": float(np.mean(k2_truth_all)),
        "k2_brute_std": float(np.std(k2_brute_all, ddof=1)) if len(k2_brute_all) > 1 else 0.0,
        "k2_proxy_std": float(np.std(k2_proxy_all, ddof=1)) if len(k2_proxy_all) > 1 else 0.0,
        "k2_truth_std": float(np.std(k2_truth_all, ddof=1)) if len(k2_truth_all) > 1 else 0.0,
        "n_brute_mean": float(np.mean(n_brute_all)),
        "n_proxy_mean": float(np.mean(n_proxy_all)),
        "n_truth_mean": float(np.mean(n_truth_all)),
    }
    summary["proxy_vs_brute"] = summary["k2_proxy_mean"] / max(summary["k2_brute_mean"], 1.0e-300)
    summary["truth_vs_brute"] = summary["k2_truth_mean"] / max(summary["k2_brute_mean"], 1.0e-300)
    summary["proxy_vs_truth"] = summary["k2_proxy_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--nrays", type=int, default=128)
    parser.add_argument("--ray-seed", type=int, default=123)
    parser.add_argument(
        "--factors",
        type=float,
        nargs="+",
        default=[
            1e-5, 3.1622776601683795e-5, 1e-4, 3.1622776601683795e-4, 1e-3,
            3.1622776601683795e-3, 1e-2, 3.1622776601683795e-2, 1e-1,
            3.1622776601683795e-1, 1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0,
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "playground" / "subhalo_factor_dgate_area_scan.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "playground" / "subhalo_factor_dgate_area_scan.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for factor in args.factors:
        row = run_one_factor(
            host_mass=args.host_mass,
            z_lens=args.z_lens,
            z_source=args.z_source,
            subhalo_factor=float(factor),
            kappa_thr_host=args.kappa_thr_host,
            m_floor=args.m_floor,
            seeds=args.seeds,
            nrays=args.nrays,
            ray_seed=args.ray_seed,
        )
        rows.append(row)
        print(
            f"factor={factor:.4g} "
            f"k2_brute={row['k2_brute_mean']:.6e} "
            f"k2_proxy={row['k2_proxy_mean']:.6e} "
            f"k2_truth={row['k2_truth_mean']:.6e}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    factors = np.array([row["subhalo_factor"] for row in rows], dtype=float)
    k2_brute = np.array([row["k2_brute_mean"] for row in rows], dtype=float)
    k2_proxy = np.array([row["k2_proxy_mean"] for row in rows], dtype=float)
    k2_truth = np.array([row["k2_truth_mean"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(factors, k2_brute, marker="o", lw=2.0, color="0.45", label="brute")
    ax.plot(factors, k2_proxy, marker="o", lw=2.0, color="#2563eb", label="proxy-r")
    ax.plot(factors, k2_truth, marker="o", lw=2.0, color="#dc2626", label="truth-d")
    ax.set_xscale("log")
    ax.set_xlabel("subhalo_factor")
    ax.set_ylabel(r"area-weighted $\langle \sum \kappa_i^2 \rangle_{\rm rays}$")
    ax.set_title("Area-weighted single-host variance proxy")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.suptitle(
        f"M_host={args.host_mass:.1e} Msun, z_l={args.z_lens:g}, z_s={args.z_source:g}, "
        f"Nrays={args.nrays}, seeds={len(args.seeds)}"
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")
    print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
