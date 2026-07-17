"""
Area-weighted paired-variance scan using the paper's observable:

    sigma_sub^2(kappa) = Var(kappa) - Var(kappa_nosub)

This is still a single-host playground diagnostic, not the full line-of-sight
production sampler, but the y-axis now matches the paper quantity directly.

For one host mass/redshift and each subhalo_factor, the script samples many ray
impact parameters inside the host kappathr disk and evaluates three matched cases:

1. brute    : clumps above m_floor, smooth host reduced by resolved mass fraction
2. proxy-r  : production-style gate based on host-center distance r
3. truth-d  : corrected gate based on actual clump-ray distance d

For each case:
  kappa       = kappa_host_eff + sum(resolved clump kappas)
  kappa_nosub = kappa_host_full

Then the script reports
  Var(kappa) - Var(kappa_nosub)
across the ray ensemble.
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
    PSI_MAX,
    build_reach_interpolator,
    fg_kappa,
    gamma_norm,
    n_tau,
    nfw_params,
    sample_brute_realization,
    sigma_crit,
)
from playground.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402


def kappa_host_from_mass(host_mass: float, z_lens: float, z_source: float, r: np.ndarray) -> np.ndarray:
    rs, rhos, _, _ = nfw_params(host_mass, z_lens)
    sigmac = sigma_crit(z_source, z_lens)
    kappa0 = rs * rhos / sigmac
    x = np.maximum(r / rs, 1.0e-12)
    return 2.0 * kappa0 * fg_kappa(x)

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
        default=ROOT / "playground" / "subhalo_factor_paired_area_scan.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "playground" / "subhalo_factor_paired_area_scan.json",
    )
    return parser.parse_args()


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
    _, _, c_host, r200_host = nfw_params(host_mass, z_lens)
    rmax_host = host_rmax(host_mass, z_lens, z_source, kappa_thr_host)
    nt, zf = n_tau(host_mass, z_lens)
    fs = 0.3563 / nt**0.6 - 0.075
    gamma = gamma_norm(fs)
    kappa_thr_sub = subhalo_factor * kappa_thr_host
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
    kappa_nosub = kappa_host_from_mass(host_mass, z_lens, z_source, ray_r)

    all_brute = []
    all_proxy = []
    all_truth = []

    for seed in seeds:
        real = sample_brute_realization(seed, host_mass, z_lens, m_floor, gamma, r200_host, c_host)
        mass = np.asarray(real["mass"], dtype=float)
        x = np.asarray(real["x"], dtype=float)
        y = np.asarray(real["y"], dtype=float)
        if len(mass) == 0:
            all_brute.append(kappa_nosub.copy())
            all_proxy.append(kappa_nosub.copy())
            all_truth.append(kappa_nosub.copy())
            continue

        reach = reach_interp(mass)
        rs_clump = np.empty_like(mass)
        kappa0_clump = np.empty_like(mass)
        sigmac = sigma_crit(z_source, z_lens)
        for i, m in enumerate(mass):
            rs_i, rhos_i, _, _ = nfw_params(float(m), z_lens)
            rs_clump[i] = rs_i
            kappa0_clump[i] = rs_i * rhos_i / sigmac

        kappa_brute = np.empty_like(ray_r)
        kappa_proxy = np.empty_like(ray_r)
        kappa_truth = np.empty_like(ray_r)

        for j, (xr, yr, rr, phir) in enumerate(zip(ray_x, ray_y, ray_r, ray_phi)):
            cosp = np.cos(phir)
            sinp = np.sin(phir)
            x_aligned = x * cosp + y * sinp
            y_aligned = -x * sinp + y * cosp
            d = np.sqrt((x_aligned - rr) ** 2 + y_aligned**2)
            kappa_clump = 2.0 * kappa0_clump * fg_kappa(np.maximum(d / rs_clump, 1.0e-12))

            keep_proxy = reach >= rr
            keep_truth = reach >= d

            frac_brute = min(float(mass.sum() / host_mass), 0.95)
            frac_proxy = min(float(mass[keep_proxy].sum() / host_mass), 0.95)
            frac_truth = min(float(mass[keep_truth].sum() / host_mass), 0.95)

            kappa_host_brute = kappa_host_from_mass((1.0 - frac_brute) * host_mass, z_lens, z_source, np.array([rr]))[0]
            kappa_host_proxy = kappa_host_from_mass((1.0 - frac_proxy) * host_mass, z_lens, z_source, np.array([rr]))[0]
            kappa_host_truth = kappa_host_from_mass((1.0 - frac_truth) * host_mass, z_lens, z_source, np.array([rr]))[0]

            kappa_brute[j] = kappa_host_brute + float(kappa_clump.sum())
            kappa_proxy[j] = kappa_host_proxy + float(kappa_clump[keep_proxy].sum())
            kappa_truth[j] = kappa_host_truth + float(kappa_clump[keep_truth].sum())

        all_brute.append(kappa_brute)
        all_proxy.append(kappa_proxy)
        all_truth.append(kappa_truth)

    kappa_brute_all = np.concatenate(all_brute)
    kappa_proxy_all = np.concatenate(all_proxy)
    kappa_truth_all = np.concatenate(all_truth)
    kappa_nosub_all = np.tile(kappa_nosub, len(seeds))

    out = {
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
        "excess_brute": float(np.var(kappa_brute_all) - np.var(kappa_nosub_all)),
        "excess_proxy": float(np.var(kappa_proxy_all) - np.var(kappa_nosub_all)),
        "excess_truth": float(np.var(kappa_truth_all) - np.var(kappa_nosub_all)),
    }
    return out


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
            f"excess_brute={row['excess_brute']:.6e} "
            f"excess_proxy={row['excess_proxy']:.6e} "
            f"excess_truth={row['excess_truth']:.6e}"
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2))

    factors = np.array([row["subhalo_factor"] for row in rows], dtype=float)
    excess_brute = np.array([row["excess_brute"] for row in rows], dtype=float)
    excess_proxy = np.array([row["excess_proxy"] for row in rows], dtype=float)
    excess_truth = np.array([row["excess_truth"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(factors, excess_brute, marker="o", lw=2.0, color="0.45", label="brute")
    ax.plot(factors, excess_proxy, marker="o", lw=2.0, color="#2563eb", label="proxy-r")
    ax.plot(factors, excess_truth, marker="o", lw=2.0, color="#dc2626", label="truth-d")
    ax.set_xscale("log")
    ax.set_xlabel("subhalo_factor")
    ax.set_ylabel(r"$\mathrm{Var}(\kappa)-\mathrm{Var}(\kappa_{\rm nosub})$")
    ax.set_title("Area-weighted paired substructure variance")
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
