"""
Prototype the corrected subhalo split using the actual clump-ray distance d.

This is a testing/debugging script only. It draws brute-force clump realizations
for one host and one ray, then compares three ways of keeping clumps:

1. brute: keep every clump above m_floor
2. proxy-r: production-style gate, keep clump iff reach(m) >= r_ray
3. truth-d: corrected gate, keep clump iff reach(m) >= d_ray_clump

The corrected gate is not yet a production algorithm because it decides after
placement, but it isolates the geometric issue cleanly on matched realizations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from scripts.subhalo_factor_proxy_check import (  # noqa: E402
    PSI_MAX,
    build_reach_interpolator,
    kappa_nfw,
    n_tau,
    nfw_params,
    gamma_norm,
    sample_brute_realization,
)


def summarize_realization(
    mass: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r_ray: float,
    reach: np.ndarray,
    z_lens: float,
    z_source: float,
) -> dict[str, float]:
    d = np.sqrt((x - r_ray) ** 2 + y**2)
    kappa = np.array([kappa_nfw(m, z_lens, z_source, max(di, 1.0e-9)) for m, di in zip(mass, d)], dtype=float)
    k2 = kappa**2

    keep_brute = np.ones(len(mass), dtype=bool)
    keep_proxy = reach >= r_ray
    keep_truth = reach >= d

    out = {
        "n_brute": float(np.count_nonzero(keep_brute)),
        "n_proxy": float(np.count_nonzero(keep_proxy)),
        "n_truth": float(np.count_nonzero(keep_truth)),
        "kappa_brute": float(kappa[keep_brute].sum()),
        "kappa_proxy": float(kappa[keep_proxy].sum()),
        "kappa_truth": float(kappa[keep_truth].sum()),
        "k2_brute": float(k2[keep_brute].sum()),
        "k2_proxy": float(k2[keep_proxy].sum()),
        "k2_truth": float(k2[keep_truth].sum()),
        "mass_brute": float(mass[keep_brute].sum()),
        "mass_proxy": float(mass[keep_proxy].sum()),
        "mass_truth": float(mass[keep_truth].sum()),
        "n_proxy_missed_truth": float(np.count_nonzero((~keep_proxy) & keep_truth)),
        "n_proxy_overtruth": float(np.count_nonzero(keep_proxy & (~keep_truth))),
        "k2_proxy_missed_truth": float(k2[(~keep_proxy) & keep_truth].sum()),
        "k2_proxy_overtruth": float(k2[keep_proxy & (~keep_truth)].sum()),
        "d_min": float(d.min()) if len(d) else np.nan,
        "d_median": float(np.median(d)) if len(d) else np.nan,
        "d_max": float(d.max()) if len(d) else np.nan,
    }
    return out


def run_case(
    host_mass: float,
    z_lens: float,
    z_source: float,
    r_ray: float,
    subhalo_factor: float,
    kappa_thr_host: float,
    m_floor: float,
    seeds: list[int],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rs_host, _, c_host, r200_host = nfw_params(host_mass, z_lens)
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

    rows: list[dict[str, float]] = []
    for seed in seeds:
        real = sample_brute_realization(seed, host_mass, z_lens, m_floor, gamma, r200_host, c_host)
        mass = np.asarray(real["mass"], dtype=float)
        x = np.asarray(real["x"], dtype=float)
        y = np.asarray(real["y"], dtype=float)
        if len(mass) == 0:
            rows.append({"seed": float(seed), "n_brute": 0.0, "n_proxy": 0.0, "n_truth": 0.0})
            continue
        reach = reach_interp(mass)
        row = summarize_realization(mass, x, y, r_ray, reach, z_lens, z_source)
        row["seed"] = float(seed)
        rows.append(row)

    summary: dict[str, float] = {
        "host_mass": host_mass,
        "z_lens": z_lens,
        "z_source": z_source,
        "r_ray": r_ray,
        "r200_host": r200_host,
        "rs_host": rs_host,
        "subhalo_factor": subhalo_factor,
        "kappa_thr_host": kappa_thr_host,
        "kappa_thr_sub": kappa_thr_sub,
        "m_floor": m_floor,
        "gamma": gamma,
        "fs": fs,
        "zf": zf,
    }
    valid = [row for row in rows if row.get("n_brute", 0.0) > 0.0]
    if valid:
        keys = [
            "n_brute", "n_proxy", "n_truth",
            "kappa_brute", "kappa_proxy", "kappa_truth",
            "k2_brute", "k2_proxy", "k2_truth",
            "mass_brute", "mass_proxy", "mass_truth",
            "n_proxy_missed_truth", "n_proxy_overtruth",
            "k2_proxy_missed_truth", "k2_proxy_overtruth",
        ]
        for key in keys:
            summary[f"{key}_mean"] = float(np.mean([row[key] for row in valid]))
        summary["proxy_k2_vs_truth"] = summary["k2_proxy_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
        summary["proxy_k2_vs_brute"] = summary["k2_proxy_mean"] / max(summary["k2_brute_mean"], 1.0e-300)
        summary["truth_k2_vs_brute"] = summary["k2_truth_mean"] / max(summary["k2_brute_mean"], 1.0e-300)
        summary["proxy_missed_truth_k2_frac"] = (
            summary["k2_proxy_missed_truth_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
        )
        summary["proxy_overtruth_k2_frac"] = (
            summary["k2_proxy_overtruth_mean"] / max(summary["k2_truth_mean"], 1.0e-300)
        )
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-mass", type=float, default=1.0e13)
    parser.add_argument("--z-lens", type=float, default=0.5)
    parser.add_argument("--z-source", type=float, default=1.0)
    parser.add_argument("--r-ray", type=float, default=1000.0)
    parser.add_argument("--subhalo-factor", type=float, default=1.0e-4)
    parser.add_argument("--kappa-thr-host", type=float, default=1.276e-4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, rows = run_case(
        host_mass=args.host_mass,
        z_lens=args.z_lens,
        z_source=args.z_source,
        r_ray=args.r_ray,
        subhalo_factor=args.subhalo_factor,
        kappa_thr_host=args.kappa_thr_host,
        m_floor=args.m_floor,
        seeds=args.seeds,
    )

    print(
        f"host M={summary['host_mass']:.3e}  z_l={summary['z_lens']:g}  z_s={summary['z_source']:g}  "
        f"r_ray={summary['r_ray']:.1f} kpc"
    )
    print(
        f"factor={summary['subhalo_factor']:.3e}  kappa_thr_sub={summary['kappa_thr_sub']:.3e}  "
        f"r200={summary['r200_host']:.1f} kpc"
    )
    if "k2_brute_mean" in summary:
        print("\nmean over seeds:")
        print(
            f"  clumps kept: brute={summary['n_brute_mean']:.1f}  "
            f"proxy-r={summary['n_proxy_mean']:.1f}  truth-d={summary['n_truth_mean']:.1f}"
        )
        print(
            f"  kappa^2 retained / brute: proxy-r={summary['proxy_k2_vs_brute']:.6f}  "
            f"truth-d={summary['truth_k2_vs_brute']:.6f}"
        )
        print(
            f"  proxy-r / truth-d kappa^2 = {summary['proxy_k2_vs_truth']:.6f}"
        )
        print(
            f"  proxy misses {100.0 * summary['proxy_missed_truth_k2_frac']:.4f}% of truth-d kappa^2"
        )
        print(
            f"  proxy over-keeps {100.0 * summary['proxy_overtruth_k2_frac']:.4f}% of truth-d kappa^2"
        )

    print("\nper-seed:")
    for row in rows:
        if row.get("n_brute", 0.0) == 0.0:
            print(f"  seed {int(row['seed'])}: no clumps")
            continue
        print(
            f"  seed {int(row['seed'])}: "
            f"n(brute/proxy/truth)=({int(row['n_brute'])}/{int(row['n_proxy'])}/{int(row['n_truth'])})  "
            f"k2(proxy/truth)={row['k2_proxy'] / max(row['k2_truth'], 1.0e-300):.6f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "rows": rows}
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nsaved {args.json_out}")


if __name__ == "__main__":
    main()
