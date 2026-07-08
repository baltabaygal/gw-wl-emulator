"""
Multi-seed calibration of the subhalo_factor dynamic split against brute force.

This is the noise-reduced version of scripts/subhalo_factor_convergence.py.  It runs
the paired estimator

    Var(kappa) - Var(kappa_nosub)

for brute-force subhalo sampling and for a small set of dynamic subhalo_factor values,
then combines independent seeds.  The brute run is expensive, so keep the factor grid
focused around the suspected plateau.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT / "build"))
import gwlensing  # noqa: E402


DEFAULT_FACTORS = np.array([1.0e-5, 3.1622776601683795e-5, 1.0e-4, 3.1622776601683795e-4])
DEFAULT_SEEDS = np.array([42, 314, 2718, 16180])


def paired_excess(kappa: np.ndarray, kappa_nosub: np.ndarray) -> float:
    return float(kappa.var() - kappa_nosub.var())


def bootstrap_err(
    kappa: np.ndarray,
    kappa_nosub: np.ndarray,
    nboot: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kappa), size=(nboot, len(kappa)))
    boot = kappa[idx].var(axis=1) - kappa_nosub[idx].var(axis=1)
    return float(boot.std(ddof=1))


def run_job(args: tuple[str, int, dict, int]) -> dict:
    label, seed, kwargs, nboot = args
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(**kwargs, seed=int(seed))
    runtime = time.time() - t0

    kappa = np.asarray(res["kappa"], dtype=float)
    kappa_nosub = np.asarray(res["kappa_nosub"], dtype=float)
    if not np.isfinite(kappa).all() or not np.isfinite(kappa_nosub).all():
        raise RuntimeError(f"non-finite raw sample for label={label}, seed={seed}")

    return {
        "label": label,
        "seed": int(seed),
        "excess": paired_excess(kappa, kappa_nosub),
        "err": bootstrap_err(kappa, kappa_nosub, nboot, int(seed) + 10_000),
        "runtime": runtime,
    }


def combine_rows(rows: list[dict]) -> list[dict]:
    labels = sorted({row["label"] for row in rows}, key=lambda x: (x != "brute", x))
    summary = []
    for label in labels:
        group = [row for row in rows if row["label"] == label]
        excess = np.array([row["excess"] for row in group], dtype=float)
        err = np.array([max(row["err"], 1.0e-300) for row in group], dtype=float)
        weights = 1.0 / err**2
        weighted = float(np.sum(weights * excess) / np.sum(weights))
        weighted_err = float(np.sqrt(1.0 / np.sum(weights)))
        seed_scatter = float(excess.std(ddof=1)) if len(excess) > 1 else 0.0
        mean = float(excess.mean())
        mean_err = float(seed_scatter / np.sqrt(len(excess))) if len(excess) > 1 else float(err[0])
        summary.append(
            {
                "label": label,
                "n_seeds": len(group),
                "excess_weighted": weighted,
                "err_weighted": weighted_err,
                "excess_mean": mean,
                "err_seed_mean": mean_err,
                "seed_scatter": seed_scatter,
                "runtime_total": float(sum(row["runtime"] for row in group)),
            }
        )
    return summary


def paired_factor_deltas(rows: list[dict]) -> list[dict]:
    seeds = sorted({row["seed"] for row in rows})
    labels = sorted({row["label"] for row in rows if row["label"] != "brute"})
    by_key = {(row["label"], row["seed"]): row for row in rows}
    deltas = []
    for label in labels:
        diff = []
        pct = []
        for seed in seeds:
            brute = by_key.get(("brute", seed))
            factor = by_key.get((label, seed))
            if brute is None or factor is None:
                continue
            d = factor["excess"] - brute["excess"]
            diff.append(d)
            pct.append(100.0 * d / brute["excess"])
        diff_arr = np.array(diff, dtype=float)
        pct_arr = np.array(pct, dtype=float)
        n = len(diff_arr)
        deltas.append(
            {
                "label": label,
                "n_seeds": n,
                "diff_mean": float(diff_arr.mean()) if n else np.nan,
                "diff_sem": float(diff_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "pct_mean": float(pct_arr.mean()) if n else np.nan,
                "pct_sem": float(pct_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            }
        )
    return deltas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zs", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS.tolist())
    parser.add_argument("--factors", type=float, nargs="+", default=DEFAULT_FACTORS.tolist())
    parser.add_argument("--nboot", type=int, default=120)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or ROOT / "data" / f"subhalo_factor_brute_multiseed_z{args.zs:g}_N{args.n}.npz"

    common = dict(
        z=args.zs,
        h=0.674,
        OmegaM=0.315,
        sigma8=0.811,
        nsamples=args.n,
        filaments=False,
        bias=False,
        ell=False,
        Nhalos=100,
        subhalo=True,
        m_floor=float(args.m_floor),
        subhalo_threads=4,
        subhalo_parallel_threshold=1000,
        subhalo_model=1,
        custom_kappathr=-1.0,
        Mmin=1.0e7,
    )

    jobs = []
    for seed in args.seeds:
        jobs.append(("brute", int(seed), dict(common, subhalo_brute=True), args.nboot))
        for factor in args.factors:
            label = f"{float(factor):.8g}"
            jobs.append(
                (
                    label,
                    int(seed),
                    dict(common, subhalo_brute=False, subhalo_factor=float(factor)),
                    args.nboot,
                )
            )

    rows = []
    print(
        f"running zs={args.zs:g}, N={args.n:,}, seeds={list(args.seeds)}, "
        f"factors={[float(f) for f in args.factors]}, jobs={len(jobs)}",
        flush=True,
    )
    with Pool(processes=args.processes) as pool:
        for row in pool.imap_unordered(run_job, jobs, chunksize=1):
            rows.append(row)
            print(
                f"done {row['label']:>10} seed={row['seed']:>6} "
                f"excess={row['excess']:.4e} err={row['err']:.2e} "
                f"time={row['runtime']:.1f}s",
                flush=True,
            )

    summary = combine_rows(rows)
    deltas = paired_factor_deltas(rows)
    brute = next(row for row in summary if row["label"] == "brute")

    print("\ncombined summary:")
    print(f"{'label':>10} {'excess':>12} {'err':>10} {'vs brute':>10} {'runtime':>10}")
    for row in summary:
        dev = row["excess_weighted"] - brute["excess_weighted"]
        pct = 0.0 if row["label"] == "brute" else 100.0 * dev / brute["excess_weighted"]
        print(
            f"{row['label']:>10} {row['excess_weighted']:12.4e} "
            f"{row['err_weighted']:10.2e} {pct:+9.2f}% "
            f"{row['runtime_total']:9.1f}s"
        )

    print("\npaired factor-minus-brute differences by seed:")
    print(f"{'label':>10} {'mean diff':>12} {'sem':>10} {'mean pct':>10} {'pct sem':>10}")
    for row in deltas:
        print(
            f"{row['label']:>10} {row['diff_mean']:12.4e} "
            f"{row['diff_sem']:10.2e} {row['pct_mean']:+9.2f}% "
            f"{row['pct_sem']:9.2f}%"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    row_dtype = [
        ("label", "U32"),
        ("seed", "i8"),
        ("excess", "f8"),
        ("err", "f8"),
        ("runtime", "f8"),
    ]
    summary_dtype = [
        ("label", "U32"),
        ("n_seeds", "i8"),
        ("excess_weighted", "f8"),
        ("err_weighted", "f8"),
        ("excess_mean", "f8"),
        ("err_seed_mean", "f8"),
        ("seed_scatter", "f8"),
        ("runtime_total", "f8"),
    ]
    delta_dtype = [
        ("label", "U32"),
        ("n_seeds", "i8"),
        ("diff_mean", "f8"),
        ("diff_sem", "f8"),
        ("pct_mean", "f8"),
        ("pct_sem", "f8"),
    ]
    row_arr = np.array([tuple(row[k] for k, _ in row_dtype) for row in rows], dtype=row_dtype)
    summary_arr = np.array(
        [tuple(row[k] for k, _ in summary_dtype) for row in summary],
        dtype=summary_dtype,
    )
    delta_arr = np.array(
        [tuple(row[k] for k, _ in delta_dtype) for row in deltas],
        dtype=delta_dtype,
    )
    np.savez(
        out,
        rows=row_arr,
        summary=summary_arr,
        paired_deltas=delta_arr,
        factors=np.array(args.factors, dtype=float),
        seeds=np.array(args.seeds, dtype=int),
        zs=args.zs,
        N=args.n,
        nboot=args.nboot,
        m_floor=args.m_floor,
        config_json=json.dumps(common, sort_keys=True),
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
