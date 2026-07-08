"""
Production redshift check for the subhalo_factor resolution split.

For each source redshift and factor, measure the paired substructure variance
excess:
  Var(kappa) - Var(kappa_nosub)

The lowest factor in the grid is treated as the floor-limited reference.  The
script is resumable: completed rows are written to CSV as each job finishes.
"""
from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT / "build"))
import gwlensing

DEFAULT_ZS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
DEFAULT_FACTORS = [1.0e-5, 3.162e-5, 1.0e-4, 3.162e-4, 1.0e-3, 3.162e-3, 1.0e-2]
DEFAULT_FACTOR = 1.0e-5
DEFAULT_N = 40_000
DEFAULT_NBOOT = 200
DEFAULT_SEED = 42
CSV_PATH = ROOT / "data" / "subhalo_factor_redshift_check.csv"
NPZ_PATH = ROOT / "data" / "subhalo_factor_redshift_check.npz"
PLOT_PATH = ROOT / "plots" / "figures" / "subhalo_factor_redshift_check.png"


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def paired_excess(kappa: np.ndarray, kappa_nosub: np.ndarray) -> float:
    return float(kappa.var() - kappa_nosub.var())


def bootstrap_err(kappa: np.ndarray, kappa_nosub: np.ndarray, nboot: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(kappa), size=(nboot, len(kappa)))
    boot = kappa[idx].var(axis=1) - kappa_nosub[idx].var(axis=1)
    return float(boot.std())


def row_key(zs: float, factor: float, n: int, seed: int) -> tuple[str, str, int, int]:
    return (f"{zs:.8g}", f"{factor:.8g}", n, seed)


def load_completed(csv_path: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {
            row_key(float(row["zs"]), float(row["factor"]), int(row["N"]), int(row["seed"])): row
            for row in reader
        }


def append_row(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    fields = ["zs", "factor", "N", "seed", "excess", "err", "runtime"]
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_job(args: tuple[float, float, int, int, int, int]) -> dict[str, float | int]:
    zs, factor, n, seed, nboot, subhalo_threads = args
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(
        z=float(zs),
        h=0.674,
        OmegaM=0.315,
        sigma8=0.811,
        nsamples=int(n),
        seed=int(seed),
        filaments=False,
        bias=False,
        ell=False,
        Nhalos=100,
        subhalo=True,
        m_floor=1.0e7,
        subhalo_threads=int(subhalo_threads),
        subhalo_parallel_threshold=1000,
        subhalo_model=1,
        subhalo_brute=False,
        subhalo_factor=float(factor),
        custom_kappathr=-1.0,
        Mmin=1.0e7,
    )
    kappa = np.asarray(res["kappa"], dtype=float)
    kappa_nosub = np.asarray(res["kappa_nosub"], dtype=float)
    if not np.isfinite(kappa).all() or not np.isfinite(kappa_nosub).all():
        raise RuntimeError(f"non-finite sample at z={zs:g}, factor={factor:g}")
    excess = paired_excess(kappa, kappa_nosub)
    err = bootstrap_err(kappa, kappa_nosub, int(nboot), int(seed) + 17)
    return {
        "zs": float(zs),
        "factor": float(factor),
        "N": int(n),
        "seed": int(seed),
        "excess": excess,
        "err": err,
        "runtime": time.time() - t0,
    }


def rows_from_csv(csv_path: Path) -> np.ndarray:
    completed = load_completed(csv_path)
    rows = []
    for row in completed.values():
        rows.append([
            float(row["zs"]),
            float(row["factor"]),
            int(row["N"]),
            int(row["seed"]),
            float(row["excess"]),
            float(row["err"]),
            float(row["runtime"]),
        ])
    if not rows:
        return np.empty((0, 7), dtype=float)
    rows = np.array(rows, dtype=float)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def save_npz_and_plot(csv_path: Path, npz_path: Path, plot_path: Path, default_factor: float) -> None:
    rows = rows_from_csv(csv_path)
    if rows.size == 0:
        return
    zs_values = np.unique(rows[:, 0])
    factors = np.unique(rows[:, 1])
    reference_factor = factors.min()

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        rows=rows,
        zs_values=zs_values,
        factors=factors,
        reference_factor=reference_factor,
        default_factor=default_factor,
        columns=np.array(["zs", "factor", "N", "seed", "excess", "err", "runtime"]),
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(zs_values) - 1)) for i in range(len(zs_values))]
    for color, zs in zip(colors, zs_values):
        sub = rows[rows[:, 0] == zs]
        order = np.argsort(sub[:, 1])
        sub = sub[order]
        f = sub[:, 1]
        ex = sub[:, 4]
        err = sub[:, 5]
        ref = ex[np.argmin(np.abs(f - reference_factor))]
        axes[0].errorbar(f, ex, yerr=err, marker="o", lw=2.0, capsize=3, color=color, label=f"z_s={zs:g}")
        axes[1].errorbar(f, ex / ref, yerr=err / abs(ref), marker="o", lw=2.0, capsize=3, color=color, label=f"z_s={zs:g}")

    for ax in axes:
        ax.axvline(default_factor, color="#dc2626", ls=":", lw=2.0, label="default" if ax is axes[0] else None)
        ax.set_xscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("subhalo_factor")
    axes[0].set_ylabel("Var(kappa) - Var(kappa_nosub)")
    axes[0].set_title("Paired excess by redshift")
    axes[1].axhline(1.0, color="0.35", ls="--", lw=1.5)
    axes[1].set_ylabel(f"excess / excess({reference_factor:g})")
    axes[1].set_title("Relative to low-factor reference")
    axes[0].legend(frameon=True, fontsize=9)
    n_values = sorted(set(rows[:, 2].astype(int)))
    n_label = ",".join(f"{n:,}" for n in n_values)
    fig.suptitle(f"Subhalo-factor redshift check, N={n_label}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=230, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zs", default=",".join(str(v) for v in DEFAULT_ZS))
    parser.add_argument("--factors", default=",".join(str(v) for v in DEFAULT_FACTORS))
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nboot", type=int, default=DEFAULT_NBOOT)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--subhalo-threads", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--npz", type=Path, default=NPZ_PATH)
    parser.add_argument("--plot", type=Path, default=PLOT_PATH)
    parser.add_argument("--default-factor", type=float, default=DEFAULT_FACTOR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zs_values = parse_float_list(args.zs)
    factors = parse_float_list(args.factors)
    if args.overwrite and args.csv.exists():
        args.csv.unlink()

    completed = load_completed(args.csv)
    jobs = []
    for zs in zs_values:
        for factor in factors:
            key = row_key(zs, factor, args.n, args.seed)
            if key not in completed:
                jobs.append((zs, factor, args.n, args.seed, args.nboot, args.subhalo_threads))

    print(f"completed={len(completed)} pending={len(jobs)} csv={args.csv}", flush=True)
    if jobs:
        # Low factors are slower; sort so expensive jobs start first.
        jobs.sort(key=lambda x: (x[1], x[0]))
        with Pool(processes=args.processes) as pool:
            for row in pool.imap_unordered(run_job, jobs, chunksize=1):
                append_row(args.csv, row)
                print(
                    f"z={row['zs']:g} factor={row['factor']:.4g} "
                    f"excess={row['excess']:.4e} err={row['err']:.2e} "
                    f"time={row['runtime']:.1f}s",
                    flush=True,
                )

    save_npz_and_plot(args.csv, args.npz, args.plot, args.default_factor)
    print(f"saved {args.npz}")
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
