#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import gwlensing as gw  # noqa: E402


def mac_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return ""


def run_sampler(args, subhalo: bool) -> dict:
    t0 = time.perf_counter()
    lnmu = gw.sample_lnmu_ml(
        args.z,
        args.h,
        args.omega_m,
        args.sigma8,
        args.nreal,
        args.seed,
        False,
        args.mmin,
        subhalo,
        args.m_floor,
    )
    dt = time.perf_counter() - t0
    lnmu = np.asarray(lnmu, dtype=float)
    lnmu = lnmu[np.isfinite(lnmu)]
    mu = np.exp(lnmu)
    mu = mu[np.isfinite(mu) & (mu > 0.0)]
    return {
        "subhalo": subhalo,
        "seconds": dt,
        "valid_samples": int(mu.size),
        "lnmu": lnmu,
        "mu": mu,
        "mean_mu": float(np.mean(mu)) if mu.size else float("nan"),
        "std_mu": float(np.std(mu)) if mu.size else float("nan"),
    }


def histogram_pdf(mu: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(mu, bins=edges)
    widths = np.diff(edges)
    total = counts.sum()
    pdf = counts / (total * widths) if total > 0 else np.zeros_like(widths)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dP/dmu runtime with and without subhalos.")
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--omega-m", type=float, default=0.315)
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--nreal", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmin", type=float, default=1.0e7)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--outdir", type=Path, default=ROOT / "plots" / "figures" / "subhalo_runtime")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    results = [run_sampler(args, False), run_sampler(args, True)]

    all_mu = np.concatenate([r["mu"] for r in results if r["mu"].size])
    lo, hi = np.quantile(all_mu, [0.001, 0.999])
    lo = max(0.05, float(lo))
    hi = max(float(hi), lo * 1.01)
    edges = np.linspace(lo, hi, args.bins + 1)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    colors = {False: "#2ca02c", True: "#d62728"}
    labels = {False: "subhalo = false", True: "subhalo = true"}
    for r in results:
        centers, pdf = histogram_pdf(r["mu"], edges)
        ax.step(
            centers,
            pdf,
            where="mid",
            lw=2.0,
            color=colors[r["subhalo"]],
            label=f"{labels[r['subhalo']]} ({r['seconds']:.2f} s)",
        )

    ax.set_xlabel(r"Magnification $\mu$")
    ax.set_ylabel(r"$dP/d\mu$")
    ax.set_title(
        rf"Subhalo PDF runtime comparison: z={args.z}, "
        rf"$\Omega_M$={args.omega_m}, $\sigma_8$={args.sigma8}, h={args.h}"
    )
    ax.set_xlim(lo, hi)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = args.outdir / "subhalo_false_vs_true_dpdmu.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    summary = {
        "parameters": {
            "z": args.z,
            "h": args.h,
            "OmegaM": args.omega_m,
            "sigma8": args.sigma8,
            "Nreal": args.nreal,
            "seed": args.seed,
            "Mmin": args.mmin,
            "m_floor": args.m_floor,
            "bins": args.bins,
            "mu_plot_range_quantiles": [0.001, 0.999],
        },
        "machine": {
            "platform": platform.platform(),
            "processor": mac_sysctl("machdep.cpu.brand_string") or platform.processor(),
            "logical_cpus": mac_sysctl("hw.ncpu"),
            "performance_cores": mac_sysctl("hw.perflevel0.physicalcpu"),
            "efficiency_cores": mac_sysctl("hw.perflevel1.physicalcpu"),
        },
        "paper_reference": {
            "source": "Vaskonen 2026, Sec. 3",
            "reported": "single likelihood evaluation, including halo/filament mass functions and weak-lensing distributions at multiple source redshifts, approximately 20 seconds on Apple M1",
        },
        "results": [
            {
                "subhalo": r["subhalo"],
                "seconds": r["seconds"],
                "valid_samples": r["valid_samples"],
                "mean_mu": r["mean_mu"],
                "std_mu": r["std_mu"],
            }
            for r in results
        ],
        "speed_ratio_subhalo_true_over_false": results[1]["seconds"] / results[0]["seconds"],
        "plot": str(plot_path),
    }

    json_path = args.outdir / "subhalo_runtime_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
