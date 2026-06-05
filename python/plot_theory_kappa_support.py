import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_gwlensing():
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))
    import gwlensing
    return gwlensing


def main():
    parser = argparse.ArgumentParser(
        description="Plot the theory-side weighted kappa support to diagnose near-threshold structure."
    )
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--n-u", dest="n_u", type=int, default=512)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--out-prefix", default="theory_kappa_support")
    args = parser.parse_args()

    gw = load_gwlensing()
    support = gw.theory_kappa_support(
        z=args.z,
        OmegaM=args.OmegaM,
        sigma8=args.sigma8,
        h=args.h,
        Nreal=args.Nreal,
        seed=args.seed,
        filaments=False,
        bias=False,
        ell=False,
        exact_poisson=True,
        Nhalos=args.Nhalos,
        n_u=args.n_u,
    )

    kappa = np.asarray(support["kappa"], dtype=float)
    weight = np.asarray(support["weight"], dtype=float)
    kthr = float(support["kappa_threshold"])

    kmax = float(np.max(kappa))
    bins = np.geomspace(max(kthr * 1.000001, 1e-12), kmax, args.bins)
    hist, edges = np.histogram(kappa, bins=bins, weights=weight, density=False)
    width = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = hist / width

    excess = np.clip(centers - kthr, 1e-16, None)
    valid = np.isfinite(density) & (density > 0) & np.isfinite(excess) & (excess > 0)

    def fit_window(mult_hi):
        lo = kthr * 1.0
        hi = kthr * mult_hi
        m = valid & (centers >= lo) & (centers <= hi)
        if np.count_nonzero(m) < 3:
            return None
        x = np.log(centers[m] - kthr)
        y = np.log(density[m])
        slope, intercept = np.polyfit(x, y, 1)
        return {
            "kappa_min": float(lo),
            "kappa_max": float(hi),
            "n_points": int(np.count_nonzero(m)),
            "alpha": float(-slope),
            "log_amplitude": float(intercept),
        }

    fit_windows = {}
    for mult_hi in [2.0, 1.5, 1.2, 1.1]:
        key = f"1.0_to_{mult_hi:.1f}".replace(".", "p")
        fit_windows[key] = fit_window(mult_hi)

    finest = fit_windows["1p0_to_1p1"]
    slope = -finest["alpha"] if finest is not None else np.nan

    summary = {
        "z": args.z,
        "Nhalos": args.Nhalos,
        "n_u": args.n_u,
        "kappa_threshold": kthr,
        "kappa_max": kmax,
        "total_weight": float(np.sum(weight)),
        "near_threshold_loglog_slope": float(slope),
        "window_fits": fit_windows,
    }

    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.0))
    ax[0].plot(centers, density, lw=1.8)
    ax[0].axvline(kthr, color="k", ls="--", lw=0.9)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("kappa")
    ax[0].set_ylabel("weighted density")
    ax[0].set_title("Theory-side kappa support")

    order = np.argsort(excess[valid])
    x_sorted = excess[valid][order]
    y_sorted = density[valid][order]
    ax[1].plot(x_sorted, y_sorted, lw=1.8, label="weighted density")
    colors = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
    for color, (name, fit) in zip(colors, fit_windows.items()):
        if fit is None:
            continue
        x0 = np.geomspace(
            max(fit["kappa_min"] - kthr, 1e-16),
            fit["kappa_max"] - kthr,
            60,
        )
        y0 = np.exp(fit["log_amplitude"]) * x0 ** (-fit["alpha"])
        ax[1].plot(x0, y0, ls="--", lw=1.2, color=color, label=f"{name}: alpha={fit['alpha']:.3f}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("kappa - kappa_threshold")
    ax[1].set_ylabel("weighted density")
    ax[1].set_title("Near-threshold structure")
    ax[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=170)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {args.out_prefix}.png {args.out_prefix}_summary.json")


if __name__ == "__main__":
    main()
