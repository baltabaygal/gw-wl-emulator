import argparse
import csv
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


def sample_xi(gw, *, z, sigma8, omega_m, h, nreal, seed, nhalos):
    events = gw.sample_lensing_events(
        z=z,
        OmegaM=omega_m,
        sigma8=sigma8,
        h=h,
        Nreal=nreal,
        seed=seed,
        filaments=False,
        bias=False,
        ell=False,
        exact_poisson=True,
        Nhalos=nhalos,
    )
    kappa_evt = np.asarray(events["kappa"], dtype=float)
    gamma1_evt = np.asarray(events["gamma1"], dtype=float)
    gamma2_evt = np.asarray(events["gamma2"], dtype=float)
    gamma_evt = np.sqrt(gamma1_evt ** 2 + gamma2_evt ** 2)
    mu_evt = 1.0 / ((1.0 - kappa_evt) ** 2 - gamma_evt ** 2)
    good = np.isfinite(mu_evt) & (mu_evt > 0.0)
    return -np.log((1.0 - kappa_evt[good]) ** 2 - gamma_evt[good] ** 2)


def hill_estimator(sample, threshold):
    tail = np.asarray(sample[sample >= threshold], dtype=float)
    if tail.size < 30:
        return None
    denom = np.sum(np.log(tail / threshold))
    if denom <= 0.0:
        return None
    alpha = tail.size / denom
    return {"alpha": float(alpha), "n_samples": int(tail.size), "threshold": float(threshold)}


def pareto_mle(sample, threshold):
    # Same closed-form estimator as Hill for a pure Pareto tail, kept separate for clarity in outputs.
    fit = hill_estimator(sample, threshold)
    if fit is None:
        return None
    return {"alpha": fit["alpha"], "n_samples": fit["n_samples"], "threshold": fit["threshold"]}


def threshold_scan(sample, thresholds):
    out = []
    for thr in thresholds:
        hill = hill_estimator(sample, thr)
        mle = pareto_mle(sample, thr)
        out.append(
            {
                "xi_min": float(thr),
                "alpha_hill": None if hill is None else hill["alpha"],
                "alpha_pareto": None if mle is None else mle["alpha"],
                "tail_event_count": 0 if hill is None else hill["n_samples"],
            }
        )
    return out


def summarize_plateau(scan_rows, alpha_key, diff_tol, min_run):
    valid = [row for row in scan_rows if row[alpha_key] is not None]
    if len(valid) < min_run:
        return None

    best = None
    current_start = 0
    values = [row[alpha_key] for row in valid]
    for i in range(1, len(valid)):
        if abs(values[i] - values[i - 1]) > diff_tol:
            if i - current_start >= min_run:
                segment = valid[current_start:i]
                width = segment[-1]["xi_min"] / segment[0]["xi_min"]
                cand = {
                    "xi_min_lo": float(segment[0]["xi_min"]),
                    "xi_min_hi": float(segment[-1]["xi_min"]),
                    "n_points": len(segment),
                    "alpha_mean": float(np.mean([row[alpha_key] for row in segment])),
                    "alpha_std": float(np.std([row[alpha_key] for row in segment], ddof=1)) if len(segment) > 1 else 0.0,
                    "width_ratio": float(width),
                }
                if best is None or cand["width_ratio"] > best["width_ratio"]:
                    best = cand
            current_start = i

    if len(valid) - current_start >= min_run:
        segment = valid[current_start:]
        width = segment[-1]["xi_min"] / segment[0]["xi_min"]
        cand = {
            "xi_min_lo": float(segment[0]["xi_min"]),
            "xi_min_hi": float(segment[-1]["xi_min"]),
            "n_points": len(segment),
            "alpha_mean": float(np.mean([row[alpha_key] for row in segment])),
            "alpha_std": float(np.std([row[alpha_key] for row in segment], ddof=1)) if len(segment) > 1 else 0.0,
            "width_ratio": float(width),
        }
        if best is None or cand["width_ratio"] > best["width_ratio"]:
            best = cand

    return best


def plot_scan_grid(results, out_path):
    z_list = sorted(set(row["z"] for row in results))
    sigma8_list = sorted(set(row["sigma8"] for row in results))
    fig, axes = plt.subplots(len(z_list), len(sigma8_list), figsize=(12, 10), sharex=True, sharey=True)
    for i, z in enumerate(z_list):
        for j, sigma8 in enumerate(sigma8_list):
            ax = axes[i, j]
            row = next(r for r in results if r["z"] == z and r["sigma8"] == sigma8)
            scan = row["scan"]
            x = [r["xi_min"] for r in scan if r["alpha_hill"] is not None]
            y_h = [r["alpha_hill"] for r in scan if r["alpha_hill"] is not None]
            y_p = [r["alpha_pareto"] for r in scan if r["alpha_pareto"] is not None]
            if x:
                ax.plot(x, y_h, lw=1.5, label="Hill")
                ax.plot(x, y_p, lw=1.2, ls="--", label="Pareto")
                ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"s8={sigma8:.2f}")
            if j == 0:
                ax.set_ylabel(f"z={z:.1f}\nalpha")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("xi_min")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def write_scan_csv(results, path):
    fieldnames = ["z", "sigma8", "xi_min", "alpha_hill", "alpha_pareto", "tail_event_count"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for scan in row["scan"]:
                writer.writerow(
                    {
                        "z": row["z"],
                        "sigma8": row["sigma8"],
                        "xi_min": scan["xi_min"],
                        "alpha_hill": scan["alpha_hill"],
                        "alpha_pareto": scan["alpha_pareto"],
                        "tail_event_count": scan["tail_event_count"],
                    }
                )


def main():
    parser = argparse.ArgumentParser(description="Scan threshold-dependent tail exponents alpha(xi_min) across the clean Track B grid.")
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--xmin-log10", type=float, default=-4.5)
    parser.add_argument("--xmax-log10", type=float, default=-2.0)
    parser.add_argument("--n-thresholds", type=int, default=40)
    parser.add_argument("--plateau-diff-tol", type=float, default=0.03)
    parser.add_argument("--plateau-min-run", type=int, default=4)
    parser.add_argument("--z-list", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--sigma8-list", default="0.75,0.81,0.87")
    parser.add_argument("--out-prefix", default="alpha_threshold_scan")
    args = parser.parse_args()

    gw = load_gwlensing()
    z_list = [float(x) for x in args.z_list.split(",") if x.strip()]
    sigma8_list = [float(x) for x in args.sigma8_list.split(",") if x.strip()]
    thresholds = np.logspace(args.xmin_log10, args.xmax_log10, args.n_thresholds)

    results = []
    for z in z_list:
        for sigma8 in sigma8_list:
            xi_evt = sample_xi(
                gw,
                z=z,
                sigma8=sigma8,
                omega_m=args.OmegaM,
                h=args.h,
                nreal=args.Nreal,
                seed=args.seed,
                nhalos=args.Nhalos,
            )
            scan = threshold_scan(xi_evt, thresholds)
            results.append(
                {
                    "z": float(z),
                    "sigma8": float(sigma8),
                    "Nreal": int(args.Nreal),
                    "n_events_good_mu": int(xi_evt.size),
                    "scan": scan,
                    "plateau_hill": summarize_plateau(scan, "alpha_hill", args.plateau_diff_tol, args.plateau_min_run),
                    "plateau_pareto": summarize_plateau(scan, "alpha_pareto", args.plateau_diff_tol, args.plateau_min_run),
                }
            )

    out_prefix = Path(args.out_prefix)
    write_scan_csv(results, f"{out_prefix}.csv")
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    plot_scan_grid(results, f"{out_prefix}_grid.png")
    print("WROTE", f"{out_prefix}.csv", f"{out_prefix}.json", f"{out_prefix}_grid.png")


if __name__ == "__main__":
    main()
