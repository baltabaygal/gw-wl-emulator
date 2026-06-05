import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scan(path):
    with open(path) as f:
        return json.load(f)


def effective_alpha_truncated(alpha, xi_c, thresholds):
    out = []
    for u in thresholds:
        x_max = max(50.0 * xi_c, 30.0 * u)
        x = np.geomspace(u, x_max, 240)
        w = x ** (-1.0 - alpha) * np.exp(-x / xi_c)
        z = np.trapezoid(w, x)
        if not np.isfinite(z) or z <= 0.0:
            out.append(np.nan)
            continue
        elog = np.trapezoid(np.log(x / u) * w, x) / z
        out.append(np.nan if elog <= 0.0 else 1.0 / elog)
    return np.asarray(out, dtype=float)


def fit_scan_curve(thresholds, alpha_obs, counts, alpha_grid, xi_c_grid, min_count):
    mask = np.isfinite(alpha_obs) & np.isfinite(thresholds) & (counts >= min_count)
    if np.count_nonzero(mask) < 6:
        return None

    t = thresholds[mask]
    y = alpha_obs[mask]
    c = counts[mask].astype(float)
    weights = np.sqrt(c / np.max(c))

    best = None
    for alpha in alpha_grid:
        for xi_c in xi_c_grid:
            y_model = effective_alpha_truncated(alpha, xi_c, t)
            if not np.all(np.isfinite(y_model)):
                continue
            resid = y - y_model
            wrss = float(np.sum((weights * resid) ** 2))
            if best is None or wrss < best["wrss"]:
                best = {
                    "alpha": float(alpha),
                    "xi_c": float(xi_c),
                    "wrss": wrss,
                    "n_points": int(t.size),
                    "threshold_min": float(np.min(t)),
                    "threshold_max": float(np.max(t)),
                    "model_alpha": y_model.tolist(),
                    "obs_alpha": y.tolist(),
                    "thresholds": t.tolist(),
                    "counts": c.tolist(),
                }
    if best is None:
        return None

    y = np.asarray(best["obs_alpha"], dtype=float)
    y_model = np.asarray(best["model_alpha"], dtype=float)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    best["rmse"] = float(np.sqrt(np.mean((y - y_model) ** 2)))
    best["r2"] = None if ss_tot == 0.0 else float(1.0 - np.sum((y - y_model) ** 2) / ss_tot)
    return best


def summarize_by_z(rows):
    out = {}
    for z in sorted(set(row["z"] for row in rows)):
        subset = [row for row in rows if row["z"] == z and row["fit"] is not None]
        if not subset:
            continue
        out[str(z)] = {
            "alpha_mean": float(np.mean([row["fit"]["alpha"] for row in subset])),
            "alpha_min": float(np.min([row["fit"]["alpha"] for row in subset])),
            "alpha_max": float(np.max([row["fit"]["alpha"] for row in subset])),
            "xi_c_mean": float(np.mean([row["fit"]["xi_c"] for row in subset])),
            "xi_c_min": float(np.min([row["fit"]["xi_c"] for row in subset])),
            "xi_c_max": float(np.max([row["fit"]["xi_c"] for row in subset])),
        }
    return out


def write_csv(rows, path):
    fieldnames = [
        "z",
        "sigma8",
        "alpha_fit",
        "xi_c_fit",
        "rmse",
        "r2",
        "n_points",
        "threshold_min",
        "threshold_max",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            fit = row["fit"] or {}
            writer.writerow(
                {
                    "z": row["z"],
                    "sigma8": row["sigma8"],
                    "alpha_fit": fit.get("alpha"),
                    "xi_c_fit": fit.get("xi_c"),
                    "rmse": fit.get("rmse"),
                    "r2": fit.get("r2"),
                    "n_points": fit.get("n_points"),
                    "threshold_min": fit.get("threshold_min"),
                    "threshold_max": fit.get("threshold_max"),
                }
            )


def plot_fit_grid(rows, out_path):
    z_list = sorted(set(row["z"] for row in rows))
    sigma8_list = sorted(set(row["sigma8"] for row in rows))
    fig, axes = plt.subplots(len(z_list), len(sigma8_list), figsize=(12, 10), sharex=True, sharey=True)
    for i, z in enumerate(z_list):
        for j, sigma8 in enumerate(sigma8_list):
            ax = axes[i, j]
            row = next(r for r in rows if r["z"] == z and r["sigma8"] == sigma8)
            fit = row["fit"]
            scan = row["scan"]
            x = [s["xi_min"] for s in scan if s["alpha_hill"] is not None]
            y = [s["alpha_hill"] for s in scan if s["alpha_hill"] is not None]
            ax.plot(x, y, color="0.7", lw=1.2, label="observed")
            if fit is not None:
                tx = np.asarray(fit["thresholds"], dtype=float)
                ty = np.asarray(fit["model_alpha"], dtype=float)
                ax.plot(tx, ty, color="C1", lw=1.8, label="truncated-tail fit")
                ax.set_title(f"s8={sigma8:.2f}\nalpha={fit['alpha']:.2f}, xi_c={fit['xi_c']:.4g}")
            else:
                ax.set_title(f"s8={sigma8:.2f}\nno fit")
            ax.set_xscale("log")
            if j == 0:
                ax.set_ylabel(f"z={z:.1f}\nalpha_eff")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("xi_min")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def main():
    parser = argparse.ArgumentParser(description="Fit threshold-drift curves alpha(xi_min) with a truncated power-law tail model.")
    parser.add_argument("--scan-json", default="/Users/baltabay/Desktop/gw-wl-emulator/alpha_threshold_scan.json")
    parser.add_argument("--out-prefix", default="/Users/baltabay/Desktop/gw-wl-emulator/alpha_threshold_truncated_fit")
    parser.add_argument("--min-count", type=int, default=3000)
    parser.add_argument("--alpha-min", type=float, default=0.4)
    parser.add_argument("--alpha-max", type=float, default=1.4)
    parser.add_argument("--n-alpha", type=int, default=101)
    parser.add_argument("--xic-min", type=float, default=5e-4)
    parser.add_argument("--xic-max", type=float, default=2e-2)
    parser.add_argument("--n-xic", type=int, default=120)
    args = parser.parse_args()

    raw = load_scan(args.scan_json)
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.n_alpha)
    xi_c_grid = np.geomspace(args.xic_min, args.xic_max, args.n_xic)

    rows = []
    for row in raw:
        thresholds = np.asarray([s["xi_min"] for s in row["scan"]], dtype=float)
        alpha_obs = np.asarray([np.nan if s["alpha_hill"] is None else s["alpha_hill"] for s in row["scan"]], dtype=float)
        counts = np.asarray([s["tail_event_count"] for s in row["scan"]], dtype=int)
        fit = fit_scan_curve(thresholds, alpha_obs, counts, alpha_grid, xi_c_grid, args.min_count)
        rows.append(
            {
                "z": float(row["z"]),
                "sigma8": float(row["sigma8"]),
                "scan": row["scan"],
                "fit": fit,
            }
        )

    out_prefix = Path(args.out_prefix)
    write_csv(rows, f"{out_prefix}.csv")
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(
            {
                "rows": rows,
                "summary_by_z": summarize_by_z(rows),
                "fit_config": {
                    "min_count": args.min_count,
                    "alpha_min": args.alpha_min,
                    "alpha_max": args.alpha_max,
                    "n_alpha": args.n_alpha,
                    "xic_min": args.xic_min,
                    "xic_max": args.xic_max,
                    "n_xic": args.n_xic,
                },
            },
            f,
            indent=2,
            sort_keys=True,
        )
    plot_fit_grid(rows, f"{out_prefix}_grid.png")
    print("WROTE", f"{out_prefix}.csv", f"{out_prefix}.json", f"{out_prefix}_grid.png")


if __name__ == "__main__":
    main()
