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


def fit_power_law(x, y):
    lx = np.log(x)
    ly = np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    alpha = -slope - 1.0
    A = np.exp(intercept)
    pred = intercept + slope * lx
    sse = float(np.sum((ly - pred) ** 2))
    return {"A": float(A), "alpha": float(alpha), "sse_log": sse}


def fit_power_law_window(x, y, xmin, xmax):
    mask = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 5:
        return None
    fit = fit_power_law(x[mask], y[mask])
    fit["xmin"] = float(xmin)
    fit["xmax"] = float(xmax)
    fit["n_points"] = int(np.count_nonzero(mask))
    return fit


def hist_density(values, bins):
    counts, edges = np.histogram(values, bins=bins, density=False)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = counts / np.sum(counts) / widths
    return centers, density, counts, edges


def smooth_series(values, window):
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def detect_break(x, y, counts, *, detect_xmin, run_length, drop_ratio, min_count_frac):
    valid = np.isfinite(y) & (y > 0) & (x >= detect_xmin)
    if np.count_nonzero(valid) < 12:
        return None

    xv = x[valid]
    yv = y[valid]
    cv = counts[valid]
    baseline_mask = (xv >= detect_xmin) & (xv <= max(detect_xmin * 5.0, detect_xmin * 1.5))
    if np.count_nonzero(baseline_mask) < 5:
        baseline_mask = np.arange(xv.size) < min(8, xv.size)
    if np.count_nonzero(baseline_mask) < 5:
        return None

    baseline_fit = fit_power_law(xv[baseline_mask], yv[baseline_mask])
    y_pred = baseline_fit["A"] * xv ** (-1.0 - baseline_fit["alpha"])
    ratio = yv / y_pred
    count_threshold = max(5.0, min_count_frac * float(np.max(cv)))
    trigger = (cv >= count_threshold) & (ratio < drop_ratio)

    for i in range(0, max(0, trigger.size - run_length + 1)):
        if np.all(trigger[i : i + run_length]):
            return {
                "xi_break": float(xv[i]),
                "baseline_alpha": float(baseline_fit["alpha"]),
                "drop_ratio": float(drop_ratio),
                "ratio_at_break": float(ratio[i]),
                "count_threshold": float(count_threshold),
                "trigger_index": int(i),
            }

    candidate = np.where(cv >= count_threshold)[0]
    if candidate.size == 0:
        return None
    idx = int(candidate[np.argmin(ratio[candidate])])
    return {
        "xi_break": float(xv[idx]),
        "baseline_alpha": float(baseline_fit["alpha"]),
        "drop_ratio": float(drop_ratio),
        "ratio_at_break": float(ratio[idx]),
        "count_threshold": float(count_threshold),
        "trigger_index": idx,
    }


def bootstrap_alpha(xi_evt, bins, centers, xmin, xmax, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    alpha_samples = []
    tail_counts = []
    for _ in range(n_bootstrap):
        sample = rng.choice(xi_evt, size=xi_evt.size, replace=True)
        hist, _ = np.histogram(sample, bins=bins, density=False)
        dens = hist / np.sum(hist) / np.diff(bins)
        fit = fit_power_law_window(centers, dens, xmin, xmax)
        if fit is not None and np.isfinite(fit["alpha"]):
            alpha_samples.append(fit["alpha"])
            tail_counts.append(int(np.count_nonzero((sample >= xmin) & (sample <= xmax))))
    if not alpha_samples:
        return None
    alpha_samples = np.asarray(alpha_samples, dtype=float)
    tail_counts = np.asarray(tail_counts, dtype=int)
    return {
        "n_success": int(alpha_samples.size),
        "mean": float(np.mean(alpha_samples)),
        "std": float(np.std(alpha_samples, ddof=1)) if alpha_samples.size > 1 else 0.0,
        "p16": float(np.percentile(alpha_samples, 16)),
        "p50": float(np.percentile(alpha_samples, 50)),
        "p84": float(np.percentile(alpha_samples, 84)),
        "tail_event_count_mean": float(np.mean(tail_counts)),
        "tail_event_count_min": int(np.min(tail_counts)),
        "tail_event_count_max": int(np.max(tail_counts)),
    }


def measure_alpha_for_run(
    gw,
    *,
    z,
    sigma8,
    omega_m,
    h,
    nreal,
    seed,
    nhalos,
    bootstrap,
    detect_xmin,
    rel_lo,
    rel_hi,
):
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
    xi_evt = -np.log((1.0 - kappa_evt[good]) ** 2 - gamma_evt[good] ** 2)

    xi_bins_log = np.geomspace(max(1e-8, xi_evt.min()), xi_evt.max(), 120)
    xi_centers, xi_density, xi_counts, _ = hist_density(xi_evt, xi_bins_log)

    break_info = detect_break(
        xi_centers,
        xi_density,
        xi_counts,
        detect_xmin=detect_xmin,
        run_length=3,
        drop_ratio=0.7,
        min_count_frac=0.02,
    )
    if break_info is None:
        return {
            "z": float(z),
            "sigma8": float(sigma8),
            "Nreal": int(nreal),
            "Nhalos": int(nhalos),
            "alpha": None,
            "alpha_bootstrap_std": None,
            "alpha_p16": None,
            "alpha_p50": None,
            "alpha_p84": None,
            "tail_event_count": 0,
            "n_events_good_mu": int(xi_evt.size),
            "xi_break": None,
            "fit_window": None,
            "bootstrap": None,
            "break_info": None,
        }

    xmin = max(detect_xmin, rel_lo * break_info["xi_break"])
    xmax = rel_hi * break_info["xi_break"]
    fit_window = fit_power_law_window(xi_centers, xi_density, xmin, xmax)
    if fit_window is None:
        return {
            "z": float(z),
            "sigma8": float(sigma8),
            "Nreal": int(nreal),
            "Nhalos": int(nhalos),
            "alpha": None,
            "alpha_bootstrap_std": None,
            "alpha_p16": None,
            "alpha_p50": None,
            "alpha_p84": None,
            "tail_event_count": 0,
            "n_events_good_mu": int(xi_evt.size),
            "xi_break": float(break_info["xi_break"]),
            "fit_window": None,
            "bootstrap": None,
            "break_info": break_info,
        }

    boot = bootstrap_alpha(
        xi_evt,
        xi_bins_log,
        xi_centers,
        xmin,
        xmax,
        bootstrap,
        seed + 31,
    ) if bootstrap > 0 else None

    return {
        "z": float(z),
        "sigma8": float(sigma8),
        "Nreal": int(nreal),
        "Nhalos": int(nhalos),
        "alpha": float(fit_window["alpha"]),
        "alpha_bootstrap_std": None if boot is None else boot["std"],
        "alpha_p16": None if boot is None else boot["p16"],
        "alpha_p50": None if boot is None else boot["p50"],
        "alpha_p84": None if boot is None else boot["p84"],
        "tail_event_count": int(np.count_nonzero((xi_evt >= xmin) & (xi_evt <= xmax))),
        "n_events_good_mu": int(xi_evt.size),
        "xi_break": float(break_info["xi_break"]),
        "fit_window": fit_window,
        "bootstrap": boot,
        "break_info": break_info,
    }


def write_csv(rows, path):
    fieldnames = [
        "z",
        "sigma8",
        "Nreal",
        "Nhalos",
        "alpha",
        "alpha_bootstrap_std",
        "alpha_p16",
        "alpha_p50",
        "alpha_p84",
        "xi_break",
        "tail_event_count",
        "n_events_good_mu",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_alpha_vs_z(rows, out_path):
    sigma8_list = sorted(set(row["sigma8"] for row in rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for sigma8 in sigma8_list:
        subset = sorted([r for r in rows if r["sigma8"] == sigma8 and r["alpha"] is not None], key=lambda r: r["z"])
        z = [r["z"] for r in subset]
        a = [r["alpha_p50"] if r["alpha_p50"] is not None else r["alpha"] for r in subset]
        lo = [r["alpha_p16"] if r["alpha_p16"] is not None else r["alpha"] for r in subset]
        hi = [r["alpha_p84"] if r["alpha_p84"] is not None else r["alpha"] for r in subset]
        ax.plot(z, a, "o-", lw=1.8, label=f"sigma8={sigma8:.2f}")
        ax.fill_between(z, lo, hi, alpha=0.15)
    ax.set_xlabel("z")
    ax.set_ylabel("alpha")
    ax.set_title("Relative-window alpha vs redshift")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def plot_alpha_vs_sigma8(rows, out_path):
    z_list = sorted(set(row["z"] for row in rows))
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for z in z_list:
        subset = sorted([r for r in rows if r["z"] == z and r["alpha"] is not None], key=lambda r: r["sigma8"])
        s = [r["sigma8"] for r in subset]
        a = [r["alpha_p50"] if r["alpha_p50"] is not None else r["alpha"] for r in subset]
        lo = [r["alpha_p16"] if r["alpha_p16"] is not None else r["alpha"] for r in subset]
        hi = [r["alpha_p84"] if r["alpha_p84"] is not None else r["alpha"] for r in subset]
        ax.plot(s, a, "o-", lw=1.8, label=f"z={z:.1f}")
        ax.fill_between(s, lo, hi, alpha=0.15)
    ax.set_xlabel("sigma8")
    ax.set_ylabel("alpha")
    ax.set_title("Relative-window alpha vs sigma8")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def plot_alpha_histogram(rows, out_path):
    alpha = np.array(
        [r["alpha_p50"] if r["alpha_p50"] is not None else r["alpha"] for r in rows if r["alpha"] is not None],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(alpha, bins=min(8, max(4, alpha.size // 2)), edgecolor="black")
    ax.set_xlabel("alpha")
    ax.set_ylabel("count")
    ax.set_title("Distribution of relative-window alpha values")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def main():
    parser = argparse.ArgumentParser(
        description="Measure alpha(z, sigma8) in a turnover-relative xi window across the Track A grid."
    )
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=64)
    parser.add_argument("--detect-xmin", type=float, default=1e-4)
    parser.add_argument("--rel-lo", type=float, default=0.05)
    parser.add_argument("--rel-hi", type=float, default=0.5)
    parser.add_argument("--z-list", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--sigma8-list", default="0.75,0.81,0.87")
    parser.add_argument("--out-prefix", default="alpha_grid_relative")
    args = parser.parse_args()

    gw = load_gwlensing()
    z_list = [float(x) for x in args.z_list.split(",") if x.strip()]
    sigma8_list = [float(x) for x in args.sigma8_list.split(",") if x.strip()]

    rows = []
    for z in z_list:
        for sigma8 in sigma8_list:
            rows.append(
                measure_alpha_for_run(
                    gw,
                    z=z,
                    sigma8=sigma8,
                    omega_m=args.OmegaM,
                    h=args.h,
                    nreal=args.Nreal,
                    seed=args.seed,
                    nhalos=args.Nhalos,
                    bootstrap=args.bootstrap,
                    detect_xmin=args.detect_xmin,
                    rel_lo=args.rel_lo,
                    rel_hi=args.rel_hi,
                )
            )

    out_prefix = Path(args.out_prefix)
    write_csv(rows, f"{out_prefix}.csv")
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    plot_alpha_vs_z(rows, f"{out_prefix}_vs_z.png")
    plot_alpha_vs_sigma8(rows, f"{out_prefix}_vs_sigma8.png")
    plot_alpha_histogram(rows, f"{out_prefix}_histogram.png")
    print(
        "WROTE",
        f"{out_prefix}.csv",
        f"{out_prefix}.json",
        f"{out_prefix}_vs_z.png",
        f"{out_prefix}_vs_sigma8.png",
        f"{out_prefix}_histogram.png",
    )


if __name__ == "__main__":
    main()
