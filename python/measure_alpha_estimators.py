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
    pred = intercept + slope * lx
    sse = float(np.sum((ly - pred) ** 2))
    return {"A": float(np.exp(intercept)), "alpha": float(alpha), "sse_log": sse}


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
    return centers, density, counts


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
                "ratio_at_break": float(ratio[i]),
            }

    candidate = np.where(cv >= count_threshold)[0]
    if candidate.size == 0:
        return None
    idx = int(candidate[np.argmin(ratio[candidate])])
    return {
        "xi_break": float(xv[idx]),
        "baseline_alpha": float(baseline_fit["alpha"]),
        "ratio_at_break": float(ratio[idx]),
    }


def local_slope_alpha(x, y, xmin, xmax, smooth_window=5):
    mask = (x >= xmin) & (x <= xmax) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < max(7, smooth_window + 2):
        return None
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
    pad = smooth_window // 2
    ly_s = np.convolve(np.pad(ly, pad, mode="edge"), kernel, mode="valid")
    slope = np.gradient(ly_s, lx)
    alpha_local = -slope - 1.0
    return {
        "alpha": float(np.median(alpha_local)),
        "alpha_p16": float(np.percentile(alpha_local, 16)),
        "alpha_p84": float(np.percentile(alpha_local, 84)),
        "n_points": int(alpha_local.size),
    }


def truncated_power_law_mle(x, xmin, xmax):
    sample = np.asarray(x[(x >= xmin) & (x <= xmax)], dtype=float)
    if sample.size < 30:
        return None

    alpha_grid = np.linspace(0.6, 1.4, 161)
    xi_c_grid = np.geomspace(max(1.2 * xmax, xmin * 8.0), max(50.0 * xmax, xmin * 200.0), 140)
    logx = np.log(sample)
    best = None
    norm_grid = np.geomspace(xmin, xmax, 600)

    for alpha in alpha_grid:
        base = norm_grid ** (-1.0 - alpha)
        for xi_c in xi_c_grid:
            f = base * np.exp(-norm_grid / xi_c)
            z_norm = np.trapezoid(f, norm_grid)
            if not np.isfinite(z_norm) or z_norm <= 0.0:
                continue
            logpdf = -(1.0 + alpha) * logx - sample / xi_c - np.log(z_norm)
            nll = -float(np.sum(logpdf))
            if best is None or nll < best["nll"]:
                best = {
                    "alpha": float(alpha),
                    "xi_c": float(xi_c),
                    "nll": nll,
                    "n_samples": int(sample.size),
                    "xmin": float(xmin),
                    "xmax": float(xmax),
                }
    return best


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


def measure_run(gw, *, z, sigma8, omega_m, h, nreal, seed, nhalos, fixed_xmin, fixed_xmax, detect_xmin, rel_lo, rel_hi):
    xi_evt = sample_xi(
        gw,
        z=z,
        sigma8=sigma8,
        omega_m=omega_m,
        h=h,
        nreal=nreal,
        seed=seed,
        nhalos=nhalos,
    )
    xi_bins = np.geomspace(max(1e-8, xi_evt.min()), xi_evt.max(), 120)
    xi_centers, xi_density, xi_counts = hist_density(xi_evt, xi_bins)

    fixed_fit = fit_power_law_window(xi_centers, xi_density, fixed_xmin, fixed_xmax)
    break_info = detect_break(
        xi_centers,
        xi_density,
        xi_counts,
        detect_xmin=detect_xmin,
        run_length=3,
        drop_ratio=0.7,
        min_count_frac=0.02,
    )

    rel_fit = None
    local_fit = None
    mle_fit = None
    if break_info is not None:
        rel_xmin = max(detect_xmin, rel_lo * break_info["xi_break"])
        rel_xmax = rel_hi * break_info["xi_break"]
        rel_fit = fit_power_law_window(xi_centers, xi_density, rel_xmin, rel_xmax)
        local_fit = local_slope_alpha(xi_centers, xi_density, rel_xmin, rel_xmax)
        mle_fit = truncated_power_law_mle(xi_evt, rel_xmin, rel_xmax)

    return {
        "z": float(z),
        "sigma8": float(sigma8),
        "Nreal": int(nreal),
        "n_events_good_mu": int(xi_evt.size),
        "xi_break": None if break_info is None else float(break_info["xi_break"]),
        "alpha_fixed": None if fixed_fit is None else float(fixed_fit["alpha"]),
        "alpha_relative": None if rel_fit is None else float(rel_fit["alpha"]),
        "alpha_local": None if local_fit is None else float(local_fit["alpha"]),
        "alpha_local_p16": None if local_fit is None else float(local_fit["alpha_p16"]),
        "alpha_local_p84": None if local_fit is None else float(local_fit["alpha_p84"]),
        "alpha_mle": None if mle_fit is None else float(mle_fit["alpha"]),
        "xi_c_mle": None if mle_fit is None else float(mle_fit["xi_c"]),
        "tail_event_count_relative": 0 if rel_fit is None else int(np.count_nonzero((xi_evt >= rel_fit["xmin"]) & (xi_evt <= rel_fit["xmax"]))),
        "fixed_fit": fixed_fit,
        "relative_fit": rel_fit,
        "local_fit": local_fit,
        "mle_fit": mle_fit,
        "break_info": break_info,
    }


def write_csv(rows, path):
    fieldnames = [
        "z",
        "sigma8",
        "Nreal",
        "n_events_good_mu",
        "xi_break",
        "alpha_fixed",
        "alpha_relative",
        "alpha_local",
        "alpha_local_p16",
        "alpha_local_p84",
        "alpha_mle",
        "xi_c_mle",
        "tail_event_count_relative",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_alpha_vs_z(rows, out_path):
    methods = [
        ("alpha_fixed", "Fixed Window"),
        ("alpha_relative", "Relative Window"),
        ("alpha_local", "Local Slope"),
        ("alpha_mle", "TPL MLE"),
    ]
    z_list = sorted(set(r["z"] for r in rows))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, (key, label) in zip(axes.flat, methods):
        for sigma8 in sorted(set(r["sigma8"] for r in rows)):
            subset = sorted([r for r in rows if r["sigma8"] == sigma8 and r[key] is not None], key=lambda r: r["z"])
            ax.plot([r["z"] for r in subset], [r[key] for r in subset], "o-", lw=1.5, label=f"s8={sigma8:.2f}")
        ax.set_title(label)
        ax.set_ylabel("alpha")
    for ax in axes[1]:
        ax.set_xlabel("z")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple alpha estimators across the clean Track B grid.")
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--fixed-xmin", type=float, default=1e-4)
    parser.add_argument("--fixed-xmax", type=float, default=1e-2)
    parser.add_argument("--detect-xmin", type=float, default=1e-4)
    parser.add_argument("--rel-lo", type=float, default=0.05)
    parser.add_argument("--rel-hi", type=float, default=0.5)
    parser.add_argument("--z-list", default="0.5,1.0,2.0,5.0")
    parser.add_argument("--sigma8-list", default="0.75,0.81,0.87")
    parser.add_argument("--out-prefix", default="alpha_estimators")
    args = parser.parse_args()

    gw = load_gwlensing()
    z_list = [float(x) for x in args.z_list.split(",") if x.strip()]
    sigma8_list = [float(x) for x in args.sigma8_list.split(",") if x.strip()]

    rows = []
    for z in z_list:
        for sigma8 in sigma8_list:
            rows.append(
                measure_run(
                    gw,
                    z=z,
                    sigma8=sigma8,
                    omega_m=args.OmegaM,
                    h=args.h,
                    nreal=args.Nreal,
                    seed=args.seed,
                    nhalos=args.Nhalos,
                    fixed_xmin=args.fixed_xmin,
                    fixed_xmax=args.fixed_xmax,
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
    print("WROTE", f"{out_prefix}.csv", f"{out_prefix}.json", f"{out_prefix}_vs_z.png")


if __name__ == "__main__":
    main()
