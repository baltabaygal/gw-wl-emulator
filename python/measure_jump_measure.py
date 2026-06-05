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


def fit_truncated_power_law(x, y):
    lx = np.log(x)
    ly = np.log(y)
    X = np.column_stack([np.ones_like(x), lx, x])
    coeff, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
    c0, c1, c2 = coeff
    alpha = -c1 - 1.0
    xi_c = np.inf if c2 >= 0 else -1.0 / c2
    A = np.exp(c0)
    pred = X @ coeff
    sse = float(np.sum((ly - pred) ** 2))
    return {"A": float(A), "alpha": float(alpha), "xi_c": float(xi_c), "sse_log": sse}


def fit_stretched_exponential(x, y):
    p_grid = np.linspace(0.3, 2.0, 120)
    best = None
    ly = np.log(y)
    for p in p_grid:
        xp = x ** p
        X = np.column_stack([np.ones_like(x), xp])
        coeff, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
        c0, c1 = coeff
        if c1 >= 0:
            continue
        xi_c = (-1.0 / c1) ** (1.0 / p)
        pred = X @ coeff
        sse = float(np.sum((ly - pred) ** 2))
        cand = {"A": float(np.exp(c0)), "p": float(p), "xi_c": float(xi_c), "sse_log": sse}
        if best is None or cand["sse_log"] < best["sse_log"]:
            best = cand
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Measure the single-event jump measure R(kappa) and R(xi) from event-level simulator output."
    )
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--filaments", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--ell", action="store_true")
    parser.add_argument("--exact-poisson", action="store_true")
    parser.add_argument("--tail-quantile", type=float, default=0.8)
    parser.add_argument("--power-xmin", type=float, default=1e-4)
    parser.add_argument("--power-xmax", type=float, default=1e-2)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--out-prefix", default="jump_measure")
    args = parser.parse_args()

    gw = load_gwlensing()
    events = gw.sample_lensing_events(
        z=args.z,
        OmegaM=args.OmegaM,
        sigma8=args.sigma8,
        h=args.h,
        Nreal=args.Nreal,
        seed=args.seed,
        filaments=args.filaments,
        bias=args.bias,
        ell=args.ell,
        exact_poisson=args.exact_poisson,
        Nhalos=args.Nhalos,
    )

    is_filament = np.asarray(events["is_filament"], dtype=int)
    z_evt = np.asarray(events["z"], dtype=float)
    M_evt = np.asarray(events["M"], dtype=float)
    b_evt = np.asarray(events["b"], dtype=float)
    kappa_evt = np.asarray(events["kappa"], dtype=float)
    gamma1_evt = np.asarray(events["gamma1"], dtype=float)
    gamma2_evt = np.asarray(events["gamma2"], dtype=float)
    gamma_evt = np.sqrt(gamma1_evt ** 2 + gamma2_evt ** 2)

    mu_evt = 1.0 / ((1.0 - kappa_evt) ** 2 - gamma_evt ** 2)
    good = np.isfinite(mu_evt) & (mu_evt > 0.0)
    xi_evt = -np.log((1.0 - kappa_evt[good]) ** 2 - gamma_evt[good] ** 2)

    def hist_density(values, bins):
        hist, edges = np.histogram(values, bins=bins, density=False)
        width = np.diff(edges)
        centers = np.sqrt(edges[:-1] * edges[1:]) if np.all(edges > 0) else 0.5 * (edges[:-1] + edges[1:])
        density = hist / np.sum(hist) / width
        return centers, density, edges

    kappa_bins_lin = np.linspace(max(0.0, kappa_evt.min()), kappa_evt.max(), 140)
    xi_bins_lin = np.linspace(max(0.0, xi_evt.min()), xi_evt.max(), 140)
    xi_bins_log = np.geomspace(max(1e-8, xi_evt.min()), xi_evt.max(), 120)

    kappa_c_lin, kappa_d_lin, _ = hist_density(kappa_evt, kappa_bins_lin)
    xi_c_lin, xi_d_lin, _ = hist_density(xi_evt, xi_bins_lin)
    xi_c_log, xi_d_log, _ = hist_density(xi_evt, xi_bins_log)

    tail_cut = float(np.quantile(xi_evt, args.tail_quantile))
    tail_mask = (xi_c_log >= tail_cut) & np.isfinite(xi_d_log) & (xi_d_log > 0)
    x_tail = xi_c_log[tail_mask]
    y_tail = xi_d_log[tail_mask]

    fit_power = fit_power_law(x_tail, y_tail) if x_tail.size >= 5 else None
    fit_tpl = fit_truncated_power_law(x_tail, y_tail) if x_tail.size >= 5 else None
    fit_se = fit_stretched_exponential(x_tail, y_tail) if x_tail.size >= 5 else None
    fit_power_window = fit_power_law_window(xi_c_log, xi_d_log, args.power_xmin, args.power_xmax)

    window_scan = []
    for xmin, xmax in [
        (1e-4, 1e-2),
        (2e-4, 1e-2),
        (1e-4, 5e-3),
        (2e-4, 5e-3),
        (5e-4, 1e-2),
    ]:
        fit = fit_power_law_window(xi_c_log, xi_d_log, xmin, xmax)
        if fit is not None:
            window_scan.append(fit)

    bootstrap_alpha = None
    if args.bootstrap > 0 and fit_power_window is not None:
        rng = np.random.default_rng(args.seed)
        alpha_samples = []
        for _ in range(args.bootstrap):
            sample = rng.choice(xi_evt, size=xi_evt.size, replace=True)
            hist, _ = np.histogram(sample, bins=xi_bins_log, density=False)
            dens = hist / np.sum(hist) / np.diff(xi_bins_log)
            fit = fit_power_law_window(xi_c_log, dens, args.power_xmin, args.power_xmax)
            if fit is not None and np.isfinite(fit["alpha"]):
                alpha_samples.append(fit["alpha"])
        if alpha_samples:
            alpha_samples = np.asarray(alpha_samples, dtype=float)
            bootstrap_alpha = {
                "n_success": int(alpha_samples.size),
                "mean": float(np.mean(alpha_samples)),
                "std": float(np.std(alpha_samples, ddof=1)) if alpha_samples.size > 1 else 0.0,
                "p16": float(np.percentile(alpha_samples, 16)),
                "p50": float(np.percentile(alpha_samples, 50)),
                "p84": float(np.percentile(alpha_samples, 84)),
            }

    summary = {
        "z": args.z,
        "Nreal": args.Nreal,
        "Nhalos": args.Nhalos,
        "filaments": bool(args.filaments),
        "bias": bool(args.bias),
        "ell": bool(args.ell),
        "exact_poisson": bool(args.exact_poisson),
        "n_events_total": int(kappa_evt.size),
        "n_events_good_mu": int(xi_evt.size),
        "n_halo_events": int(np.count_nonzero(is_filament == 0)),
        "n_filament_events": int(np.count_nonzero(is_filament == 1)),
        "tail_quantile": float(args.tail_quantile),
        "tail_xi_min": tail_cut,
        "fit_power_law_window": fit_power_window,
        "fit_power_law_window_scan": window_scan,
        "bootstrap_alpha": bootstrap_alpha,
        "fit_power_law": fit_power,
        "fit_truncated_power_law": fit_tpl,
        "fit_stretched_exponential": fit_se,
    }

    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    np.savez(
        f"{args.out_prefix}_events.npz",
        is_filament=is_filament,
        z=z_evt,
        M=M_evt,
        b=b_evt,
        kappa=kappa_evt,
        gamma1=gamma1_evt,
        gamma2=gamma2_evt,
        xi=xi_evt,
    )

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.2))

    ax[0, 0].plot(kappa_c_lin, kappa_d_lin, lw=1.6)
    ax[0, 0].set_xlabel("kappa")
    ax[0, 0].set_ylabel("R(kappa) [normalized]")
    ax[0, 0].set_title("Single-event kappa distribution")

    ax[0, 1].plot(xi_c_lin, xi_d_lin, lw=1.6)
    ax[0, 1].set_xlabel("xi")
    ax[0, 1].set_ylabel("R(xi) [normalized]")
    ax[0, 1].set_title("Single-event xi distribution (linear)")

    ax[1, 0].plot(xi_c_log, xi_d_log, lw=1.6, label="empirical")
    if fit_power_window is not None:
        xw = np.geomspace(fit_power_window["xmin"], fit_power_window["xmax"], 120)
        ax[1, 0].plot(
            xw,
            fit_power_window["A"] * xw ** (-1.0 - fit_power_window["alpha"]),
            ls="-.",
            lw=1.3,
            label=f"window alpha={fit_power_window['alpha']:.3f}",
        )
    if fit_power is not None:
        ax[1, 0].plot(x_tail, fit_power["A"] * x_tail ** (-1.0 - fit_power["alpha"]), ls="--", lw=1.2, label="power law")
    if fit_tpl is not None and np.isfinite(fit_tpl["xi_c"]):
        ax[1, 0].plot(
            x_tail,
            fit_tpl["A"] * x_tail ** (-1.0 - fit_tpl["alpha"]) * np.exp(-x_tail / fit_tpl["xi_c"]),
            ls="--",
            lw=1.2,
            label="truncated power law",
        )
    if fit_se is not None:
        ax[1, 0].plot(
            x_tail,
            fit_se["A"] * np.exp(-(x_tail / fit_se["xi_c"]) ** fit_se["p"]),
            ls="--",
            lw=1.2,
            label="stretched exp",
        )
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("xi")
    ax[1, 0].set_ylabel("R(xi)")
    ax[1, 0].set_title("Tail region (log-log)")
    ax[1, 0].legend(frameon=False, fontsize=8)

    ax[1, 1].plot(xi_c_log, np.log(np.clip(xi_d_log, 1e-300, None)), lw=1.6)
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xlabel("xi")
    ax[1, 1].set_ylabel("log R(xi)")
    title = "Tail region (log-linear)"
    if bootstrap_alpha is not None:
        title += f"\nalpha={bootstrap_alpha['p50']:.3f} +/- {bootstrap_alpha['std']:.3f}"
    elif fit_power_window is not None:
        title += f"\nalpha={fit_power_window['alpha']:.3f}"
    ax[1, 1].set_title(title)

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=170)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {args.out_prefix}.png {args.out_prefix}_summary.json {args.out_prefix}_events.npz")


if __name__ == "__main__":
    main()
