import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.special import fresnel
except Exception:
    fresnel = None


def load_gwlensing():
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))
    import gwlensing
    return gwlensing


def fit_powerlaw(centers, density, kthr, mult_hi):
    mask = (
        np.isfinite(centers)
        & np.isfinite(density)
        & (density > 0)
        & (centers >= kthr)
        & (centers <= mult_hi * kthr)
    )
    if np.count_nonzero(mask) < 3:
        raise RuntimeError(f"Not enough points for fit window up to {mult_hi} * kappa_c")
    x = np.log(np.clip(centers[mask] - kthr, 1e-300, None))
    y = np.log(density[mask])
    slope, intercept = np.polyfit(x, y, 1)
    alpha = -slope
    amp = np.exp(intercept)
    return alpha, amp, mask


def singular_transform(q, kthr, kmax, amp):
    delta = max(kmax - kthr, 0.0)
    if delta <= 0.0:
        return np.zeros_like(q, dtype=np.complex128)
    if fresnel is None:
        raise RuntimeError("scipy.special.fresnel is required for singular_transform")

    out = np.zeros_like(q, dtype=np.complex128)
    sqrt_delta = np.sqrt(delta)
    zero = np.abs(q) < 1.0e-14
    out[zero] = np.zeros(np.count_nonzero(zero), dtype=np.complex128)

    nz = ~zero
    qnz = q[nz]
    t = np.sqrt(2.0 * np.abs(qnz) / np.pi) * sqrt_delta
    s, c = fresnel(t)
    fres = c + 1j * np.sign(qnz) * s
    pref = amp * np.exp(-1j * qnz * kthr) * np.sqrt(2.0 * np.pi / np.abs(qnz))
    osc = np.exp(-1j * np.sign(qnz) * np.pi / 4.0) * fres
    integral = pref * osc
    out[nz] = integral - 2.0 * amp * sqrt_delta
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Prototype singularity subtraction for the additive-field kappa generator."
    )
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--n-u", dest="n_u", type=int, default=512)
    parser.add_argument("--qmax", type=float, default=40.0)
    parser.add_argument("--nq", type=int, default=101)
    parser.add_argument("--bins", type=int, default=220)
    parser.add_argument("--fit-window", dest="fit_window", type=float, default=1.1)
    parser.add_argument("--phi-floor", dest="phi_floor", type=float, default=2e-2)
    parser.add_argument("--out-prefix", default="singularity_subtraction_test")
    args = parser.parse_args()

    gw = load_gwlensing()
    q = np.linspace(-args.qmax, args.qmax, args.nq)

    raw = gw.sample_lensing_raw(
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
    )
    kappa_mc = np.asarray(raw["kappa"], dtype=float)
    phi_emp = np.exp(-1j * np.outer(q, kappa_mc)).mean(axis=1)
    lam_emp = np.log(phi_emp)

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
    edges = np.geomspace(max(kthr * 1.000001, 1e-12), kmax, args.bins + 1)
    hist, _ = np.histogram(kappa, bins=edges, weights=weight, density=False)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = hist / widths

    alpha, amp, fit_mask = fit_powerlaw(centers, density, kthr, args.fit_window)
    singular_density = amp * np.clip(centers - kthr, 1e-300, None) ** (-alpha)
    smooth_density = density - singular_density

    phi_hist = np.array(
        [np.sum(density * widths * np.exp(-1j * qq * centers)) for qq in q],
        dtype=np.complex128,
    )
    lam_hist = phi_hist - np.sum(density * widths)

    phi_smooth = np.array(
        [np.sum(smooth_density * widths * np.exp(-1j * qq * centers)) for qq in q],
        dtype=np.complex128,
    )
    lam_sub = phi_smooth - np.sum(smooth_density * widths) + singular_transform(q, kthr, edges[-1], amp)

    mask = np.abs(phi_emp) > args.phi_floor
    delta_hist = lam_emp - lam_hist
    delta_sub = lam_emp - lam_sub

    def summarize(delta):
        abs_delta = np.abs(delta)
        return {
            "median_abs_error_reliable": float(np.median(abs_delta[mask])),
            "max_abs_error_reliable": float(np.max(abs_delta[mask])),
            "corr_abs_delta_vs_abs_q": float(np.corrcoef(np.abs(q[mask]), abs_delta[mask])[0, 1]),
            "high_to_low_ratio": float(
                np.median(abs_delta[mask][np.argsort(np.abs(q[mask]))][-max(1, np.count_nonzero(mask)//3):]) /
                np.median(abs_delta[mask][np.argsort(np.abs(q[mask]))][:max(1, np.count_nonzero(mask)//3)])
            ),
        }

    summary = {
        "z": args.z,
        "Nreal": args.Nreal,
        "Nhalos": args.Nhalos,
        "n_u": args.n_u,
        "fit_window_multiplier": args.fit_window,
        "kappa_c": kthr,
        "A": float(amp),
        "alpha": float(alpha),
        "hist_mode": summarize(delta_hist),
        "subtracted_mode": summarize(delta_sub),
    }

    out_prefix = args.out_prefix
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots(2, 2, figsize=(10.0, 7.6))
    order = np.argsort(np.abs(q))

    ax[0, 0].plot(centers, density, lw=1.8, label="R")
    ax[0, 0].plot(centers, singular_density, lw=1.4, ls="--", label="R_sing")
    ax[0, 0].plot(centers, np.abs(smooth_density), lw=1.2, label="|R_smooth|")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_yscale("log")
    ax[0, 0].legend(frameon=False, fontsize=8)
    ax[0, 0].set_title(f"alpha={alpha:.3f}, A={amp:.3e}")

    ax[0, 1].plot(np.abs(q)[order], np.abs(delta_hist)[order], lw=1.6, label="raw hist residual")
    ax[0, 1].plot(np.abs(q)[order], np.abs(delta_sub)[order], lw=1.6, label="subtracted residual")
    ax[0, 1].set_xlabel("|q|")
    ax[0, 1].set_ylabel("|deltaLambda|")
    ax[0, 1].legend(frameon=False, fontsize=8)

    ax[1, 0].plot(q, delta_hist.real, lw=1.2, label="raw Re")
    ax[1, 0].plot(q, delta_sub.real, lw=1.2, label="sub Re")
    ax[1, 0].axhline(0.0, color="k", ls="--", lw=0.8)
    ax[1, 0].legend(frameon=False, fontsize=8)

    ax[1, 1].plot(q, delta_hist.imag, lw=1.2, label="raw Im")
    ax[1, 1].plot(q, delta_sub.imag, lw=1.2, label="sub Im")
    ax[1, 1].axhline(0.0, color="k", ls="--", lw=0.8)
    ax[1, 1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
