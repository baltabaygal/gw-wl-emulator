import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw


@dataclass
class ConfigResult:
    z: float
    sigma8: float
    invalid_fraction: float
    stats: Dict[str, float]


def run_one(z: float, h: float, omega_m: float, sigma8: float, nsamples: int, seed: int, strict_weak_lensing: bool):
    out = gw.sample_lnmu_ml_with_diagnostics(
        z=z,
        h=h,
        OmegaM=omega_m,
        sigma8=sigma8,
        nsamples=nsamples,
        seed=seed,
        strict_weak_lensing=strict_weak_lensing,
    )
    stats = dict(out["invalid_stats"])
    total = max(1, int(stats["total_samples"]))
    invalid_fraction = float(stats["invalid_samples"]) / total
    return ConfigResult(z=z, sigma8=sigma8, invalid_fraction=invalid_fraction, stats=stats)


def collect_detA_mu(z: float, h: float, omega_m: float, sigma8: float, nsamples: int, seed: int):
    raw = gw.sample_lensing_raw_ml(
        z=z,
        h=h,
        OmegaM=omega_m,
        sigma8=sigma8,
        nsamples=nsamples,
        seed=seed,
        filaments=True,
        bias=True,
        ell=True,
        Nhalos=100,
    )
    kappa = np.asarray(raw["kappa"])
    gamma1 = np.asarray(raw["gamma1"])
    gamma2 = np.asarray(raw["gamma2"])

    meankappa = float(np.mean(kappa))
    kappaj = kappa - meankappa
    gamma = np.sqrt(gamma1 * gamma1 + gamma2 * gamma2)
    detA = (1.0 - kappaj) ** 2 - gamma ** 2
    mu = np.where(detA != 0.0, 1.0 / detA, np.inf)
    valid_mu_mask = np.isfinite(mu) & (mu > 0.0)
    lnmu = np.full_like(mu, np.nan, dtype=np.float64)
    lnmu[valid_mu_mask] = np.log(mu[valid_mu_mask])
    return detA, mu, lnmu


def plot_invalid_fraction(results: List[ConfigResult], zs: np.ndarray, sigma8s: np.ndarray, outdir: str):
    heat = np.full((len(zs), len(sigma8s)), np.nan, dtype=np.float64)
    for r in results:
        iz = np.where(np.isclose(zs, r.z))[0][0]
        is8 = np.where(np.isclose(sigma8s, r.sigma8))[0][0]
        heat[iz, is8] = r.invalid_fraction

    by_z = np.nanmean(heat, axis=1)
    by_s8 = np.nanmean(heat, axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(zs, by_z, marker="o")
    plt.xlabel("z")
    plt.ylabel("mean invalid fraction")
    plt.title("Invalid Fraction vs Redshift")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "invalid_fraction_vs_z.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(sigma8s, by_s8, marker="o")
    plt.xlabel("sigma8")
    plt.ylabel("mean invalid fraction")
    plt.title("Invalid Fraction vs sigma8")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "invalid_fraction_vs_sigma8.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    im = plt.imshow(heat, aspect="auto", origin="lower")
    plt.colorbar(im, label="invalid fraction")
    plt.xticks(np.arange(len(sigma8s)), [f"{v:.1f}" for v in sigma8s])
    plt.yticks(np.arange(len(zs)), [f"{v:g}" for v in zs])
    plt.xlabel("sigma8")
    plt.ylabel("z")
    plt.title("Invalid Fraction Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "invalid_fraction_heatmap.png"), dpi=200)
    plt.close()

    return heat


def plot_detA_mu(detA_all: np.ndarray, mu_all: np.ndarray, lnmu_all: np.ndarray, outdir: str):
    finite_detA = detA_all[np.isfinite(detA_all)]
    plt.figure(figsize=(8, 5))
    plt.hist(finite_detA, bins=200, log=True)
    plt.axvline(0.0, color="red", linestyle="--", label="detA = 0")
    plt.xlabel("detA")
    plt.ylabel("count")
    plt.title("detA Histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "detA_histogram.png"), dpi=200)
    plt.close()

    near = finite_detA[(finite_detA > -0.05) & (finite_detA < 0.05)]
    plt.figure(figsize=(8, 5))
    plt.hist(near, bins=200)
    plt.axvline(0.0, color="red", linestyle="--", label="detA = 0")
    plt.xlabel("detA")
    plt.ylabel("count")
    plt.title("detA Near-Zero Zoom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "detA_near_zero_zoom.png"), dpi=200)
    plt.close()

    abs_detA = np.abs(finite_detA)
    abs_detA = abs_detA[np.isfinite(abs_detA)]
    abs_detA_sorted = np.sort(abs_detA)
    cdf = np.arange(1, len(abs_detA_sorted) + 1) / len(abs_detA_sorted)
    plt.figure(figsize=(8, 5))
    plt.plot(abs_detA_sorted, cdf)
    plt.xscale("log")
    plt.xlabel("|detA|")
    plt.ylabel("CDF")
    plt.title("CDF of |detA| (near criticality indicator)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "detA_cdf_abs.png"), dpi=200)
    plt.close()

    finite_mu = mu_all[np.isfinite(mu_all)]
    pos_mu = finite_mu[finite_mu > 0]
    plt.figure(figsize=(8, 5))
    plt.hist(pos_mu, bins=200, density=True, log=True)
    plt.xlabel("mu")
    plt.ylabel("PDF")
    plt.title("mu PDF (positive finite)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mu_pdf.png"), dpi=200)
    plt.close()

    valid_lnmu = lnmu_all[np.isfinite(lnmu_all)]
    plt.figure(figsize=(8, 5))
    plt.hist(valid_lnmu, bins=200, density=True, log=True)
    plt.xlabel("ln(mu)")
    plt.ylabel("PDF")
    plt.title("ln(mu) PDF (valid)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lnmu_pdf.png"), dpi=200)
    plt.close()

    sorted_mu = np.sort(pos_mu)
    surv = 1.0 - (np.arange(1, len(sorted_mu) + 1) / len(sorted_mu))
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_mu, np.maximum(surv, 1e-12))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("mu")
    plt.ylabel("1-CDF")
    plt.title("mu Tail Survival")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mu_tail_survival.png"), dpi=200)
    plt.close()


def write_report(results: List[ConfigResult], heat: np.ndarray, zs: np.ndarray, sigma8s: np.ndarray, out_md: str):
    total_samples = sum(int(r.stats["total_samples"]) for r in results)
    total_invalid = sum(int(r.stats["invalid_samples"]) for r in results)
    total_negative_detA = sum(int(r.stats["negative_detA"]) for r in results)
    total_nonfinite_mu = sum(int(r.stats["nonfinite_mu"]) for r in results)
    total_negative_mu = sum(int(r.stats["negative_mu"]) for r in results)
    total_overflow_mu = sum(int(r.stats["overflow_mu"]) for r in results)

    global_invalid_frac = total_invalid / max(1, total_samples)
    global_negative_detA_frac = total_negative_detA / max(1, total_samples)

    row_means = np.nanmean(heat, axis=1)
    col_means = np.nanmean(heat, axis=0)
    z_trend = "increasing" if row_means[-1] > row_means[0] else "flat_or_decreasing"
    s8_trend = "increasing" if col_means[-1] > col_means[0] else "flat_or_decreasing"

    validity_points = []
    for r in results:
        if r.invalid_fraction < 0.01:
            validity_points.append((r.z, r.sigma8, r.invalid_fraction))

    lines = [
        "# Invalid Sample Analysis",
        "",
        "## Summary Statistics",
        f"- Total samples scanned: {total_samples}",
        f"- Total invalid samples: {total_invalid}",
        f"- Global invalid fraction: {global_invalid_frac:.6f}",
        f"- detA <= 0 count: {total_negative_detA}",
        f"- detA <= 0 fraction: {global_negative_detA_frac:.6f}",
        f"- nonfinite mu count: {total_nonfinite_mu}",
        f"- mu <= 0 count: {total_negative_mu}",
        f"- overflow mu count: {total_overflow_mu}",
        "",
        "## Parameter Dependence",
        f"- Invalid fraction vs z trend: **{z_trend}**",
        f"- Invalid fraction vs sigma8 trend: **{s8_trend}**",
        "",
        "## detA Diagnostics",
        "- See figures: `detA_histogram.png`, `detA_near_zero_zoom.png`, `detA_cdf_abs.png`.",
        "- detA <= 0 corresponds to critical-curve crossing / strong-lensing regime where mu diverges or flips sign.",
        "",
        "## Interpretation",
        "- If invalid fractions increase with both z and sigma8, this supports a **physical-tail origin** (nonlinear structure and high-path-length lensing).",
        "- nonfinite/overflow mu events should mostly coincide with detA near 0, consistent with magnification divergence.",
        "- Persistent NaN kappa/gamma without detA crowding near 0 would suggest numerical pathologies.",
        "",
        "## Validity Domain Estimate",
        "- Practical weak-lensing-safe region criterion used: invalid fraction < 1%.",
    ]

    if validity_points:
        lines.append("- Grid points passing criterion:")
        for z, s8, finv in validity_points:
            lines.append(f"  - z={z:g}, sigma8={s8:.1f}, f_invalid={finv:.4f}")
    else:
        lines.append("- No scanned grid points reached invalid fraction < 1% under current physics settings.")

    lines.extend([
        "",
        "## Recommendation",
        "- For Phase 2 baseline emulator, model only valid ln(mu) samples and explicitly report invalid/filtered fraction.",
        "- Keep strong-lensing-like events (detA <= 0) out of the first weak-lensing emulator scope, and treat them as separate regime in future work.",
    ])

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--strict_weak_lensing", action="store_true")
    parser.add_argument("--output_dir", type=str, default="plots/figures/invalid_diagnostics")
    parser.add_argument("--report", type=str, default="docs/reference/invalid_sample_analysis.md")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    zs = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    sigma8s = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4], dtype=float)

    results: List[ConfigResult] = []
    detA_all = []
    mu_all = []
    lnmu_all = []

    idx = 0
    for z in zs:
        for s8 in sigma8s:
            seed = args.seed + idx
            idx += 1
            res = run_one(z, args.h, args.OmegaM, s8, args.nsamples, seed, args.strict_weak_lensing)
            results.append(res)

            detA, mu, lnmu = collect_detA_mu(z, args.h, args.OmegaM, s8, args.nsamples, seed)
            detA_all.append(detA)
            mu_all.append(mu)
            lnmu_all.append(lnmu)

            print(
                f"z={z:4.1f} sigma8={s8:3.1f} f_invalid={res.invalid_fraction:.4f} "
                f"neg_detA={int(res.stats['negative_detA'])} overflow_mu={int(res.stats['overflow_mu'])}"
            )

    detA_all = np.concatenate(detA_all) if detA_all else np.array([])
    mu_all = np.concatenate(mu_all) if mu_all else np.array([])
    lnmu_all = np.concatenate(lnmu_all) if lnmu_all else np.array([])

    heat = plot_invalid_fraction(results, zs, sigma8s, args.output_dir)
    plot_detA_mu(detA_all, mu_all, lnmu_all, args.output_dir)
    write_report(results, heat, zs, sigma8s, args.report)

    print(f"Saved diagnostics to {args.output_dir}")
    print(f"Saved report to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
