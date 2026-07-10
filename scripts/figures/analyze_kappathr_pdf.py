#!/usr/bin/env python3
"""Acceptance analysis for adaptive versus flat kappa threshold rules.

This is the reproducible runner for tmp/pdf_compare_spec.md. It writes the raw
samples, machine-readable metrics, a compact comparison figure, and a Markdown
verdict. Subhalos are disabled so that only the threshold rule changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


RULES = {
    "adaptive": -1.0,
    "flat_1e-4": 1.0e-4,
    "flat_1e-3": 1.0e-3,
}
QUANTILES = (0.99, 0.999, 0.9999)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-low", type=int, default=1_000_000,
                        help="Requested draws at z=0.2 and z=1 (default: 1e6).")
    parser.add_argument("--n-high", type=int, default=300_000,
                        help="Requested draws at z=10 (default: 3e5).")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--nbins", type=int, default=400)
    parser.add_argument("--kl-replicates", type=int, default=400)
    parser.add_argument("--reuse-samples", action="store_true",
                        help="Recompute metrics/plots from the existing raw-sample NPZ.")
    return parser.parse_args()


def histogram_prob(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    return counts.astype(float)


def kl_from_counts(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> float:
    """KL(a || b) with Jeffreys pseudocount smoothing."""
    pa = (a + alpha) / (np.sum(a) + alpha * a.size)
    pb = (b + alpha) / (np.sum(b) + alpha * b.size)
    return float(np.sum(pa * np.log(pa / pb)))


def kl_sampling_floor(counts: np.ndarray, n_a: int, n_b: int, nrep: int,
                      rng: np.random.Generator) -> tuple[float, float, float]:
    """Null KL distribution from two draws of the same histogram distribution."""
    probability = (counts + 0.5) / (np.sum(counts) + 0.5 * counts.size)
    values = np.empty(nrep)
    for i in range(nrep):
        ca = rng.multinomial(n_a, probability)
        cb = rng.multinomial(n_b, probability)
        values[i] = kl_from_counts(ca, cb)
    return tuple(float(v) for v in np.quantile(values, (0.5, 0.025, 0.975)))


def quantile_ci_sorted(sorted_x: np.ndarray, probability: float,
                       confidence: float = 0.95) -> tuple[float, float, float]:
    """Distribution-free point estimate and order-statistic confidence interval."""
    n = sorted_x.size
    point = float(np.quantile(sorted_x, probability))
    alpha = 1.0 - confidence
    lower_rank = int(binom.ppf(alpha / 2.0, n, probability))
    upper_rank = int(binom.ppf(1.0 - alpha / 2.0, n, probability)) + 1
    lower_index = max(0, min(n - 1, lower_rank - 1))
    upper_index = max(0, min(n - 1, upper_rank - 1))
    return point, float(sorted_x[lower_index]), float(sorted_x[upper_index])


def sigma_se(x: np.ndarray) -> float:
    """Large-sample standard error of the population-standard-deviation estimate."""
    n = x.size
    centered = x - np.mean(x)
    variance = float(np.mean(centered * centered))
    fourth = float(np.mean(centered ** 4))
    variance_of_variance = max(0.0, (fourth - variance * variance) / n)
    return math.sqrt(variance_of_variance) / (2.0 * math.sqrt(variance))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "build"))
    import gwlensing as gw  # pylint: disable=import-error,import-outside-toplevel

    out_dir = root / "data" / "results"
    plot_dir = root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "kappathr_pdf_acceptance_samples.npz"
    csv_path = out_dir / "kappathr_pdf_acceptance_metrics.csv"
    json_path = out_dir / "kappathr_pdf_acceptance_summary.json"
    report_path = out_dir / "kappathr_pdf_acceptance_report.md"
    png_path = plot_dir / "kappathr_pdf_acceptance.png"
    pdf_path = plot_dir / "kappathr_pdf_acceptance.pdf"

    sample_map: dict[tuple[float, str], np.ndarray] = {}
    invalid_map: dict[tuple[float, str], dict] = {}
    timings: dict[tuple[float, str], float] = {}
    requested = {0.2: args.n_low, 1.0: args.n_low, 10.0: args.n_high}
    common = dict(h=0.674, OmegaM=0.315, sigma8=0.811, seed=args.seed,
                  strict_weak_lensing=False, subhalo=False)
    archive = np.load(raw_path) if args.reuse_samples else None
    previous = json.loads(json_path.read_text(encoding="utf-8")) if args.reuse_samples else None

    for z in (0.2, 1.0, 10.0):
        for label, threshold in RULES.items():
            if args.reuse_samples:
                key = f"z{str(z).replace('.', 'p')}_{label}"
                values = np.asarray(archive[key], dtype=np.float64)
                old = previous["results"][str(z)]["rules"][label]
                invalid_stats = {"invalid_samples": old["invalid"]}
                elapsed = old["runtime_seconds"]
            else:
                start = time.monotonic()
                result = gw.sample_lnmu_ml_with_diagnostics(
                    z=z, nsamples=requested[z], kappathr_flat=threshold, **common)
                values = np.asarray(result["lnmu"], dtype=np.float64)
                invalid_stats = dict(result["invalid_stats"])
                elapsed = time.monotonic() - start
            sample_map[(z, label)] = values
            invalid_map[(z, label)] = invalid_stats
            timings[(z, label)] = elapsed
            print(f"z={z:g} {label}: {values.size:,}/{requested[z]:,} valid "
                  f"in {timings[(z, label)]:.1f} s", flush=True)

    save_items = {
        f"z{str(z).replace('.', 'p')}_{label}": values
        for (z, label), values in sample_map.items()
    }
    np.savez_compressed(raw_path, **save_items)

    rows: list[dict] = []
    per_z: dict[str, dict] = {}
    rng = np.random.default_rng(args.seed + 8192)
    for z in (0.2, 1.0, 10.0):
        arrays = [sample_map[(z, label)] for label in RULES]
        pooled = np.concatenate(arrays)
        lo, hi = np.quantile(pooled, (1.0e-5, 1.0 - 1.0e-5))
        edges = np.linspace(lo, hi, args.nbins + 1)
        counts = {label: histogram_prob(sample_map[(z, label)], edges) for label in RULES}
        adaptive_counts = counts["adaptive"]
        floor_median, floor_lo, floor_hi = kl_sampling_floor(
            adaptive_counts, sample_map[(z, "adaptive")].size,
            sample_map[(z, "flat_1e-4")].size, args.kl_replicates, rng)

        z_result = {
            "histogram_range_lnmu": [float(lo), float(hi)],
            "kl_null_median": floor_median,
            "kl_null_95_ci": [floor_lo, floor_hi],
            "rules": {},
        }
        adaptive = sample_map[(z, "adaptive")]
        adaptive_sigma = float(np.std(adaptive))
        adaptive_sigma_se = sigma_se(adaptive)

        for label in RULES:
            lnmu = sample_map[(z, label)]
            mu = np.exp(lnmu)
            sorted_mu = np.sort(mu)
            inv_mu = np.exp(-lnmu)
            sigma = float(np.std(lnmu))
            sigma_error = sigma_se(lnmu)
            kl = 0.0 if label == "adaptive" else kl_from_counts(counts[label], adaptive_counts)
            tail = {}
            for probability in QUANTILES:
                point, ci_lo, ci_hi = quantile_ci_sorted(sorted_mu, probability)
                tail[f"q{100 * probability:g}"] = {
                    "value": point, "ci95": [ci_lo, ci_hi],
                }

            sigma_relative = sigma / adaptive_sigma - 1.0
            sigma_relative_se = math.sqrt(
                (sigma_error / adaptive_sigma) ** 2
                + (sigma * adaptive_sigma_se / adaptive_sigma ** 2) ** 2)
            rule_result = {
                "requested": requested[z],
                "valid": int(lnmu.size),
                "invalid": int(invalid_map[(z, label)]["invalid_samples"]),
                "runtime_seconds": timings[(z, label)],
                "mean_inverse_mu": float(np.mean(inv_mu)),
                "mean_inverse_mu_se": float(np.std(inv_mu) / math.sqrt(inv_mu.size)),
                "sigma_lnmu": sigma,
                "sigma_lnmu_se": sigma_error,
                "sigma_relative_to_adaptive": sigma_relative,
                "sigma_relative_se": sigma_relative_se,
                "kl_to_adaptive": kl,
                "tail": tail,
            }
            z_result["rules"][label] = rule_result

            row = {
                "z": z, "rule": label, "kappathr_flat": RULES[label],
                **{k: rule_result[k] for k in (
                    "requested", "valid", "invalid", "runtime_seconds",
                    "mean_inverse_mu", "mean_inverse_mu_se", "sigma_lnmu",
                    "sigma_lnmu_se", "sigma_relative_to_adaptive",
                    "sigma_relative_se", "kl_to_adaptive")},
                "kl_null_median": floor_median,
            }
            for q_label, q_values in tail.items():
                row[q_label] = q_values["value"]
                row[f"{q_label}_ci_lo"] = q_values["ci95"][0]
                row[f"{q_label}_ci_hi"] = q_values["ci95"][1]
            rows.append(row)
        per_z[str(z)] = z_result

    checks = {}
    for z in (0.2, 1.0, 10.0):
        z_result = per_z[str(z)]
        adaptive_result = z_result["rules"]["adaptive"]
        chosen = z_result["rules"]["flat_1e-4"]
        tail_overlap = {}
        for q_label in ("q99", "q99.9", "q99.99"):
            tail_overlap[q_label] = intervals_overlap(
                tuple(adaptive_result["tail"][q_label]["ci95"]),
                tuple(chosen["tail"][q_label]["ci95"]))
        checks[str(z)] = {
            "flux_all_rules": all(
                abs(z_result["rules"][label]["mean_inverse_mu"] - 1.0) <= 0.003
                for label in RULES),
            "sigma_within_0p5_percent": abs(chosen["sigma_relative_to_adaptive"]) <= 0.005,
            "kl_below_1e-3": chosen["kl_to_adaptive"] <= 1.0e-3,
            "tail_ci_overlap": tail_overlap,
        }
        checks[str(z)]["pass"] = (
            checks[str(z)]["flux_all_rules"]
            and checks[str(z)]["sigma_within_0p5_percent"]
            and checks[str(z)]["kl_below_1e-3"]
            and all(tail_overlap.values()))

    summary = {
        "spec": "tmp/pdf_compare_spec.md",
        "cosmology": {"OmegaM": 0.315, "sigma8": 0.811, "h": 0.674},
        "seed": args.seed,
        "subhalo": False,
        "histogram_bins": args.nbins,
        "kl_definition": "KL(flat || adaptive), Jeffreys 0.5-count smoothing",
        "results": per_z,
        "checks": checks,
        "overall_pass": all(value["pass"] for value in checks.values()),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"adaptive": "#2468a2", "flat_1e-4": "#cf4b3e", "flat_1e-3": "#3c8d64"}
    labels = {"adaptive": "adaptive", "flat_1e-4": "flat 1e-4", "flat_1e-3": "flat 1e-3"}
    plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.22})
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.0))
    for row_index, z in enumerate((0.2, 1.0, 10.0)):
        ax_pdf, ax_tail, ax_quant = axes[row_index]
        pooled = np.concatenate([sample_map[(z, label)] for label in RULES])
        lo, hi = np.quantile(pooled, (1.0e-4, 1.0 - 1.0e-4))
        edges = np.linspace(lo, hi, 180)
        for label in RULES:
            lnmu = sample_map[(z, label)]
            ax_pdf.hist(lnmu, bins=edges, density=True, histtype="step", lw=1.35,
                        color=colors[label], label=labels[label])
            mu_sorted = np.sort(np.exp(lnmu))
            start = int(0.98 * mu_sorted.size)
            survival = (mu_sorted.size - np.arange(start, mu_sorted.size)) / mu_sorted.size
            ax_tail.plot(mu_sorted[start:], survival, lw=1.35, color=colors[label],
                         label=labels[label])
        ax_pdf.set_yscale("log")
        ax_pdf.set_xlabel("ln(mu)")
        ax_pdf.set_ylabel("density")
        ax_pdf.set_title(f"z_s = {z:g}: PDF")
        ax_pdf.legend(frameon=False, fontsize=8)
        ax_tail.set_yscale("log")
        ax_tail.set_xscale("log")
        ax_tail.set_xlabel("mu")
        ax_tail.set_ylabel("P(Mu > mu)")
        ax_tail.set_title("upper-tail survival")

        q_names = ("q99", "q99.9", "q99.99")
        x_positions = np.arange(3)
        reference = per_z[str(z)]["rules"]["adaptive"]["tail"]
        for offset, label in ((-0.08, "flat_1e-4"), (0.08, "flat_1e-3")):
            ratios, low, high = [], [], []
            for q_name in q_names:
                values = per_z[str(z)]["rules"][label]["tail"][q_name]
                ref_value = reference[q_name]["value"]
                ratios.append(values["value"] / ref_value)
                low.append((values["value"] - values["ci95"][0]) / ref_value)
                high.append((values["ci95"][1] - values["value"]) / ref_value)
            ax_quant.errorbar(x_positions + offset, ratios, yerr=[low, high], fmt="o",
                              ms=4, capsize=2, color=colors[label], label=labels[label])
        ax_quant.axhline(1.0, color=colors["adaptive"], lw=1.0, ls="--")
        ax_quant.set_xticks(x_positions, ("99", "99.9", "99.99"))
        ax_quant.set_xlabel("mu percentile")
        ax_quant.set_ylabel("quantile / adaptive")
        ax_quant.set_title("tail ratios (95% CI)")
        ax_quant.legend(frameon=False, fontsize=8)
    fig.suptitle("Magnification PDF acceptance: adaptive vs flat kappa threshold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    lines = [
        "# Flat vs adaptive kappa-threshold PDF acceptance",
        "",
        f"**Verdict: {'PASS' if summary['overall_pass'] else 'FAIL'}** for flat `kappathr_flat=1e-4`.",
        "",
        "Subhalos were disabled; all runs used seed 123 and the default cosmology "
        "(OmegaM=0.315, sigma8=0.811, h=0.674). Confidence intervals on tail "
        "quantiles are distribution-free 95% order-statistic intervals.",
        "",
        "| z_s | rule | valid | <1/mu> | sigma(lnmu) | delta sigma | KL to adaptive | q99 | q99.9 | q99.99 |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['z']:g} | {row['rule']} | {row['valid']:,} | "
            f"{row['mean_inverse_mu']:.6f} | {row['sigma_lnmu']:.6g} | "
            f"{100 * row['sigma_relative_to_adaptive']:+.3f}% | "
            f"{row['kl_to_adaptive']:.3g} | {row['q99']:.6g} | "
            f"{row['q99.9']:.6g} | {row['q99.99']:.6g} |")
    lines.extend(["", "## Acceptance checks", ""])
    for z, result in checks.items():
        overlap_text = ", ".join(
            f"{name}={'yes' if value else 'no'}" for name, value in result["tail_ci_overlap"].items())
        lines.append(
            f"- z_s={z}: **{'PASS' if result['pass'] else 'FAIL'}**; "
            f"flux={'pass' if result['flux_all_rules'] else 'fail'}, "
            f"sigma={'pass' if result['sigma_within_0p5_percent'] else 'fail'}, "
            f"KL={'pass' if result['kl_below_1e-3'] else 'fail'}, tail overlap: {overlap_text}.")
    lines.extend([
        "", "The same-distribution KL null medians (from multinomial replicas of the "
        "adaptive histogram) are recorded in the JSON and CSV outputs. The acceptance "
        "KL is `KL(flat || adaptive)` with 400 shared ln(mu) bins and a Jeffreys "
        "0.5-count pseudocount.", "",
        "## Raw-kappa diagnostic", "",
        "The z_s=10 flux-conservation sanity check fails for all three rules, including "
        "adaptive, so it is a common high-redshift sampler issue rather than evidence specific "
        "to the threshold switch. The KL, sigma, and q99 comparisons independently fail there.", "",
        "A separate 200,000-draw raw-kappa check (`scripts/figures/check_kappathr_raw_kappa.py`) "
        "shows that untrimmed second moments are tail-noisy, but it does not rescue the "
        "literal pure-split premise. With default bias enabled, the flat-1e-4 raw sigma_kappa "
        "point differences relative to adaptive are +7.32%, +4.09%, and +2.39% at "
        "z_s=0.2, 1, and 10. After retaining only centered |kappa| < 1, the corresponding "
        "differences are +7.32%, +0.11%, and +1.09%.", "",
        "The live implementation uses the threshold-dependent encounter table for both the "
        "lognormal bias width and filament count/radius (`cpp/lensing.cpp`, around lines "
        "401-402 and 434-439). Only the NFW sub-threshold host field receives `sigmakappaW`; "
        "filaments below the threshold are not Gaussian-compensated. Therefore changing "
        "kappa_thr is not a pure NFW computational repartition under the full default model, "
        "which is consistent with the observed redshift-dependent PDF gaps.", "",
        "**Decision:** do not lock in flat 1e-4 as science-neutral on this evidence. The "
        "z_s=1 case passes, but the full three-redshift acceptance criterion does not.", "",
        f"Figure: `{png_path.relative_to(root)}`", "",
        f"Raw samples: `{raw_path.relative_to(root)}`", "",
        f"Machine-readable summary: `{json_path.relative_to(root)}`", "",
        "Raw-kappa metrics: `data/results/kappathr_raw_kappa_metrics.csv`", "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
