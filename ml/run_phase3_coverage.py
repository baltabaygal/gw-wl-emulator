import argparse
import json
from pathlib import Path

import numpy as np

from ml.generate_mock_catalogs import build_catalog
from ml.phase3_common import TRUE_COSMOLOGIES, build_grid, ensure_dir, load_npz_with_metadata, load_nsf_model, normalize_log_grid
from ml.posterior_metrics import coverage_summary, posterior_summary
from ml.run_phase3_posterior_grid import nsf_mixed_catalog_log_likelihood


def truth_inside(summary: dict, truth: dict[str, float], mass: str) -> dict[str, bool]:
    out = {}
    for name, truth_val in truth.items():
        if name in summary:
            lo, hi = summary[name][f"ci{mass}"]
            out[name] = bool(lo <= truth_val <= hi)
    return out


def run_coverage(
    n_catalogs: int = 20,
    n_events: int = 1000,
    grid_resolution: int = 12,
    catalog_type: str = "mixed_uniform",
    cosmology_id: str = "central",
    output_json: str | Path = "data/results/phase3_coverage_results.json",
    catalog_dir: str | Path = "data/mock_catalogs/phase3/coverage",
    figures_dir: str | Path = "plots/figures/phase3_coverage",
) -> dict:
    ensure_dir(catalog_dir)
    ensure_dir(figures_dir)
    model = load_nsf_model()
    records = []
    grid = build_grid("3d", grid_resolution)
    shape = tuple(len(v) for v in grid["axes"].values())
    for idx in range(n_catalogs):
        seed = 420000 + idx
        catalog_path = build_catalog(catalog_type, n_events, seed, cosmology_id, output_dir=catalog_dir)
        arrays, md = load_npz_with_metadata(catalog_path)
        truth = {"h": md["true_h"], "OmegaM": md["true_OmegaM"], "sigma8": md["true_sigma8"]}
        log_lik = nsf_mixed_catalog_log_likelihood(model, arrays["z"], arrays["lnmu"], grid["theta"], chunk_size=128).reshape(shape)
        posterior = normalize_log_grid(log_lik)
        summary = posterior_summary(grid["axes"], posterior)
        record = {
            "catalog_id": md["catalog_id"],
            "catalog_hash": md["catalog_hash"],
            "seed": seed,
            "truth": truth,
            "nsf_summary": summary,
            "truth_coverage": {
                "nsf_68": truth_inside(summary, truth, "68"),
                "nsf_95": truth_inside(summary, truth, "95"),
            },
            "posterior_mean_bias": {k: float(summary[k]["mean"] - truth[k]) for k in truth},
            "posterior_width": {k: float(summary[k]["std"]) for k in truth},
        }
        records.append(record)
        print(f"Coverage catalog {idx + 1}/{n_catalogs}: {record['truth_coverage']}")

    results = {
        "configuration": {
            "n_catalogs": int(n_catalogs),
            "n_events": int(n_events),
            "grid_resolution": int(grid_resolution),
            "catalog_type": catalog_type,
            "cosmology_id": cosmology_id,
        },
        "records": records,
        "coverage_summary": coverage_summary(records),
    }
    ensure_dir(Path(output_json).parent)
    Path(output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def write_report(results: dict, output_report: str | Path) -> None:
    lines = [
        "# Phase 3 Coverage Report",
        "",
        f"Coverage catalogs: {results['configuration']['n_catalogs']}",
        "",
        "| Parameter | Observed 68% | Binomial sigma | Observed 95% | Binomial sigma |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in results["coverage_summary"].get("parameters", {}).items():
        lines.append(
            f"| {name} | {stats['coverage_68']:.3f} | {stats['binomial_sigma_68']:.3f} | "
            f"{stats['coverage_95']:.3f} | {stats['binomial_sigma_95']:.3f} |"
        )
    lines.append("")
    lines.append("For N=20, deviations of order the listed binomial uncertainty should not be over-interpreted.")
    ensure_dir(Path(output_report).parent)
    Path(output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 NSF coverage study.")
    parser.add_argument("--n_catalogs", type=int, default=20)
    parser.add_argument("--n_events", type=int, default=1000)
    parser.add_argument("--grid_resolution", type=int, default=12)
    parser.add_argument("--catalog_type", default="mixed_uniform")
    parser.add_argument("--cosmology_id", choices=sorted(TRUE_COSMOLOGIES), default="central")
    parser.add_argument("--output_json", default="data/results/phase3_coverage_results.json")
    parser.add_argument("--output_report", default="docs/phase3_coverage_report.md")
    args = parser.parse_args()
    results = run_coverage(
        args.n_catalogs,
        args.n_events,
        args.grid_resolution,
        args.catalog_type,
        args.cosmology_id,
        args.output_json,
    )
    write_report(results, args.output_report)


if __name__ == "__main__":
    main()
