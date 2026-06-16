import argparse
import json
import time
from pathlib import Path

import numpy as np

from ml.generate_mock_catalogs import build_catalog
from ml.phase3_common import build_grid, ensure_dir, load_npz_with_metadata, load_nsf_model
from ml.run_phase3_posterior_grid import evaluate_simulator_grid, load_bin_edges, nsf_mixed_catalog_log_likelihood


def benchmark(output_json: str | Path = "data/results/phase3_runtime_results.json", n_events: int = 1000, grid_resolution: int = 8) -> dict:
    catalog_path = build_catalog("mixed_uniform", n_events, 530000, "central", output_dir="data/mock_catalogs/phase3/runtime")
    arrays, _ = load_npz_with_metadata(catalog_path)
    z = arrays["z"]
    lnmu = arrays["lnmu"]
    grid2 = build_grid("2d", grid_resolution, fixed_h=0.67)
    grid3 = build_grid("3d", grid_resolution)
    model = load_nsf_model()
    bin_edges = load_bin_edges()

    timings = {}
    t0 = time.time()
    _ = nsf_mixed_catalog_log_likelihood(model, z[:10], lnmu[:10], grid3["theta"][:1], chunk_size=1)
    timings["nsf_single_likelihood_seconds"] = time.time() - t0

    t0 = time.time()
    _ = nsf_mixed_catalog_log_likelihood(model, z, lnmu, grid2["theta"], chunk_size=128)
    timings["nsf_2d_grid_seconds"] = time.time() - t0

    t0 = time.time()
    _ = nsf_mixed_catalog_log_likelihood(model, z, lnmu, grid3["theta"], chunk_size=128)
    timings["nsf_3d_grid_seconds"] = time.time() - t0

    t0 = time.time()
    _ = evaluate_simulator_grid(z, lnmu, grid2["theta"][:1], (1,), bin_edges, nsim_per_z=2000)
    timings["simulator_single_likelihood_seconds"] = time.time() - t0
    timings["simulator_2d_grid_projected_seconds"] = timings["simulator_single_likelihood_seconds"] * grid2["theta"].shape[0]
    timings["simulator_3d_grid_projected_seconds"] = timings["simulator_single_likelihood_seconds"] * grid3["theta"].shape[0]
    timings["coverage_catalog_inference_seconds"] = timings["nsf_3d_grid_seconds"]
    timings["full_posterior_benchmark_projected_seconds"] = timings["simulator_3d_grid_projected_seconds"] + timings["nsf_3d_grid_seconds"]

    speedups = {
        "single_likelihood": timings["simulator_single_likelihood_seconds"] / max(timings["nsf_single_likelihood_seconds"], 1e-12),
        "2d_grid_projected": timings["simulator_2d_grid_projected_seconds"] / max(timings["nsf_2d_grid_seconds"], 1e-12),
        "3d_grid_projected": timings["simulator_3d_grid_projected_seconds"] / max(timings["nsf_3d_grid_seconds"], 1e-12),
    }
    results = {
        "configuration": {"n_events": n_events, "grid_resolution": grid_resolution, "catalog": str(catalog_path)},
        "timings": timings,
        "speedups": speedups,
    }
    ensure_dir(Path(output_json).parent)
    Path(output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def write_report(results: dict, output_report: str | Path) -> None:
    lines = [
        "# Phase 3 Runtime Benchmark",
        "",
        "| Quantity | Seconds |",
        "|---|---:|",
    ]
    for key, val in results["timings"].items():
        lines.append(f"| {key} | {val:.6g} |")
    lines.extend(["", "| Speedup | Factor |", "|---|---:|"])
    for key, val in results["speedups"].items():
        lines.append(f"| {key} | {val:.2f}x |")
    ensure_dir(Path(output_report).parent)
    Path(output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 3 end-to-end inference runtime.")
    parser.add_argument("--output_json", default="data/results/phase3_runtime_results.json")
    parser.add_argument("--output_report", default="docs/phase3_runtime_benchmark.md")
    parser.add_argument("--n_events", type=int, default=1000)
    parser.add_argument("--grid_resolution", type=int, default=8)
    args = parser.parse_args()
    results = benchmark(args.output_json, args.n_events, args.grid_resolution)
    write_report(results, args.output_report)


if __name__ == "__main__":
    main()
