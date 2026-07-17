import os
import json
import time
import argparse
from pathlib import Path
import numpy as np

from ml.phase3_common import (
    load_npz_with_metadata,
    save_npz_with_metadata,
    load_nsf_model,
    PRIOR_BOUNDS,
    file_sha256,
    optional_file_sha256,
    simulator_git_commit,
    stable_json_hash,
)
from ml.generate_mock_catalogs import build_catalog
from ml.run_phase3_posterior_grid import run_grid, load_bin_edges
from ml.phase3b_diagnostics import (
    summarize_grid_metrics,
    plot_overlay,
)
from ml.posterior_scaling_metrics import plot_scaling_curves, fit_power_law, power_law

RESULTS_DIR = Path("data/results")
DOCS_DIR = Path("docs")
PLOTS_DIR = Path("plots/figures")

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3D posterior statistics scaling study.")
    parser.add_argument("--nsim_per_z", type=int, default=2000, help="Simulator sample count per redshift block")
    parser.add_argument("--resolution", type=int, default=12, help="Posterior grid resolution")
    parser.add_argument("--output_dir", default="data/results/phase3d_grids", help="Output directory for posterior grids")
    parser.add_argument("--overwrite_cache", action="store_true", help="Force recalculation of grids")
    parser.add_argument("--quick", action="store_true", help="Run a quick minimal sweep for testing")
    args = parser.parse_args()
    
    grid_dir = Path(args.output_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Study parameters
    sizes = [250, 1000] if args.quick else [250, 1000, 5000]
    catalog_types = ["mixed_uniform", "mixed_low_z_dominated", "mixed_high_z_dominated"]
    seeds = [710001, 710002] if args.quick else [710001, 710002, 710003]
    
    results_by_type = {cat_type: {size: [] for size in sizes} for cat_type in catalog_types}
    all_runs = []
    
    print(f"Starting Phase 3D scaling sweep (sizes={sizes}, types={catalog_types}, seeds={seeds})...")
    
    for cat_type in catalog_types:
        for size in sizes:
            for seed in seeds:
                print(f"\n--- Running: type={cat_type}, N={size}, seed={seed} ---")
                t0 = time.time()
                
                # A. Generate mock catalog
                catalog_path = build_catalog(
                    catalog_type=cat_type,
                    n=size,
                    seed=seed,
                    cosmology_id="central",
                    output_dir="data/mock_catalogs/phase3d"
                )
                
                # B. Run grid posteriors
                grid_path = run_grid(
                    catalog_path=catalog_path,
                    grid_type="2d",
                    resolution=args.resolution,
                    output_dir=grid_dir,
                    use_cache=True,
                    overwrite_cache=args.overwrite_cache,
                    nsim_per_z=args.nsim_per_z,
                    likelihood_mode="simulator_compatible"
                )
                
                # C. Extract metrics
                metrics = summarize_grid_metrics(grid_path)
                
                # Mean shifts and widths
                omega_m_shift = float(metrics["nsf_summary"]["OmegaM"]["mean"] - metrics["sim_summary"]["OmegaM"]["mean"])
                sigma8_shift = float(metrics["nsf_summary"]["sigma8"]["mean"] - metrics["sim_summary"]["sigma8"]["mean"])
                
                run_record = {
                    "catalog_type": cat_type,
                    "N": size,
                    "seed": seed,
                    "catalog_path": str(catalog_path),
                    "grid_path": str(grid_path),
                    "jsd": float(metrics["posterior_jsd"]),
                    "tv": float(metrics["total_variation"]),
                    "overlap_68": float(metrics["credible_region_overlap"]["credible_region_68"]),
                    "overlap_95": float(metrics["credible_region_overlap"]["credible_region_95"]),
                    "mle_offset": {
                        "OmegaM": float(metrics["mle_offset"]["OmegaM"]),
                        "sigma8": float(metrics["mle_offset"]["sigma8"]),
                    },
                    "mean_shift": {
                        "OmegaM": omega_m_shift,
                        "sigma8": sigma8_shift,
                    },
                    "nsf_width": {
                        "OmegaM": float(metrics["nsf_summary"]["OmegaM"]["std"]),
                        "sigma8": float(metrics["nsf_summary"]["sigma8"]["std"]),
                    },
                    "sim_width": {
                        "OmegaM": float(metrics["sim_summary"]["OmegaM"]["std"]),
                        "sigma8": float(metrics["sim_summary"]["sigma8"]["std"]),
                    },
                    "duration_seconds": time.time() - t0
                }
                
                results_by_type[cat_type][size].append(run_record)
                all_runs.append(run_record)
                
                # Generate contours overlay plot
                plot_dir = PLOTS_DIR / f"phase3d_scaling/{cat_type}"
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"N{size}_seed{seed}_contours.png"
                plot_overlay(grid_path, plot_path, f"{cat_type.replace('_', ' ').title()} N={size} (seed={seed})")
                
    # 2. Write main JSON results
    with open(RESULTS_DIR / "phase3d_scaling_results.json", "w") as f:
        json.dump(all_runs, f, indent=2)
        
    # 3. Generate Scaling Curves Plots
    plot_scaling_curves(results_by_type, PLOTS_DIR / "phase3d_scaling")
    
    # 4. Redshift scaling decomposition
    redshift_results = {
        "mixed_uniform": results_by_type["mixed_uniform"],
        "mixed_low_z_dominated": results_by_type["mixed_low_z_dominated"],
        "mixed_high_z_dominated": results_by_type["mixed_high_z_dominated"]
    }
    with open(RESULTS_DIR / "phase3d_redshift_scaling_results.json", "w") as f:
        json.dump(redshift_results, f, indent=2)
        
    # Generate redshift specific plots under phase3d_redshift_scaling/
    plot_scaling_curves(results_by_type, PLOTS_DIR / "phase3d_redshift_scaling")
    
    # 5. Fit trends and write reports
    write_scaling_reports(results_by_type, sizes)
    
    # 6. Cache and Reproducibility Verification
    verify_cache_and_reproducibility(all_runs, args.nsim_per_z)
    
    print("\nPhase 3D scaling runner execution complete!")

def write_scaling_reports(results_by_type, sizes):
    # Fit trends for each catalog type
    fit_summary = {}
    
    for cat_type, data in results_by_type.items():
        N_vals = sorted(data.keys())
        jsd_means = []
        tv_means = []
        for N in N_vals:
            jsd_means.append(np.mean([r["jsd"] for r in data[N]]))
            tv_means.append(np.mean([r["tv"] for r in data[N]]))
            
        jsd_popt = fit_power_law(N_vals, jsd_means)
        tv_popt = fit_power_law(N_vals, tv_means)
        
        fit_summary[cat_type] = {
            "jsd": {"A": jsd_popt[0], "alpha": jsd_popt[1], "B": jsd_popt[2]},
            "tv": {"A": tv_popt[0], "alpha": tv_popt[1], "B": tv_popt[2]},
        }
        
    # Write docs/phase3/phase3d_scaling_metrics.md
    metrics_lines = [
        "# Phase 3D — Posterior Scaling Metrics Report",
        "",
        "This report fits convergence trends to evaluate whether posterior discrepancy vanishes asymptotically ($B \\to 0$).",
        "",
        "## Convergence Fit Parameters",
        "",
        "Fit model: $\\text{Metric}(N) = A \\cdot N^{-\\alpha} + B$",
        "",
        "| Catalog Type | Metric | Scale A | Decay Power $\\alpha$ | Asymptotic Floor B | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for cat_type in fit_summary:
        for metric in ["jsd", "tv"]:
            rec = fit_summary[cat_type][metric]
            status = "CONVERGENT" if rec["B"] < 0.05 else "SATURATED"
            metrics_lines.append(
                f"| `{cat_type}` | {metric.upper()} | {rec['A']:.4f} | {rec['alpha']:.4f} | {rec['B']:.4f} | {status} |"
            )
            
    metrics_lines.extend([
        "",
        "## Interpretation",
        "If the asymptotic floor $B$ is close to 0, the mismatch is statistically dominated. If $B$ remains large, the mismatch is systematic and persists even at high statistics."
    ])
    with open(DOCS_DIR / "phase3d_scaling_metrics.md", "w") as f:
        f.write("\n".join(metrics_lines) + "\n")
        
    # Write docs/phase3/phase3d_scaling_report.md
    report_lines = [
        "# Phase 3D — Posterior Statistics Scaling Report",
        "",
        "Summary of posterior metrics across increasing event counts.",
        "",
        "## Performance Table",
        "",
        "| Catalog Type | N | Mean JSD | Mean TV | Mean 68% Overlap | Mean 95% Overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat_type in results_by_type:
        for size in sizes:
            runs = results_by_type[cat_type][size]
            mean_jsd = np.mean([r["jsd"] for r in runs])
            mean_tv = np.mean([r["tv"] for r in runs])
            mean_68 = np.mean([r["overlap_68"] for r in runs])
            mean_95 = np.mean([r["overlap_95"] for r in runs])
            report_lines.append(
                f"| `{cat_type}` | {size} | {mean_jsd:.6f} | {mean_tv:.6f} | {mean_68:.3f} | {mean_95:.3f} |"
            )
            
    report_lines.extend([
        "",
        "## Scaling Curves",
        "![JSD Scaling](figures/phase3d_scaling/jsd_scaling.png)",
        "![TV Scaling](figures/phase3d_scaling/tv_scaling.png)",
    ])
    with open(DOCS_DIR / "phase3d_scaling_report.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")
        
    # Write docs/phase3/phase3d_redshift_scaling_decomposition.md
    decomp_lines = [
        "# Phase 3D — Low-z vs Mixed-catalog Scaling Decomposition",
        "",
        "Comparison of redshift-dominated catalog scaling behaviors.",
        "",
        "## Redshift scaling Comparison",
        "",
        "| Catalog Type | N | Mean JSD | Mean TV | Width OmegaM | MAP offset OmegaM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat_type in results_by_type:
        for size in sizes:
            runs = results_by_type[cat_type][size]
            mean_jsd = np.mean([r["jsd"] for r in runs])
            mean_tv = np.mean([r["tv"] for r in runs])
            mean_width = np.mean([r["nsf_width"]["OmegaM"] for r in runs])
            mean_mle_off = np.mean([r["mle_offset"]["OmegaM"] for r in runs])
            decomp_lines.append(
                f"| `{cat_type}` | {size} | {mean_jsd:.6f} | {mean_tv:.6f} | {mean_width:.4f} | {mean_mle_off:+.4f} |"
            )
            
    decomp_lines.extend([
        "",
        "## Figures",
        "![JSD Redshift Scaling](figures/phase3d_redshift_scaling/jsd_scaling.png)",
        "![TV Redshift Scaling](figures/phase3d_redshift_scaling/tv_scaling.png)",
    ])
    with open(DOCS_DIR / "phase3d_redshift_scaling_decomposition.md", "w") as f:
        f.write("\n".join(decomp_lines) + "\n")

def verify_cache_and_reproducibility(all_runs, nsim_per_z):
    # Verify metadata checks
    cache_repro_results = {
        "phase": "3D",
        "objective": "cache_reproducibility",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "success",
        "checked_runs": []
    }
    
    nsf_model_path = "data/models/conditional_nsf_backend_current.pt"
    preprocessing_stats_path = "data/models/nsf_preprocessing_stats.json"
    
    for run in all_runs:
        grid_path = run["grid_path"]
        _, md = load_npz_with_metadata(grid_path)
        
        # Verify hashes
        catalog_ok = md["catalog_hash"] == load_npz_with_metadata(run["catalog_path"])[1]["catalog_hash"]
        nsf_ok = md["nsf_checkpoint_hash"] == file_sha256(nsf_model_path)
        preproc_ok = md["preprocessing_stats_hash"] == optional_file_sha256(preprocessing_stats_path)
        commit_ok = md["simulator_git_commit"] == simulator_git_commit()
        mode_ok = md.get("likelihood_mode") == "simulator_compatible"
        
        check_record = {
            "catalog_type": run["catalog_type"],
            "N": run["N"],
            "seed": run["seed"],
            "cache_verifications": {
                "catalog_hash_matches": bool(catalog_ok),
                "nsf_checkpoint_hash_matches": bool(nsf_ok),
                "preprocessing_stats_hash_matches": bool(preproc_ok),
                "simulator_git_commit_matches": bool(commit_ok),
                "likelihood_mode_matches": bool(mode_ok)
            }
        }
        cache_repro_results["checked_runs"].append(check_record)
        
    with open(RESULTS_DIR / "phase3d_cache_reproducibility.json", "w") as f:
        json.dump(cache_repro_results, f, indent=2)
        
    # Write report
    report_lines = [
        "# Phase 3D — Cache and Reproducibility Report",
        "",
        "This report verifies cache keys and metadata sanity across the scaling sweep.",
        "",
        "## Cache Verifications Summary",
        "",
        "| Catalog | N | Seed | Catalog Hash Match | NSF Hash Match | Likelihood Mode Match | Git Commit Match |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for r in cache_repro_results["checked_runs"]:
        v = r["cache_verifications"]
        report_lines.append(
            f"| `{r['catalog_type']}` | {r['N']} | {r['seed']} | `{v['catalog_hash_matches']}` | "
            f"`{v['nsf_checkpoint_hash_matches']}` | `{v['likelihood_mode_matches']}` | `{v['simulator_git_commit_matches']}` |"
        )
        
    with open(DOCS_DIR / "phase3d_cache_reproducibility.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

if __name__ == "__main__":
    main()
