import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ml.generate_mock_catalogs import build_catalog
from ml.phase3_common import (
    LIKELIHOOD_VERSION,
    PRIOR_BOUNDS,
    catalog_hash,
    ensure_dir,
    file_sha256,
    load_npz_with_metadata,
    normalize_log_grid,
    optional_file_sha256,
    save_npz_with_metadata,
    simulator_git_commit,
    utc_timestamp,
)
from ml.posterior_metrics import compare_posteriors, marginalize_to_axes
from ml.run_phase3_posterior_grid import (
    load_bin_edges,
    nsf_mixed_catalog_log_likelihood,
    run_grid,
)


SMOKE_CATALOG = Path("data/mock_catalogs/phase3/mixed_uniform_central_N1000_seed610001_57cb44ad1ef0f9d4.npz")
SMOKE_GRID = Path("data/results/phase3_posterior_grids/combined_2d_mixed_uniform_central_N1000_seed610001_57cb44ad1ef0f9d4_e51caa6b6820.npz")
RESULTS_DIR = Path("data/results")
DOCS_DIR = Path("docs")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_grid_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for key, arr in arrays.items():
        if key.startswith("axis_"):
            checks[key] = {
                "finite": bool(np.isfinite(arr).all()),
                "strictly_increasing": bool(np.all(np.diff(arr) > 0)),
                "size": int(arr.size),
            }
        elif "log_likelihood" in key:
            post = normalize_log_grid(arr)
            checks[key] = {
                "finite": bool(np.isfinite(arr).all()),
                "shape": list(arr.shape),
                "posterior_sum": float(np.sum(post)),
                "posterior_finite": bool(np.isfinite(post).all()),
            }
    return checks


def likelihood_audit_payload(
    catalog_path: str | Path = SMOKE_CATALOG,
    grid_path: str | Path = SMOKE_GRID,
    nsf_model_path: str = "data/models/conditional_nsf_backend_current.pt",
    preprocessing_stats_path: str = "data/models/nsf_preprocessing_stats.json",
) -> dict[str, Any]:
    catalog_arrays, catalog_md = load_npz_with_metadata(catalog_path)
    grid_arrays, grid_md = load_npz_with_metadata(grid_path)
    z = catalog_arrays["z"]
    lnmu = catalog_arrays["lnmu"]
    bin_edges = load_bin_edges()
    unique_z, counts = np.unique(np.round(z, 2), return_counts=True)

    checks = {
        "catalog_finite": bool(np.isfinite(z).all() and np.isfinite(lnmu).all()),
        "catalog_hash_matches_grid": bool(catalog_md["catalog_hash"] == grid_md["catalog_hash"]),
        "grid_integrity": validate_grid_arrays(grid_arrays),
        "metadata_hashes_present": {
            "simulator_git_commit": bool(grid_md.get("simulator_git_commit")),
            "nsf_checkpoint_hash": bool(grid_md.get("nsf_checkpoint_hash")),
            "preprocessing_stats_hash": bool(grid_md.get("preprocessing_stats_hash")),
        },
        "checkpoint_hash_matches_file": bool(grid_md.get("nsf_checkpoint_hash") == file_sha256(nsf_model_path)),
        "preprocessing_hash_matches_file": bool(grid_md.get("preprocessing_stats_hash") == optional_file_sha256(preprocessing_stats_path)),
        "current_simulator_commit": simulator_git_commit(),
        "grid_simulator_commit": grid_md.get("simulator_git_commit"),
    }

    audit_items = [
        {
            "topic": "event_ordering",
            "simulator_path": "Groups events by rounded redshift, then accumulates histogram counts.",
            "nsf_path": "Repeats the full event vector for each theta and sums per-event log probabilities.",
            "status": "pass",
            "impact": "Ordering does not affect either sum.",
        },
        {
            "topic": "redshift_handling",
            "simulator_path": "Rounds z to two decimals and evaluates one simulator PDF per unique rounded z.",
            "nsf_path": "Uses each event's raw z value in the continuous context.",
            "status": "conditional_pass",
            "impact": "The smoke catalog uses exact support z={0.5,1.5,2.5}, so rounding is not active there; this would be unsafe for continuous-z catalogs.",
        },
        {
            "topic": "prior_truncation",
            "simulator_path": "Returns -inf outside the configured flat prior.",
            "nsf_path": "Phase 3 grid construction only evaluates points inside the same prior bounds.",
            "status": "pass",
            "impact": "No prior-bound mismatch for grid posterior benchmarks.",
        },
        {
            "topic": "normalization_constants",
            "simulator_path": "Uses fixed-bin histogram probability masses; the missing bin-width density constant is theta-independent for fixed bins.",
            "nsf_path": "Uses raw continuous density with the learned lnmu standardization Jacobian.",
            "status": "conditional_pass",
            "impact": "The bin-width constant should not move a posterior on fixed bins, but simulator and NSF likelihood values are not absolute-density comparable.",
        },
        {
            "topic": "zero_probability_handling",
            "simulator_path": "Adds 1e-12 to every histogram probability before logging.",
            "nsf_path": "Uses the flow density directly and raises on non-finite outputs.",
            "status": "fail",
            "impact": "The simulator has an explicit low-probability floor and the NSF does not; tail events can receive systematically different leverage.",
        },
        {
            "topic": "clamping",
            "simulator_path": f"Clamps catalog and simulator samples to ({bin_edges[0]:.3f}, {bin_edges[-1]:.3f}) before histogramming.",
            "nsf_path": "Does not clamp lnmu before density evaluation.",
            "status": "fail",
            "impact": "Out-of-range or edge-near events are treated differently by the two likelihoods.",
        },
        {
            "topic": "per_event_accumulation",
            "simulator_path": "Equivalent to summing log bin probabilities per event after redshift slicing.",
            "nsf_path": "Sums log p(lnmu_j | z_j, theta) directly over events.",
            "status": "conditional_pass",
            "impact": "Aggregation is a sum in both paths; the density approximation and tail handling differ.",
        },
        {
            "topic": "event_weighting",
            "simulator_path": "Each catalog event contributes one histogram count.",
            "nsf_path": "Each catalog event contributes one log-density term.",
            "status": "pass",
            "impact": "No explicit weighting mismatch found.",
        },
        {
            "topic": "catalog_slicing",
            "simulator_path": "Slices by rounded redshift and sums slice likelihoods.",
            "nsf_path": "Processes all events in one flattened batch while retaining event-level z.",
            "status": "pass",
            "impact": "For fixed discrete redshifts, both implement the same catalog membership.",
        },
    ]

    failures = [item for item in audit_items if item["status"] == "fail"]
    verdict = "fail" if failures else "pass"
    return {
        "phase": "3B",
        "objective": "likelihood_construction_audit",
        "verdict": verdict,
        "catalog": {
            "path": str(catalog_path),
            "N": int(len(z)),
            "catalog_hash": catalog_md["catalog_hash"],
            "z_counts": {f"{float(k):.2f}": int(v) for k, v in zip(unique_z, counts)},
            "lnmu_min": float(np.min(lnmu)),
            "lnmu_max": float(np.max(lnmu)),
            "outside_histogram_range_count": int(np.sum((lnmu <= bin_edges[0]) | (lnmu >= bin_edges[-1]))),
        },
        "configuration": {
            "prior_bounds": {k: list(v) for k, v in PRIOR_BOUNDS.items()},
            "likelihood_version": LIKELIHOOD_VERSION,
            "simulator_nsim_per_z": int(grid_md.get("nsim_per_z", -1)),
            "grid_definition": grid_md.get("grid_definition"),
        },
        "checks": checks,
        "audit_items": audit_items,
        "blocking_mismatches": failures,
        "interpretation": (
            "The catalog-level sum is consistent, but the simulator reference applies histogram clamping and a 1e-12 tail floor "
            "while the NSF evaluates an unclamped continuous density. Phase 3B should therefore treat the smoke failure as a "
            "real density/tail-handling mismatch unless single-z isolation points to a bookkeeping-only issue."
        ),
    }


def write_likelihood_audit(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_likelihood_audit.json", payload)
    rows = []
    for item in payload["audit_items"]:
        rows.append(f"| {item['topic']} | {item['status']} | {item['impact']} |")
    lines = [
        "# Phase 3B Likelihood Construction Audit",
        "",
        f"Verdict: **{payload['verdict'].upper()}**",
        "",
        "Both benchmark paths sum catalog log-likelihood contributions, but two implementation details are not identical: simulator-side histogram clamping and the `1e-12` zero-probability floor.",
        "",
        "## Audit Matrix",
        "",
        "| Topic | Status | Impact |",
        "|---|---|---|",
        *rows,
        "",
        "## Cache and Grid Integrity",
        "",
        f"- Catalog hash matches grid metadata: `{payload['checks']['catalog_hash_matches_grid']}`",
        f"- Checkpoint hash matches file: `{payload['checks']['checkpoint_hash_matches_file']}`",
        f"- Preprocessing stats hash matches file: `{payload['checks']['preprocessing_hash_matches_file']}`",
        f"- Catalog finite: `{payload['checks']['catalog_finite']}`",
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
    ]
    ensure_dir(DOCS_DIR)
    (DOCS_DIR / "phase3b_likelihood_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def contour_levels(post: np.ndarray) -> list[float]:
    flat = np.sort(post.ravel())[::-1]
    csum = np.cumsum(flat)
    levels = []
    for mass in [0.95, 0.68]:
        idx = min(int(np.searchsorted(csum, mass)), flat.size - 1)
        levels.append(float(flat[idx]))
    return sorted(set(levels))


def plot_overlay(grid_file: str | Path, output_path: str | Path, title: str) -> None:
    arrays, md = load_npz_with_metadata(grid_file)
    axes = {k.removeprefix("axis_"): arrays[k] for k in arrays if k.startswith("axis_")}
    sim_post = normalize_log_grid(arrays["sim_log_likelihood"])
    nsf_post = normalize_log_grid(arrays["nsf_log_likelihood"])
    pair = ("OmegaM", "sigma8")
    kept_axes, sim_m = marginalize_to_axes(axes, sim_post, pair)
    _, nsf_m = marginalize_to_axes(axes, nsf_post, pair)
    x = kept_axes[pair[0]]
    y = kept_axes[pair[1]]

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.contour(x, y, sim_m.T, levels=contour_levels(sim_m), colors="#CC6F47", linestyles="--", linewidths=1.8)
    ax.contour(x, y, nsf_m.T, levels=contour_levels(nsf_m), colors="#5477C4", linewidths=1.8)
    ax.plot(md["truth"]["OmegaM"], md["truth"]["sigma8"], marker="s", ms=5, color="#1F2430", label="Truth")
    ax.plot([], [], color="#CC6F47", linestyle="--", label="Simulator")
    ax.plot([], [], color="#5477C4", label="NSF")
    ax.set_title(title)
    ax.set_xlabel("OmegaM")
    ax.set_ylabel("sigma8")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out = Path(output_path)
    ensure_dir(out.parent)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _mle_offset(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for name in ("OmegaM", "sigma8"):
        out[name] = float(metrics["nsf_summary"][name]["MAP"] - metrics["sim_summary"][name]["MAP"])
    return out


def summarize_grid_metrics(grid_file: str | Path) -> dict[str, Any]:
    arrays, md = load_npz_with_metadata(grid_file)
    axes = {k.removeprefix("axis_"): arrays[k] for k in arrays if k.startswith("axis_")}
    metrics = compare_posteriors(axes, arrays["sim_log_likelihood"], arrays["nsf_log_likelihood"], md["truth"])
    metrics["mle_offset"] = _mle_offset(metrics)
    return metrics


def single_z_catalog_path(z: float, n: int, seed: int, output_dir: str | Path) -> Path:
    suffix = f"single_z_central_N{n}_seed{seed}_z{z:g}_*.npz"
    matches = sorted(Path(output_dir).glob(suffix))
    if matches:
        return matches[0]
    return build_catalog("single_z", n, seed, "central", z=z, output_dir=output_dir)


def run_single_z_isolation(
    z_values: tuple[float, ...] = (0.5, 1.5, 2.5),
    n: int = 1000,
    resolution: int = 12,
    nsim_per_z: int = 1000,
    sim_max_workers: int | None = None,
) -> dict[str, Any]:
    catalog_dir = Path("data/mock_catalogs/phase3b_single_z")
    grid_dir = Path("data/results/phase3b_single_z_grids")
    fig_dir = Path("plots/figures/phase3b_single_z")
    results: dict[str, Any] = {
        "phase": "3B",
        "objective": "single_z_isolation",
        "configuration": {
            "N": int(n),
            "resolution": int(resolution),
            "nsim_per_z": int(nsim_per_z),
            "grid_type": "2d",
            "fixed_h": 0.67,
            "z_values": list(z_values),
        },
        "results": {},
    }
    for i, z in enumerate(z_values):
        catalog = single_z_catalog_path(z, n, 620500 + i, catalog_dir)
        grid = run_grid(
            catalog,
            grid_type="2d",
            resolution=resolution,
            output_dir=grid_dir,
            use_cache=True,
            overwrite_cache=False,
            nsim_per_z=nsim_per_z,
            sim_max_workers=sim_max_workers,
        )
        metrics = summarize_grid_metrics(grid)
        plot_path = fig_dir / f"single_z_z{str(z).replace('.', 'p')}_contours.png"
        plot_overlay(grid, plot_path, f"Phase 3B single-z posterior overlay, z={z:g}")
        results["results"][f"z={z:g}"] = {
            "catalog": str(catalog),
            "grid": str(grid),
            "plot": str(plot_path),
            "posterior_jsd": metrics["posterior_jsd"],
            "total_variation": metrics["total_variation"],
            "credible_region_overlap": metrics["credible_region_overlap"],
            "normalized_posterior_shift": metrics["normalized_posterior_shift"],
            "mle_offset": metrics["mle_offset"],
            "sim_summary": metrics["sim_summary"],
            "nsf_summary": metrics["nsf_summary"],
        }
    return results


def write_single_z_report(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_single_z_results.json", payload)
    lines = [
        "# Phase 3B Single-z Isolation Benchmark",
        "",
        "Fixed-redshift catalogs were evaluated with the same central cosmology as the smoke catalog and the same 2D posterior axes.",
        "",
        "| z | JSD | TV | 68% overlap | 95% overlap | dOmegaM MAP | dsigma8 MAP | Max normalized shift |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for z_key, rec in payload["results"].items():
        max_shift = max(float(v) for v in rec["normalized_posterior_shift"].values())
        lines.append(
            f"| {z_key.removeprefix('z=')} | {rec['posterior_jsd']:.6f} | {rec['total_variation']:.6f} | "
            f"{rec['credible_region_overlap']['credible_region_68']:.3f} | {rec['credible_region_overlap']['credible_region_95']:.3f} | "
            f"{rec['mle_offset']['OmegaM']:+.4f} | {rec['mle_offset']['sigma8']:+.4f} | {max_shift:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the fixed-redshift rows are healthy while the mixed catalog fails, the next target is mixed-catalog aggregation or grid resolution. If one or more fixed-redshift rows also fail, the mismatch is already present before redshift mixing and should be treated as density-model or low-z behavior.",
            "",
            "## Figures",
            "",
        ]
    )
    for z_key, rec in payload["results"].items():
        lines.append(f"- {z_key}: `{rec['plot']}`")
    (DOCS_DIR / "phase3b_single_z_isolation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid_refinement(
    resolution: int = 24,
    nsim_per_z: int = 1000,
    sim_max_workers: int | None = None,
) -> dict[str, Any]:
    grid_dir = Path("data/results/phase3b_grid_refinement_grids")
    refined = run_grid(
        SMOKE_CATALOG,
        grid_type="2d",
        resolution=resolution,
        output_dir=grid_dir,
        use_cache=True,
        overwrite_cache=False,
        nsim_per_z=nsim_per_z,
        sim_max_workers=sim_max_workers,
    )
    coarse_metrics = summarize_grid_metrics(SMOKE_GRID)
    refined_metrics = summarize_grid_metrics(refined)
    fig_dir = Path("plots/figures/phase3b_grid_refinement")
    refined_plot = fig_dir / f"mixed_refined_{resolution}x{resolution}_contours.png"
    plot_overlay(refined, refined_plot, f"Phase 3B mixed posterior overlay, {resolution}x{resolution}")
    return {
        "phase": "3B",
        "objective": "posterior_grid_refinement",
        "configuration": {
            "coarse_grid": str(SMOKE_GRID),
            "refined_resolution": int(resolution),
            "nsim_per_z": int(nsim_per_z),
        },
        "coarse": {
            "posterior_jsd": coarse_metrics["posterior_jsd"],
            "total_variation": coarse_metrics["total_variation"],
            "credible_region_overlap": coarse_metrics["credible_region_overlap"],
            "normalized_posterior_shift": coarse_metrics["normalized_posterior_shift"],
            "mle_offset": coarse_metrics["mle_offset"],
        },
        "refined": {
            "grid": str(refined),
            "plot": str(refined_plot),
            "posterior_jsd": refined_metrics["posterior_jsd"],
            "total_variation": refined_metrics["total_variation"],
            "credible_region_overlap": refined_metrics["credible_region_overlap"],
            "normalized_posterior_shift": refined_metrics["normalized_posterior_shift"],
            "mle_offset": refined_metrics["mle_offset"],
        },
    }


def write_grid_refinement_report(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_grid_refinement.json", payload)
    lines = [
        "# Phase 3B Posterior Grid Refinement",
        "",
        "The central mixed-catalog posterior was rerun on a refined 2D grid and compared with the 12x12 smoke result.",
        "",
        "| Grid | JSD | TV | 68% overlap | 95% overlap | dOmegaM MAP | dsigma8 MAP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ["coarse", "refined"]:
        rec = payload[label]
        lines.append(
            f"| {label} | {rec['posterior_jsd']:.6f} | {rec['total_variation']:.6f} | "
            f"{rec['credible_region_overlap']['credible_region_68']:.3f} | {rec['credible_region_overlap']['credible_region_95']:.3f} | "
            f"{rec['mle_offset']['OmegaM']:+.4f} | {rec['mle_offset']['sigma8']:+.4f} |"
        )
    lines.extend(["", f"Refined contour overlay: `{payload['refined']['plot']}`"])
    (DOCS_DIR / "phase3b_grid_refinement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_slice_catalog(source_catalog: str | Path, z_value: float, output_dir: str | Path) -> Path:
    arrays, md = load_npz_with_metadata(source_catalog)
    z = arrays["z"].astype(np.float64)
    lnmu = arrays["lnmu"].astype(np.float64)
    mask = np.round(z, 2) == round(float(z_value), 2)
    if not np.any(mask):
        raise ValueError(f"No events found at z={z_value}")
    slice_md = {
        "catalog_id": f"{md['catalog_id']}_slice_z{z_value:g}",
        "catalog_type": "phase3b_redshift_slice",
        "source_catalog_hash": md["catalog_hash"],
        "z_distribution": {"type": "single_slice_from_mixed", "z": float(z_value)},
        "N": int(mask.sum()),
        "seed": int(md["seed"]),
        "cosmology_id": md["cosmology_id"],
        "true_h": float(md["true_h"]),
        "true_OmegaM": float(md["true_OmegaM"]),
        "true_sigma8": float(md["true_sigma8"]),
        "simulator_git_commit": md["simulator_git_commit"],
        "generation_timestamp": utc_timestamp(),
    }
    slice_md["catalog_hash"] = catalog_hash(z[mask], lnmu[mask], slice_md)
    path = ensure_dir(output_dir) / f"mixed_slice_z{z_value:g}_N{int(mask.sum())}_{slice_md['catalog_hash'][:16]}.npz"
    if not path.exists():
        save_npz_with_metadata(path, {"z": z[mask], "lnmu": lnmu[mask]}, slice_md)
    return path


def run_redshift_slice_decomposition(
    nsim_per_z: int = 1000,
    resolution: int = 12,
    sim_max_workers: int | None = None,
) -> dict[str, Any]:
    catalog_dir = Path("data/mock_catalogs/phase3b_redshift_slices")
    grid_dir = Path("data/results/phase3b_redshift_slice_grids")
    fig_dir = Path("plots/figures/phase3b_redshift_slice")
    full_metrics = summarize_grid_metrics(SMOKE_GRID)
    payload: dict[str, Any] = {
        "phase": "3B",
        "objective": "redshift_slice_decomposition",
        "configuration": {
            "source_catalog": str(SMOKE_CATALOG),
            "resolution": int(resolution),
            "nsim_per_z": int(nsim_per_z),
        },
        "full_mixed_smoke": {
            "posterior_jsd": full_metrics["posterior_jsd"],
            "total_variation": full_metrics["total_variation"],
            "credible_region_overlap": full_metrics["credible_region_overlap"],
            "normalized_posterior_shift": full_metrics["normalized_posterior_shift"],
            "mle_offset": full_metrics["mle_offset"],
        },
        "slices": {},
    }
    for label, z_val in [("low_z", 0.5), ("mid_z", 1.5), ("high_z", 2.5)]:
        catalog = write_slice_catalog(SMOKE_CATALOG, z_val, catalog_dir)
        grid = run_grid(
            catalog,
            grid_type="2d",
            resolution=resolution,
            output_dir=grid_dir,
            use_cache=True,
            overwrite_cache=False,
            nsim_per_z=nsim_per_z,
            sim_max_workers=sim_max_workers,
        )
        metrics = summarize_grid_metrics(grid)
        plot_path = fig_dir / f"{label}_contours.png"
        plot_overlay(grid, plot_path, f"Phase 3B mixed-catalog {label} slice, z={z_val:g}")
        payload["slices"][label] = {
            "z": float(z_val),
            "catalog": str(catalog),
            "grid": str(grid),
            "plot": str(plot_path),
            "posterior_jsd": metrics["posterior_jsd"],
            "total_variation": metrics["total_variation"],
            "credible_region_overlap": metrics["credible_region_overlap"],
            "normalized_posterior_shift": metrics["normalized_posterior_shift"],
            "mle_offset": metrics["mle_offset"],
        }
    return payload


def write_redshift_slice_report(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_redshift_slice_results.json", payload)
    lines = [
        "# Phase 3B Redshift-Slice Decomposition",
        "",
        "The central mixed catalog was split into low-, mid-, and high-redshift subsets, then each subset was evaluated on the same 2D posterior grid.",
        "",
        "| Slice | z | JSD | TV | 68% overlap | 95% overlap | dOmegaM MAP | dsigma8 MAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rec in payload["slices"].items():
        lines.append(
            f"| {label} | {rec['z']:.1f} | {rec['posterior_jsd']:.6f} | {rec['total_variation']:.6f} | "
            f"{rec['credible_region_overlap']['credible_region_68']:.3f} | {rec['credible_region_overlap']['credible_region_95']:.3f} | "
            f"{rec['mle_offset']['OmegaM']:+.4f} | {rec['mle_offset']['sigma8']:+.4f} |"
        )
    lines.extend(["", "## Figures", ""])
    for label, rec in payload["slices"].items():
        lines.append(f"- {label}: `{rec['plot']}`")
    (DOCS_DIR / "phase3b_redshift_slice_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def local_density_residuals(
    catalog_path: str | Path,
    theta_points: dict[str, list[float]],
    nsim_per_z: int = 1000,
) -> dict[str, Any]:
    from ml.phase3_common import import_gwlensing, load_nsf_model

    arrays, _ = load_npz_with_metadata(catalog_path)
    z = arrays["z"].astype(np.float64)
    lnmu = arrays["lnmu"].astype(np.float64)
    bin_edges = load_bin_edges()
    model = load_nsf_model()
    gw = import_gwlensing()
    output: dict[str, Any] = {"theta_points": {}, "nsim_per_z": int(nsim_per_z)}
    for label, theta in theta_points.items():
        theta_arr = np.asarray(theta, dtype=float)
        nsf_ll = nsf_mixed_catalog_log_likelihood(model, z, lnmu, theta_arr.reshape(1, 3))[0]
        sim_event_logp = np.empty_like(lnmu)
        for z_val in np.unique(np.round(z, 2)):
            mask = np.round(z, 2) == z_val
            res = gw.sample_lnmu_ml_with_diagnostics(float(z_val), float(theta_arr[0]), float(theta_arr[1]), float(theta_arr[2]), int(nsim_per_z), 100, False)
            samples = np.asarray(res["lnmu"], dtype=np.float64)
            samples = samples[np.isfinite(samples)]
            clamped = np.clip(samples, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
            counts, _ = np.histogram(clamped, bins=bin_edges)
            probs = counts / max(float(np.sum(counts)), 1.0)
            event_bins = np.searchsorted(bin_edges, np.clip(lnmu[mask], bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9), side="right") - 1
            event_bins = np.clip(event_bins, 0, len(probs) - 1)
            sim_event_logp[mask] = np.log(probs[event_bins] + 1e-12)
        # NSF event logs for a single theta.
        import torch

        device = model.context_mean.device
        x_t = torch.tensor(lnmu.astype(np.float32), device=device).reshape(-1, 1)
        z_t = torch.tensor(z.astype(np.float32), device=device).reshape(-1, 1)
        theta_t = torch.tensor(np.tile(theta_arr.astype(np.float32), (len(z), 1)), device=device)
        context = torch.cat([z_t, theta_t], dim=1)
        with torch.no_grad():
            nsf_event = model.log_prob(x_t, context).detach().cpu().numpy().reshape(-1)
        residual = nsf_event - sim_event_logp
        residual_rows = [
            {
                "z": float(z_i),
                "lnmu": float(lnmu_i),
                "residual_nsf_minus_sim": float(res_i),
            }
            for z_i, lnmu_i, res_i in zip(z, lnmu, residual)
        ]
        by_z = {}
        for z_val in np.unique(np.round(z, 2)):
            mask = np.round(z, 2) == z_val
            by_z[f"{float(z_val):.2f}"] = {
                "n": int(mask.sum()),
                "mean_residual_nsf_minus_sim": float(np.mean(residual[mask])),
                "median_residual_nsf_minus_sim": float(np.median(residual[mask])),
                "sum_residual_nsf_minus_sim": float(np.sum(residual[mask])),
            }
        output["theta_points"][label] = {
            "theta": [float(x) for x in theta_arr],
            "nsf_catalog_log_likelihood": float(nsf_ll),
            "sim_histogram_event_sum": float(np.sum(sim_event_logp)),
            "sum_residual_nsf_minus_sim": float(np.sum(residual)),
            "mean_residual_nsf_minus_sim": float(np.mean(residual)),
            "positive_residual_fraction": float(np.mean(residual > 0)),
            "event_residuals": residual_rows,
            "by_z": by_z,
        }
    return output


def write_local_density_outputs(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_local_density_check.json", payload)
    fig_dir = ensure_dir("plots/figures/phase3b_local_density_check")
    labels = []
    sums = []
    for label, rec in payload["theta_points"].items():
        for z_key, zrec in rec["by_z"].items():
            labels.append(f"{label}\nz={z_key}")
            sums.append(zrec["sum_residual_nsf_minus_sim"])
    fig, ax = plt.subplots(figsize=(max(7.5, 0.7 * len(labels)), 4.5))
    ax.bar(labels, sums, color="#A3BEFA", edgecolor="#2E4780")
    ax.axhline(0, color="#1F2430", linewidth=1)
    ax.set_title("Local density residual sums by redshift")
    ax.set_ylabel("sum(log p_NSF - log p_sim-hist)")
    fig.tight_layout()
    plot_path = fig_dir / "residual_sum_by_z.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    scatter_path = fig_dir / "event_residuals_vs_redshift.png"
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"truth": "#5477C4", "sim_smoke_map": "#CC6F47", "nsf_smoke_map": "#71B436"}
    plotted_scatter = False
    for label, rec in payload["theta_points"].items():
        rows = rec.get("event_residuals", [])
        if not rows:
            continue
        plotted_scatter = True
        z_vals = [row["z"] for row in rows]
        residuals = [row["residual_nsf_minus_sim"] for row in rows]
        ax.scatter(z_vals, residuals, s=10, alpha=0.22, label=label, color=colors.get(label, "#7A828F"), edgecolors="none")
    ax.axhline(0, color="#1F2430", linewidth=1)
    ax.set_title("Event-wise density residuals by redshift")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("log p_NSF - log p_sim-hist")
    if plotted_scatter:
        ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    lines = [
        "# Phase 3B Local Density Residual Check",
        "",
        "Per-event NSF log densities were compared against the simulator histogram event log-probability approximation at representative theta points.",
        "",
        f"Slice residual figure: `{plot_path}`",
        f"Event residual figure: `{scatter_path}`",
        "",
        "| Theta point | Sum residual | Mean residual | Positive fraction |",
        "|---|---:|---:|---:|",
    ]
    for label, rec in payload["theta_points"].items():
        lines.append(f"| {label} | {rec['sum_residual_nsf_minus_sim']:.3f} | {rec['mean_residual_nsf_minus_sim']:.4f} | {rec['positive_residual_fraction']:.3f} |")
    ensure_dir(DOCS_DIR)
    (DOCS_DIR / "phase3b_local_density_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cache_repro_payload() -> dict[str, Any]:
    catalog_arrays, catalog_md = load_npz_with_metadata(SMOKE_CATALOG)
    grid_arrays, grid_md = load_npz_with_metadata(SMOKE_GRID)
    return {
        "phase": "3B",
        "objective": "cache_reproducibility",
        "smoke_catalog_cache_key_matches_current_simulator_commit": bool(catalog_md.get("simulator_git_commit") == simulator_git_commit()),
        "grid_cache_key_matches_current_simulator_commit": bool(grid_md.get("simulator_git_commit") == simulator_git_commit()),
        "catalog_hash_matches_grid": bool(catalog_md["catalog_hash"] == grid_md["catalog_hash"]),
        "nsf_checkpoint_hash_present": bool(grid_md.get("nsf_checkpoint_hash")),
        "preprocessing_stats_hash_present": bool(grid_md.get("preprocessing_stats_hash")),
        "no_nan_inf": bool(np.isfinite(catalog_arrays["z"]).all() and np.isfinite(catalog_arrays["lnmu"]).all() and all(np.isfinite(v).all() for k, v in grid_arrays.items() if "log_likelihood" in k)),
        "grid_integrity": validate_grid_arrays(grid_arrays),
    }


def write_cache_repro(payload: dict[str, Any]) -> None:
    write_json(RESULTS_DIR / "phase3b_cache_reproducibility.json", payload)
    lines = [
        "# Phase 3B Cache and Reproducibility Verification",
        "",
        f"- Smoke catalog commit matches current simulator commit: `{payload['smoke_catalog_cache_key_matches_current_simulator_commit']}`",
        f"- Smoke grid commit matches current simulator commit: `{payload['grid_cache_key_matches_current_simulator_commit']}`",
        f"- Catalog hash matches grid: `{payload['catalog_hash_matches_grid']}`",
        f"- NSF checkpoint hash present: `{payload['nsf_checkpoint_hash_present']}`",
        f"- Preprocessing stats hash present: `{payload['preprocessing_stats_hash_present']}`",
        f"- No NaN/Inf in checked catalog/grid arrays: `{payload['no_nan_inf']}`",
    ]
    (DOCS_DIR / "phase3b_cache_reproducibility.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3B mismatch diagnostics.")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--single_z", action="store_true")
    parser.add_argument("--grid_refinement", action="store_true")
    parser.add_argument("--redshift_slices", action="store_true")
    parser.add_argument("--local_density", action="store_true")
    parser.add_argument("--cache_repro", action="store_true")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--nsim_per_z", type=int, default=1000)
    parser.add_argument("--sim_max_workers", type=int, default=None)
    args = parser.parse_args()

    if args.audit:
        payload = likelihood_audit_payload()
        write_likelihood_audit(payload)
        print("Wrote Phase 3B likelihood audit")
    if args.single_z:
        payload = run_single_z_isolation(n=args.n, resolution=args.resolution, nsim_per_z=args.nsim_per_z, sim_max_workers=args.sim_max_workers)
        write_single_z_report(payload)
        print("Wrote Phase 3B single-z isolation results")
    if args.grid_refinement:
        payload = run_grid_refinement(resolution=max(args.resolution, 24), nsim_per_z=args.nsim_per_z, sim_max_workers=args.sim_max_workers)
        write_grid_refinement_report(payload)
        print("Wrote Phase 3B grid refinement results")
    if args.redshift_slices:
        payload = run_redshift_slice_decomposition(nsim_per_z=args.nsim_per_z, resolution=args.resolution, sim_max_workers=args.sim_max_workers)
        write_redshift_slice_report(payload)
        print("Wrote Phase 3B redshift-slice decomposition")
    if args.local_density:
        theta_points = {
            "truth": [0.67, 0.30, 0.85],
            "sim_smoke_map": [0.67, 0.3272727273, 0.7590909091],
            "nsf_smoke_map": [0.67, 0.20, 1.05],
        }
        payload = local_density_residuals(SMOKE_CATALOG, theta_points, nsim_per_z=args.nsim_per_z)
        write_local_density_outputs(payload)
        print("Wrote Phase 3B local density residual check")
    if args.cache_repro:
        payload = cache_repro_payload()
        write_cache_repro(payload)
        print("Wrote Phase 3B cache reproducibility verification")


if __name__ == "__main__":
    main()
