import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon

from ml.phase3_common import PARAM_NAMES, ensure_dir, load_npz_with_metadata, normalize_log_grid


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def _credible_interval(values: np.ndarray, weights: np.ndarray, mass: float) -> list[float]:
    alpha = 0.5 * (1.0 - mass)
    return [_weighted_quantile(values, weights, alpha), _weighted_quantile(values, weights, 1.0 - alpha)]


def _credible_mask(posterior: np.ndarray, mass: float) -> np.ndarray:
    flat = posterior.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    keep = order[csum <= mass]
    if keep.size == 0:
        keep = order[:1]
    elif keep.size < flat.size:
        keep = np.append(keep, order[keep.size])
    mask = np.zeros_like(flat, dtype=bool)
    mask[keep] = True
    return mask.reshape(posterior.shape)


def _mask_overlap(a: np.ndarray, b: np.ndarray) -> float:
    union = np.sum(a | b)
    if union == 0:
        return 0.0
    return float(np.sum(a & b) / union)


def grid_coordinates(axes: dict[str, np.ndarray], posterior: np.ndarray) -> dict[str, np.ndarray]:
    names = list(axes)
    mesh = np.meshgrid(*[axes[n] for n in names], indexing="ij")
    return {name: arr.ravel() for name, arr in zip(names, mesh)}


def posterior_summary(axes: dict[str, np.ndarray], posterior: np.ndarray) -> dict[str, dict[str, float | list[float]]]:
    p = np.asarray(posterior, dtype=float)
    p = p / np.sum(p)
    coords = grid_coordinates(axes, p)
    flat_p = p.ravel()
    map_flat = int(np.argmax(flat_p))
    summaries: dict[str, dict[str, float | list[float]]] = {}
    for name, vals in coords.items():
        mean = float(np.sum(vals * flat_p))
        std = float(np.sqrt(np.sum(((vals - mean) ** 2) * flat_p)))
        summaries[name] = {
            "mean": mean,
            "median": _weighted_quantile(vals, flat_p, 0.5),
            "MAP": float(vals[map_flat]),
            "ci68": _credible_interval(vals, flat_p, 0.68),
            "ci95": _credible_interval(vals, flat_p, 0.95),
            "std": std,
        }
    return summaries


def marginalize_to_axes(axes: dict[str, np.ndarray], posterior: np.ndarray, keep: tuple[str, ...]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    names = list(axes)
    sum_axes = tuple(i for i, name in enumerate(names) if name not in keep)
    marginal = np.sum(posterior, axis=sum_axes) if sum_axes else posterior.copy()
    kept_axes = {name: axes[name] for name in names if name in keep}
    return kept_axes, marginal / np.sum(marginal)


def compare_posteriors(
    axes: dict[str, np.ndarray],
    sim_log_likelihood: np.ndarray,
    nsf_log_likelihood: np.ndarray,
    truth: dict[str, float],
) -> dict[str, Any]:
    sim_post = normalize_log_grid(sim_log_likelihood)
    nsf_post = normalize_log_grid(nsf_log_likelihood)
    if sim_post.shape != nsf_post.shape:
        raise ValueError(f"Posterior shape mismatch: {sim_post.shape} vs {nsf_post.shape}")

    sim_summary = posterior_summary(axes, sim_post)
    nsf_summary = posterior_summary(axes, nsf_post)
    params = [p for p in PARAM_NAMES if p in axes]
    bias: dict[str, Any] = {}
    shifts: dict[str, float] = {}
    coverage = {"nsf_68": {}, "nsf_95": {}, "sim_68": {}, "sim_95": {}}
    for name in params:
        sim_mean = float(sim_summary[name]["mean"])
        nsf_mean = float(nsf_summary[name]["mean"])
        sim_std = max(float(sim_summary[name]["std"]), 1e-12)
        truth_val = float(truth[name])
        bias[name] = {
            "sim_minus_truth": sim_mean - truth_val,
            "nsf_minus_truth": nsf_mean - truth_val,
            "nsf_minus_sim": nsf_mean - sim_mean,
        }
        shifts[name] = abs(nsf_mean - sim_mean) / sim_std
        for method, summary in [("nsf", nsf_summary), ("sim", sim_summary)]:
            for mass in ["68", "95"]:
                lo, hi = summary[name][f"ci{mass}"]
                coverage[f"{method}_{mass}"][name] = bool(lo <= truth_val <= hi)

    eps = 1e-15
    jsd = float(jensenshannon(sim_post.ravel() + eps, nsf_post.ravel() + eps, base=np.e) ** 2)
    tv = float(0.5 * np.sum(np.abs(sim_post - nsf_post)))
    overlaps = {
        "credible_region_68": _mask_overlap(_credible_mask(sim_post, 0.68), _credible_mask(nsf_post, 0.68)),
        "credible_region_95": _mask_overlap(_credible_mask(sim_post, 0.95), _credible_mask(nsf_post, 0.95)),
    }

    contour_overlaps = {}
    if len(params) >= 2:
        pairs = [(params[i], params[j]) for i in range(len(params)) for j in range(i + 1, len(params))]
        for pair in pairs:
            _, sim_m = marginalize_to_axes(axes, sim_post, pair)
            _, nsf_m = marginalize_to_axes(axes, nsf_post, pair)
            contour_overlaps["/".join(pair)] = {
                "overlap_68": _mask_overlap(_credible_mask(sim_m, 0.68), _credible_mask(nsf_m, 0.68)),
                "overlap_95": _mask_overlap(_credible_mask(sim_m, 0.95), _credible_mask(nsf_m, 0.95)),
            }

    return {
        "sim_summary": sim_summary,
        "nsf_summary": nsf_summary,
        "bias": bias,
        "normalized_posterior_shift": shifts,
        "posterior_jsd": jsd,
        "total_variation": tv,
        "credible_region_overlap": overlaps,
        "contour_overlap": contour_overlaps,
        "truth_coverage": coverage,
    }


def coverage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n_catalogs": 0}
    out: dict[str, Any] = {"n_catalogs": len(records), "parameters": {}}
    for name in PARAM_NAMES:
        vals68 = [bool(r["truth_coverage"]["nsf_68"].get(name, False)) for r in records if name in r["truth_coverage"]["nsf_68"]]
        vals95 = [bool(r["truth_coverage"]["nsf_95"].get(name, False)) for r in records if name in r["truth_coverage"]["nsf_95"]]
        if vals68:
            n = len(vals68)
            f68 = float(np.mean(vals68))
            f95 = float(np.mean(vals95))
            out["parameters"][name] = {
                "coverage_68": f68,
                "coverage_95": f95,
                "binomial_sigma_68": float(np.sqrt(0.68 * 0.32 / n)),
                "binomial_sigma_95": float(np.sqrt(0.95 * 0.05 / n)),
            }
    return out


def metrics_from_grid_file(path: str | Path) -> dict[str, Any]:
    arrays, metadata = load_npz_with_metadata(path)
    axes = {k.removeprefix("axis_"): arrays[k] for k in arrays if k.startswith("axis_")}
    truth = metadata["truth"]
    return compare_posteriors(axes, arrays["sim_log_likelihood"], arrays["nsf_log_likelihood"], truth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Phase 3 posterior metrics from grid files.")
    parser.add_argument("grid_files", nargs="*", default=[])
    parser.add_argument("--grid_dir", default="data/results/phase3_posterior_grids")
    parser.add_argument("--output_json", default="data/results/phase3_posterior_metrics.json")
    parser.add_argument("--output_report", default="docs/phase3/phase3_posterior_metrics.md")
    args = parser.parse_args()

    files = [Path(p) for p in args.grid_files]
    if not files:
        files = sorted(Path(args.grid_dir).glob("*combined*.npz"))
    results = {}
    for path in files:
        results[path.name] = metrics_from_grid_file(path)
    ensure_dir(Path(args.output_json).parent)
    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# Phase 3 Posterior Metrics", "", f"Computed metrics for {len(results)} posterior grid file(s).", ""]
    for name, metrics in results.items():
        max_shift = max(metrics["normalized_posterior_shift"].values()) if metrics["normalized_posterior_shift"] else float("nan")
        lines.append(f"- `{name}`: JSD={metrics['posterior_jsd']:.6f}, TV={metrics['total_variation']:.4f}, max normalized shift={max_shift:.3f}")
    ensure_dir(Path(args.output_report).parent)
    Path(args.output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
