import argparse
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from ml.phase3_common import (
    LIKELIHOOD_VERSION,
    PRIOR_BOUNDS,
    build_grid,
    ensure_dir,
    file_sha256,
    grid_definition,
    load_npz_with_metadata,
    load_nsf_model,
    mixed_catalog_simulator_log_likelihood,
    normalize_log_grid,
    optional_file_sha256,
    save_npz_with_metadata,
    simulator_git_commit,
    stable_json_hash,
)
from ml.nsf_likelihood import nsf_mixed_catalog_log_likelihood


_SIM_WORKER_Z = None
_SIM_WORKER_LNMU = None
_SIM_WORKER_BIN_EDGES = None
_SIM_WORKER_NSIM_PER_Z = None


def _init_sim_worker(z: np.ndarray, lnmu: np.ndarray, bin_edges: np.ndarray, nsim_per_z: int) -> None:
    global _SIM_WORKER_Z, _SIM_WORKER_LNMU, _SIM_WORKER_BIN_EDGES, _SIM_WORKER_NSIM_PER_Z
    _SIM_WORKER_Z = z
    _SIM_WORKER_LNMU = lnmu
    _SIM_WORKER_BIN_EDGES = bin_edges
    _SIM_WORKER_NSIM_PER_Z = nsim_per_z


def _sim_worker(theta: np.ndarray) -> float:
    return mixed_catalog_simulator_log_likelihood(
        _SIM_WORKER_Z,
        _SIM_WORKER_LNMU,
        theta,
        _SIM_WORKER_BIN_EDGES,
        nsim_per_z=int(_SIM_WORKER_NSIM_PER_Z),
    )


def load_bin_edges(mlp_model_path: str = "data/models/baseline_mlp_backend_current.pt") -> np.ndarray:
    try:
        import torch

        checkpoint = torch.load(mlp_model_path, map_location="cpu", weights_only=False)
        return np.asarray(checkpoint["bin_edges"], dtype=np.float64)
    except Exception:
        return np.linspace(-1.0, 1.0, 101)




def evaluate_simulator_grid(
    z: np.ndarray,
    lnmu: np.ndarray,
    theta_grid: np.ndarray,
    shape: tuple[int, ...],
    bin_edges: np.ndarray,
    nsim_per_z: int,
    max_workers: int | None = None,
) -> np.ndarray:
    workers = max_workers or min(multiprocessing.cpu_count(), 10)
    if workers <= 1:
        vals = np.array([_sim_direct(z, lnmu, theta, bin_edges, nsim_per_z) for theta in theta_grid], dtype=np.float64)
    else:
        with multiprocessing.Pool(
            processes=workers,
            initializer=_init_sim_worker,
            initargs=(z, lnmu, bin_edges, nsim_per_z),
        ) as pool:
            vals = np.array(pool.map(_sim_worker, list(theta_grid)), dtype=np.float64)
    bad = np.where(~np.isfinite(vals))[0]
    if bad.size:
        raise ValueError(f"Simulator produced non-finite log likelihood at grid indices {bad[:10].tolist()}")
    return vals.reshape(shape)


def _sim_direct(z: np.ndarray, lnmu: np.ndarray, theta: np.ndarray, bin_edges: np.ndarray, nsim_per_z: int) -> float:
    return mixed_catalog_simulator_log_likelihood(z, lnmu, theta, bin_edges, nsim_per_z=nsim_per_z)


def run_grid(
    catalog_path: str | Path,
    grid_type: str = "3d",
    resolution: int = 12,
    nsf_model_path: str = "data/models/conditional_nsf_backend_current.pt",
    preprocessing_stats_path: str = "data/models/nsf_preprocessing_stats.json",
    output_dir: str | Path = "data/results/phase3_posterior_grids",
    use_cache: bool = True,
    overwrite_cache: bool = False,
    nsim_per_z: int = 6000,
    nsf_chunk_size: int = 128,
    sim_max_workers: int | None = None,
    likelihood_mode: str = "simulator_compatible",
) -> Path:
    arrays, catalog_md = load_npz_with_metadata(catalog_path)
    z = arrays["z"].astype(np.float64)
    lnmu = arrays["lnmu"].astype(np.float64)
    truth = {
        "h": float(catalog_md["true_h"]),
        "OmegaM": float(catalog_md["true_OmegaM"]),
        "sigma8": float(catalog_md["true_sigma8"]),
    }
    grid = build_grid(grid_type, resolution, fixed_h=truth["h"] if grid_type == "2d" else None)
    shape = tuple(len(v) for v in grid["axes"].values())
    bin_edges = load_bin_edges()
    cache_payload = {
        "catalog_hash": catalog_md["catalog_hash"],
        "grid_definition": grid_definition(grid),
        "prior_bounds": {k: list(v) for k, v in PRIOR_BOUNDS.items()},
        "simulator_git_commit": simulator_git_commit(),
        "likelihood_version": LIKELIHOOD_VERSION,
        "nsim_per_z": int(nsim_per_z),
    }
    cache_hash = stable_json_hash(cache_payload)
    out_dir = ensure_dir(output_dir)
    sim_path = out_dir / f"sim_grid_{cache_hash}.npz"
    nsf_payload = {
        **cache_payload,
        "nsf_checkpoint_hash": file_sha256(nsf_model_path),
        "preprocessing_stats_hash": optional_file_sha256(preprocessing_stats_path),
        "likelihood_mode": likelihood_mode,
    }
    nsf_hash = stable_json_hash(nsf_payload)
    nsf_path = out_dir / f"nsf_grid_{nsf_hash}.npz"

    timings = {}
    if use_cache and sim_path.exists() and not overwrite_cache:
        sim_arrays, _ = load_npz_with_metadata(sim_path)
        sim_grid = sim_arrays["log_likelihood"]
        timings["simulator_grid_seconds"] = 0.0
    else:
        t0 = time.time()
        sim_grid = evaluate_simulator_grid(z, lnmu, grid["theta"], shape, bin_edges, nsim_per_z, max_workers=sim_max_workers)
        timings["simulator_grid_seconds"] = time.time() - t0
        save_npz_with_metadata(sim_path, {"log_likelihood": sim_grid}, cache_payload)

    if use_cache and nsf_path.exists() and not overwrite_cache:
        nsf_arrays, _ = load_npz_with_metadata(nsf_path)
        nsf_grid = nsf_arrays["log_likelihood"]
        timings["nsf_grid_seconds"] = 0.0
    else:
        model = load_nsf_model(nsf_model_path)
        t0 = time.time()
        nsf_flat = nsf_mixed_catalog_log_likelihood(
            model,
            z,
            lnmu,
            grid["theta"],
            chunk_size=nsf_chunk_size,
            likelihood_mode=likelihood_mode,
            bin_edges=bin_edges,
        )
        nsf_grid = nsf_flat.reshape(shape)
        timings["nsf_grid_seconds"] = time.time() - t0
        save_npz_with_metadata(nsf_path, {"log_likelihood": nsf_grid}, nsf_payload)

    normalize_log_grid(sim_grid)
    normalize_log_grid(nsf_grid)
    combined_md = {
        **nsf_payload,
        "catalog_path": str(catalog_path),
        "catalog_metadata": catalog_md,
        "truth": truth,
        "timings": timings,
        "sim_grid_cache": str(sim_path),
        "nsf_grid_cache": str(nsf_path),
    }
    arrays_out = {
        "sim_log_likelihood": sim_grid,
        "nsf_log_likelihood": nsf_grid,
    }
    for name, axis in grid["axes"].items():
        arrays_out[f"axis_{name}"] = axis
    combined_path = out_dir / f"combined_{grid_type}_{Path(catalog_path).stem}_{nsf_hash[:12]}.npz"
    save_npz_with_metadata(combined_path, arrays_out, combined_md)
    return combined_path


def write_report(paths: list[Path], output_report: str | Path) -> None:
    lines = [
        "# Phase 3 Posterior Grid Engine",
        "",
        "Deterministic simulator and NSF posterior grids are evaluated on identical catalogs and identical flat priors.",
        "",
        "| Grid file | Simulator cache | NSF cache |",
        "|---|---|---|",
    ]
    for path in paths:
        _, md = load_npz_with_metadata(path)
        lines.append(f"| `{path.name}` | `{Path(md['sim_grid_cache']).name}` | `{Path(md['nsf_grid_cache']).name}` |")
    lines.extend(
        [
            "",
            f"Priors: h in {PRIOR_BOUNDS['h']}, OmegaM in {PRIOR_BOUNDS['OmegaM']}, sigma8 in {PRIOR_BOUNDS['sigma8']}.",
            "Cache keys include catalog hash, grid definition, prior bounds, simulator git commit, likelihood version, NSF checkpoint hash, and preprocessing stats hash.",
        ]
    )
    ensure_dir(Path(output_report).parent)
    Path(output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 posterior grids.")
    parser.add_argument("--catalog", action="append", default=[])
    parser.add_argument("--catalog_dir", default="data/mock_catalogs/phase3")
    parser.add_argument("--grid_type", choices=["2d", "3d"], default="3d")
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--output_dir", default="data/results/phase3_posterior_grids")
    parser.add_argument("--output_report", default="docs/phase3_posterior_grid_engine.md")
    parser.add_argument("--nsim_per_z", type=int, default=6000)
    parser.add_argument("--sim_max_workers", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--likelihood_mode", choices=["continuous", "simulator_compatible"], default="simulator_compatible")
    args = parser.parse_args()

    catalogs = [Path(p) for p in args.catalog]
    if not catalogs:
        catalogs = sorted(Path(args.catalog_dir).glob("*.npz"))[:1]
    if not catalogs:
        raise FileNotFoundError("No catalogs found. Run ml/generate_mock_catalogs.py first.")

    outputs = []
    for catalog in catalogs:
        print(f"Running {args.grid_type} grid for {catalog}")
        out = run_grid(
            catalog,
            grid_type=args.grid_type,
            resolution=args.resolution,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            overwrite_cache=args.overwrite_cache,
            nsim_per_z=args.nsim_per_z,
            sim_max_workers=args.sim_max_workers,
            likelihood_mode=args.likelihood_mode,
        )
        print(f"Wrote {out}")
        outputs.append(out)
    write_report(outputs, args.output_report)


if __name__ == "__main__":
    main()
