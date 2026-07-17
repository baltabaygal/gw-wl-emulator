import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ml.cache_utils import CacheManager, get_git_commit


PRIOR_BOUNDS = {
    "h": (0.59, 0.76),
    "OmegaM": (0.20, 0.40),
    "sigma8": (0.65, 1.05),
}
PARAM_NAMES = ("h", "OmegaM", "sigma8")
LIKELIHOOD_VERSION = "phase3_v1"


@dataclass(frozen=True)
class Cosmology:
    h: float
    OmegaM: float
    sigma8: float

    def as_context(self, z: float) -> np.ndarray:
        return np.array([z, self.h, self.OmegaM, self.sigma8], dtype=np.float32)

    def as_theta(self) -> np.ndarray:
        return np.array([self.h, self.OmegaM, self.sigma8], dtype=float)


TRUE_COSMOLOGIES = {
    "central": Cosmology(0.67, 0.30, 0.85),
    "edge": Cosmology(0.75, 0.39, 1.00),
    "low_sigma8": Cosmology(0.67, 0.30, 0.68),
    "high_sigma8": Cosmology(0.67, 0.30, 1.02),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_json_hash(payload: dict[str, Any]) -> str:
    cleaned = CacheManager()._clean_dict(payload)
    encoded = json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def optional_file_sha256(path: str | Path) -> str:
    p = Path(path)
    return file_sha256(p) if p.exists() else "missing"


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def import_gwlensing():
    build_dir = repo_root() / "build"
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    import gwlensing as gw  # type: ignore

    return gw


def make_redshifts(
    catalog_type: str,
    n: int,
    seed: int,
    z: float | None = None,
    mixed_z_bins: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if catalog_type == "single_z":
        if z is None:
            raise ValueError("single_z catalogs require z")
        redshifts = np.full(n, float(z), dtype=np.float64)
        desc = {"type": "single_z", "z": float(z)}
    elif catalog_type == "mixed_uniform":
        support = np.linspace(0.5, 2.5, int(mixed_z_bins), dtype=np.float64)
        redshifts = rng.choice(support, size=n, replace=True).astype(np.float64)
        desc = {"type": "stratified_uniform_support", "z_min": 0.5, "z_max": 2.5, "n_support": int(mixed_z_bins), "support": support.tolist()}
    elif catalog_type == "mixed_high_z_weighted":
        redshifts = (0.5 + 2.0 * rng.beta(2.0, 1.2, size=n)).astype(np.float64)
        desc = {"type": "beta_high_z_weighted", "z_min": 0.5, "z_max": 2.5, "alpha": 2.0, "beta": 1.2}
    elif catalog_type == "mixed_low_z_dominated":
        support = np.linspace(0.5, 2.5, int(mixed_z_bins), dtype=np.float64)
        p = np.array([0.6, 0.3, 0.1]) if int(mixed_z_bins) == 3 else None
        if p is not None and len(p) == int(mixed_z_bins):
            redshifts = rng.choice(support, size=n, replace=True, p=p).astype(np.float64)
        else:
            w = np.linspace(1.0, 0.1, int(mixed_z_bins))
            p = w / w.sum()
            redshifts = rng.choice(support, size=n, replace=True, p=p).astype(np.float64)
        desc = {"type": "mixed_low_z_dominated", "z_min": 0.5, "z_max": 2.5, "n_support": int(mixed_z_bins), "support": support.tolist(), "probabilities": p.tolist()}
    elif catalog_type == "mixed_high_z_dominated":
        support = np.linspace(0.5, 2.5, int(mixed_z_bins), dtype=np.float64)
        p = np.array([0.1, 0.3, 0.6]) if int(mixed_z_bins) == 3 else None
        if p is not None and len(p) == int(mixed_z_bins):
            redshifts = rng.choice(support, size=n, replace=True, p=p).astype(np.float64)
        else:
            w = np.linspace(0.1, 1.0, int(mixed_z_bins))
            p = w / w.sum()
            redshifts = rng.choice(support, size=n, replace=True, p=p).astype(np.float64)
        desc = {"type": "mixed_high_z_dominated", "z_min": 0.5, "z_max": 2.5, "n_support": int(mixed_z_bins), "support": support.tolist(), "probabilities": p.tolist()}
    else:
        raise ValueError(f"Unknown catalog_type: {catalog_type}")
    return redshifts, desc



def validate_prior_bounds(prior_bounds: dict[str, tuple[float, float]] = PRIOR_BOUNDS) -> None:
    for name in PARAM_NAMES:
        lo, hi = prior_bounds[name]
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise ValueError(f"Invalid prior bounds for {name}: {(lo, hi)}")


def in_prior(theta: np.ndarray, prior_bounds: dict[str, tuple[float, float]] = PRIOR_BOUNDS) -> bool:
    theta = np.asarray(theta, dtype=float)
    return bool(
        prior_bounds["h"][0] <= theta[0] <= prior_bounds["h"][1]
        and prior_bounds["OmegaM"][0] <= theta[1] <= prior_bounds["OmegaM"][1]
        and prior_bounds["sigma8"][0] <= theta[2] <= prior_bounds["sigma8"][1]
    )


def build_grid(grid_type: str, resolution: int | tuple[int, ...], fixed_h: float | None = None) -> dict[str, Any]:
    validate_prior_bounds()
    if grid_type == "2d":
        if isinstance(resolution, tuple):
            n_om, n_s8 = resolution
        else:
            n_om = n_s8 = int(resolution)
        if fixed_h is None:
            raise ValueError("2D grids require fixed_h")
        axes = {
            "OmegaM": np.linspace(*PRIOR_BOUNDS["OmegaM"], n_om),
            "sigma8": np.linspace(*PRIOR_BOUNDS["sigma8"], n_s8),
        }
        OM, S8 = np.meshgrid(axes["OmegaM"], axes["sigma8"], indexing="ij")
        theta = np.column_stack([np.full(OM.size, fixed_h), OM.ravel(), S8.ravel()])
    elif grid_type == "3d":
        if isinstance(resolution, tuple):
            n_h, n_om, n_s8 = resolution
        else:
            n_h = n_om = n_s8 = int(resolution)
        axes = {
            "h": np.linspace(*PRIOR_BOUNDS["h"], n_h),
            "OmegaM": np.linspace(*PRIOR_BOUNDS["OmegaM"], n_om),
            "sigma8": np.linspace(*PRIOR_BOUNDS["sigma8"], n_s8),
        }
        H, OM, S8 = np.meshgrid(axes["h"], axes["OmegaM"], axes["sigma8"], indexing="ij")
        theta = np.column_stack([H.ravel(), OM.ravel(), S8.ravel()])
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")
    return {
        "grid_type": grid_type,
        "resolution": [int(x) for x in (resolution if isinstance(resolution, tuple) else ([resolution] * (2 if grid_type == "2d" else 3)))],
        "fixed_h": None if fixed_h is None else float(fixed_h),
        "axes": axes,
        "theta": theta.astype(np.float64),
        "prior_bounds": {k: list(v) for k, v in PRIOR_BOUNDS.items()},
    }


def grid_definition(grid: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid_type": grid["grid_type"],
        "resolution": grid["resolution"],
        "fixed_h": grid["fixed_h"],
        "axes": {k: [float(v[0]), float(v[-1]), int(len(v))] for k, v in grid["axes"].items()},
    }


def normalize_log_grid(log_likelihood: np.ndarray) -> np.ndarray:
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    if not np.isfinite(log_likelihood).all():
        raise ValueError("Posterior grid contains NaN or Inf log likelihood values")
    shifted = log_likelihood - np.max(log_likelihood)
    posterior = np.exp(shifted)
    total = float(np.sum(posterior))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Posterior grid has non-positive normalization")
    return posterior / total


def save_npz_with_metadata(path: str | Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    np.savez(path, metadata=json.dumps(CacheManager()._clean_dict(metadata), sort_keys=True), **arrays)


def load_npz_with_metadata(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as npz:
        metadata = json.loads(str(npz["metadata"]))
        arrays = {k: npz[k] for k in npz.files if k != "metadata"}
    return arrays, metadata


def run_command_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=repo_root(), capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def simulator_git_commit() -> str:
    return get_git_commit()


def catalog_hash(z: np.ndarray, lnmu: np.ndarray, metadata: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(z, dtype=np.float64).tobytes())
    h.update(np.asarray(lnmu, dtype=np.float64).tobytes())
    h.update(json.dumps(CacheManager()._clean_dict(metadata), sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def load_nsf_model(model_path: str = "data/models/conditional_nsf_backend_current.pt"):
    from ml.nsf_model import ConditionalNSF

    model = ConditionalNSF(input_dim=1, context_dim=4)
    model.load_checkpoint(model_path)
    model.eval()
    return model


def simulator_pdf_log_likelihood_from_counts(
    catalog_counts: np.ndarray,
    z: float,
    theta: np.ndarray,
    bin_edges: np.ndarray,
    nsim: int = 10000,
    seed: int = 100,
) -> float:
    gw = import_gwlensing()
    h, om, s8 = [float(x) for x in theta]
    if not in_prior(np.array([h, om, s8])):
        return -np.inf
    res = gw.sample_lnmu_ml_with_diagnostics(float(z), h, om, s8, int(nsim), int(seed), False)
    lnmu = np.asarray(res["lnmu"], dtype=np.float64)
    lnmu = lnmu[np.isfinite(lnmu)]
    if lnmu.size == 0:
        raise ValueError("Simulator returned no finite lnmu samples")
    clamped = np.clip(lnmu, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts, _ = np.histogram(clamped, bins=bin_edges)
    probs = counts / max(float(np.sum(counts)), 1.0)
    if not np.isfinite(probs).all():
        raise ValueError("Simulator produced non-finite histogram probabilities")
    return float(np.sum(catalog_counts * np.log(probs + 1e-12)))


def mixed_catalog_simulator_log_likelihood(
    z: np.ndarray,
    lnmu: np.ndarray,
    theta: np.ndarray,
    bin_edges: np.ndarray,
    nsim_per_z: int = 6000,
    seed: int = 100,
    z_round: int = 2,
) -> float:
    if not in_prior(theta):
        return -np.inf
    total = 0.0
    z_bins = np.round(np.asarray(z, dtype=float), z_round)
    for z_val in np.unique(z_bins):
        mask = z_bins == z_val
        clamped = np.clip(lnmu[mask], bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
        counts, _ = np.histogram(clamped, bins=bin_edges)
        total += simulator_pdf_log_likelihood_from_counts(
            counts,
            float(z_val),
            theta,
            bin_edges,
            nsim=nsim_per_z,
            seed=seed,
        )
    return float(total)


def evaluate_callable_grid(
    theta_grid: np.ndarray,
    fn: Callable[[np.ndarray], float],
    shape: tuple[int, ...],
) -> np.ndarray:
    vals = np.empty(theta_grid.shape[0], dtype=np.float64)
    for i, theta in enumerate(theta_grid):
        val = fn(theta)
        if not np.isfinite(val):
            raise ValueError(f"Non-finite likelihood at grid index {i}: {val}")
        vals[i] = val
    return vals.reshape(shape)
