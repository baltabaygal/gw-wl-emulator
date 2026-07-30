"""Glue to the parent repo: locate the root, load ml/params.py by file path.

The package must stay importable from any cwd and on both pythons (Mac test env
3.12, sandbox 3.10), so ml.params is loaded via importlib from an explicit path
rather than relying on the repo root being on sys.path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_params = _load_module("_hdr_ml_params", REPO_ROOT / "ml" / "params.py")

FIDUCIAL = dict(_params.FIDUCIAL)
PRIOR_6D = dict(_params.PRIOR_6D)
WIDE_6D = dict(_params.WIDE_6D)
PRODUCTION_CONFIG = dict(_params.PRODUCTION_CONFIG)
PRODUCTION_CONFIG_HASH = _params.PRODUCTION_CONFIG_HASH

# Vaskonen's 3-parameter inference space, in his MCMC order (main_lensing.cpp:188
# par = {OmegaM, sigma8, h}); the remaining three are held at FIDUCIAL.
THETA3_KEYS = ("Om", "sigma8", "h")
# Full 1+6d inference space (order = ml/params.py CONTEXT_KEYS minus z).
THETA6_KEYS = ("h", "Om", "sigma8", "Ob", "ns", "zeq")

# Vaskonen's priors (paper + main_lensing.cpp:189) for the 3d mode. The 6d mode
# uses PRIOR_6D. Note his Om prior (0.15, 0.47) is wider than PRIOR_6D's.
PRIOR_VASKONEN = dict(Om=(0.15, 0.47), sigma8=(0.4, 1.4), h=(0.59, 0.76))


def theta_full(theta: dict) -> dict:
    """FIDUCIAL completed/overridden by theta (dict in, full 6-key dict out)."""
    out = dict(FIDUCIAL)
    out.update(theta)
    return out


def pack(theta: dict, keys) -> "list[float]":
    return [float(theta[k]) for k in keys]


def unpack(vec, keys) -> dict:
    return {k: float(v) for k, v in zip(keys, vec)}
