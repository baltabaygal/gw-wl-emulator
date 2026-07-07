"""Canonical 1+6d parameter space for the emulator: (z_s; h, Om, As, Ob, ns, zeq).

Single source of truth for prior boxes, the ML context layout, and the mapping
between physical parameters and context features. The C++ simulator takes the
physical parameters directly (A_s-mode: pass As>0, sigma8 is then ignored);
the ML context uses ln(1e10 As) and zeq/1000 for conditioning.

Legacy 1+3d datasets (columns [z, h, OmegaM, sigma8]) remain loadable through
ml.data; this module describes the 6d scheme used for new data generation.
"""
from typing import Dict

import numpy as np

# Planck-2018-like fiducial point (As-mode; sigma8 is derived, ~0.86 with the
# code's smooth-k window — see tests/test_cosmology_params.py).
FIDUCIAL = dict(h=0.674, Om=0.315, As=2.101e-9, Ob=0.0493, ns=0.965, zeq=3402.0)

# In-distribution prior box (training/inference support). As bounds chosen so the
# derived sigma8 spans roughly the legacy [0.65, 1.05] box (sigma8 ~ sqrt(As)).
PRIOR_6D = dict(
    h=(0.59, 0.76),
    Om=(0.20, 0.40),
    lnAs10=(np.log(12.0), np.log(36.0)),   # As in [1.2e-9, 3.6e-9]
    Ob=(0.035, 0.065),
    ns=(0.90, 1.02),
    zeq=(2500.0, 4500.0),
)

# Wider sampling box for dataset generation: (Om, lnAs10) extend past the ID box
# to populate the OoD corners (mirrors the legacy OmegaM/sigma8 margins; the
# lnAs10 margins are the sigma8 ones mapped through As ~ sigma8^2).
WIDE_6D = dict(PRIOR_6D)
WIDE_6D["Om"] = (0.15, 0.45)
WIDE_6D["lnAs10"] = (PRIOR_6D["lnAs10"][0] + 2.0 * np.log(0.4 / 0.65),
                     PRIOR_6D["lnAs10"][1] + 2.0 * np.log(1.4 / 1.05))

# ML context layout. z stays at index 0 (train_smooth.py relies on it).
CONTEXT_KEYS = ("z", "h", "Om", "lnAs10", "Ob", "ns", "zeq_k")
CONTEXT_DIM = len(CONTEXT_KEYS)

# HDF5 dataset keys for the physical parameters (generate_dataset.py schema 2.0)
PARAM_KEYS_6D = ("h", "OmegaM", "As", "OmegaB", "ns", "zeq")


def lnAs10_from_As(As):
    return np.log(1.0e10 * np.asarray(As, dtype=np.float64))


def As_from_lnAs10(lnAs10):
    return np.exp(np.asarray(lnAs10, dtype=np.float64)) * 1.0e-10


def theta_to_context(z, h, Om, As, Ob, ns, zeq):
    """Physical parameters -> context vector(s) ordered like CONTEXT_KEYS."""
    return np.stack(np.broadcast_arrays(
        np.asarray(z, dtype=np.float64), h, Om, lnAs10_from_As(As),
        Ob, ns, np.asarray(zeq, dtype=np.float64) / 1000.0), axis=-1)


def context_to_theta(ctx) -> Dict[str, np.ndarray]:
    """Context vector(s) -> dict of physical parameters (no z)."""
    ctx = np.asarray(ctx, dtype=np.float64)
    return dict(h=ctx[..., 1], Om=ctx[..., 2], As=As_from_lnAs10(ctx[..., 3]),
                Ob=ctx[..., 4], ns=ctx[..., 5], zeq=ctx[..., 6] * 1000.0)


def sample_prior(rng: np.random.Generator, n: int, box: Dict = None) -> Dict[str, np.ndarray]:
    """Uniform draws from the (default: ID) prior box, in physical units."""
    box = PRIOR_6D if box is None else box
    draw = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in box.items()}
    draw["As"] = As_from_lnAs10(draw.pop("lnAs10"))
    return draw


def fiducial_theta(**overrides) -> Dict[str, float]:
    th = dict(FIDUCIAL)
    th.update(overrides)
    return th
