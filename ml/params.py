"""Canonical 1+6d parameter space for the emulator: (z_s; h, Om, sigma8, Ob, ns, zeq).

Single source of truth for prior boxes, the ML context layout, and the mapping
between physical parameters and context features. The C++ simulator takes the
physical parameters directly: sigma8 is passed as the amplitude positional and
`As` is left at its default (-1.0), which selects the sigma8 normalization
`deltaH8 = sigma8/sigmaC(M8, 1)`.

Amplitude parameter (2026-07-27): **sigma8**, not A_s. This matches Vaskonen
(2026) in both the paper (inference parameters {Om, h, sigma8}; priors
sigma8 in [0.4, 1.4]) and the upstream `halos` code (`main_lensing.cpp`:
`C.sigma8 = 0.811`). The A_s-mode plumbing in the C++ (`As > 0`) is retained as
an escape hatch but is no longer used by the ML pipeline.

⚠ OPEN DECISION — sigma8 window convention (pending Ville, on holiday as of
2026-07-27). The engine normalizes through `sigmaC`, which uses the smooth-k
window `Ws(x) = 1/(1+(0.43x)^6)`, NOT the real-space top-hat. So sigma8 here is
a smooth-k sigma8: at the fiducial, the top-hat value is 0.7786, i.e. the code's
sigma8 sits +4.2% high (~8.5% in P(k)). Vaskonen's paper §2 states a real-space
top-hat, so his own code and text disagree on this point. We deliberately keep
his CODE convention for now (option "a": zero change, bitwise-safe). The agreed
target is option "b" — anchor the normalization to a top-hat sigma8 at 8 Mpc/h
while keeping Ws for the excursion-set sigma_M(M) — which requires his sign-off
because it shifts P(k) by ~8.5% and redefines the number in his abstract. See
CLAUDE.md §"1+6d parameterization" and memory `sigma8_amplitude_parameter`.

Legacy 1+3d datasets (columns [z, h, OmegaM, sigma8]) remain loadable through
ml.data, as do schema-2.0 A_s-mode datasets (mapped in exactly via the stored
`sigma8_derived`). This module describes the scheme used for new data generation.
"""
from typing import Dict

import numpy as np

# Planck-2018-like fiducial point (sigma8-mode; the Vaskonen 2026 benchmark).
FIDUCIAL = dict(h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)

# In-distribution prior box (training/inference support).
PRIOR_6D = dict(
    h=(0.59, 0.76),
    Om=(0.20, 0.40),
    sigma8=(0.65, 1.05),
    Ob=(0.035, 0.065),
    ns=(0.90, 1.02),
    zeq=(2500.0, 4500.0),
)

# Wider sampling box for dataset generation: (Om, sigma8) extend past the ID box
# to populate the OoD corners. The sigma8 margins reproduce Vaskonen (2026)'s
# MCMC prior exactly; his Om prior is marginally wider (0.15, 0.47).
WIDE_6D = dict(PRIOR_6D)
WIDE_6D["Om"] = (0.15, 0.45)
WIDE_6D["sigma8"] = (0.40, 1.40)

# ML context layout. z stays at index 0 (train_smooth.py relies on it). Note the
# first four columns now coincide with the legacy 1+3d layout (z, h, Om, sigma8).
CONTEXT_KEYS = ("z", "h", "Om", "sigma8", "Ob", "ns", "zeq_k")
CONTEXT_DIM = len(CONTEXT_KEYS)

# HDF5 dataset keys for the physical parameters (generate_dataset.py schema 2.1)
PARAM_KEYS_6D = ("h", "OmegaM", "sigma8", "OmegaB", "ns", "zeq")


# --- deprecated A_s helpers, kept only to read schema-2.0 datasets -------------
def lnAs10_from_As(As):
    return np.log(1.0e10 * np.asarray(As, dtype=np.float64))


def As_from_lnAs10(lnAs10):
    return np.exp(np.asarray(lnAs10, dtype=np.float64)) * 1.0e-10


def theta_to_context(z, h, Om, sigma8, Ob, ns, zeq):
    """Physical parameters -> context vector(s) ordered like CONTEXT_KEYS."""
    return np.stack(np.broadcast_arrays(
        np.asarray(z, dtype=np.float64), h, Om, sigma8,
        Ob, ns, np.asarray(zeq, dtype=np.float64) / 1000.0), axis=-1)


def context_to_theta(ctx) -> Dict[str, np.ndarray]:
    """Context vector(s) -> dict of physical parameters (no z)."""
    ctx = np.asarray(ctx, dtype=np.float64)
    return dict(h=ctx[..., 1], Om=ctx[..., 2], sigma8=ctx[..., 3],
                Ob=ctx[..., 4], ns=ctx[..., 5], zeq=ctx[..., 6] * 1000.0)


def sample_prior(rng: np.random.Generator, n: int, box: Dict = None) -> Dict[str, np.ndarray]:
    """Uniform draws from the (default: ID) prior box, in physical units."""
    box = PRIOR_6D if box is None else box
    return {k: rng.uniform(lo, hi, n) for k, (lo, hi) in box.items()}


def fiducial_theta(**overrides) -> Dict[str, float]:
    th = dict(FIDUCIAL)
    th.update(overrides)
    return th
