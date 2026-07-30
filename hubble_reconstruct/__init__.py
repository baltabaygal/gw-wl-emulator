"""hubble_reconstruct -- HDR: everything downstream of the magnification PDF.

Stage 1 P(mu) -> stage 2 mock catalogue -> stage 3 likelihood -> stage 4 MCMC,
i.e. Vaskonen (2026) sec.3, reimplemented in Python so it can be driven by the
emulator instead of the ~20 s/evaluation stochastic C++.

Design doc: `docs/hubble_diagram_reconstruction.md`. Read it before changing the
plane convention (pmu.py) or the selection handling (selection.py) -- those are
the two documented silent-failure modes.

⚠ STATUS 2026-07-29: runs on a MOCK P(mu). The production 1+6d emulator has not
been retrained, so `pmu.EmulatorPDF` is a stub. Nothing this package currently
prints is a forecast.
"""
from ._repo import (  # noqa: F401
    FIDUCIAL,
    PRIOR_6D,
    PRIOR_VASKONEN,
    PRODUCTION_CONFIG,
    REPO_ROOT,
    THETA3_KEYS,
    THETA6_KEYS,
    theta_full,
)
from . import background, catalogue, likelihood, mcmc, plots, pmu, selection  # noqa: F401

__all__ = [
    "background", "catalogue", "likelihood", "mcmc", "plots", "pmu", "selection",
    "FIDUCIAL", "PRIOR_6D", "PRIOR_VASKONEN", "PRODUCTION_CONFIG",
    "THETA3_KEYS", "THETA6_KEYS", "theta_full", "REPO_ROOT",
]
