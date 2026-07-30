"""Tests for the mass-conserving host carve (lensing.h subhalo_carve, scheme A).

Scheme A (docs/subhalo/mass_conserving_carve_note.md): a subhalo-bearing host is
built at M - sum_i m_i - M_u(r), i.e. reduced by the REALIZED resolved-clump mass
plus the mean unresolved mass, so the total halo mass is M exactly in every
realization. Applies to subhalo_model 3 (production) and the brute reference
(model 1/2 + subhalo_brute; M_u = 0 there). subhalo_carve=False reproduces the
pre-2026-07-22 deterministic (1 - f_s,b)M reduction.

Key invariant: the carve only reorders the host build (which draws no randoms)
after the clump draw, so the RNG stream is identical on/off -> kappa_nosub (full
host + weak part) is bit-for-bit identical whether the carve is on or off. The
direct per-host mass-bookkeeping check (<sum m_i> + M_u = f_s,b M to ~0.1%) and
the Var(kappa) reduction are gated in the sandbox harness (carve_verify.cpp) and
the subhalo_factor_jsd acceptance rerun; here we gate the module-visible behavior.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=6000,
            seed=950_000_777)
# Model 3 is this file's carve target. subhalo_virial must be pinned OFF: it
# defaults ON since the 2026-07-29 paper-default flip and THROWS on models 0-3
# (they reduce the host with M_200-referred tables), so `subhalo_model=3` alone no
# longer runs. Same reason models 1/2 are pinned below.
SUB = dict(subhalo=True, subhalo_model=3, subhalo_virial=False)


def rawdict(**kw):
    return gw.sample_lensing_raw_ml(**BASE, **kw)


def kappa(**kw):
    return np.asarray(rawdict(**kw)["kappa"])


def test_config_default_carve_on():
    assert gw.get_simulator_config()["subhalo_carve"] is True


def test_default_equals_explicit_carve_on():
    """subhalo_carve defaults on, bit-for-bit (unchanged by the 2026-07-29 flip)."""
    a = kappa(**SUB)
    b = kappa(**SUB, subhalo_carve=True)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("carve", [True, False])
def test_deterministic_and_finite(carve):
    a = kappa(**SUB, subhalo_carve=carve)
    b = kappa(**SUB, subhalo_carve=carve)
    assert np.array_equal(a, b), "same seed gave different samples"
    assert np.all(np.isfinite(a))


def test_rng_stream_preserved():
    """The carve reorders the host build (no randoms) after the clump draw, so the
    full-host convergence kappa_nosub (drawn from the identical stream) is bitwise
    identical on/off. This is the invariant that makes the carve auditable."""
    on = rawdict(**SUB, subhalo_carve=True)
    off = rawdict(**SUB, subhalo_carve=False)
    assert np.array_equal(np.asarray(on["kappa_nosub"]),
                          np.asarray(off["kappa_nosub"]))


def test_carve_not_inert():
    """A flag that silently did nothing would pass every gate above: the carve
    changes the smooth-host mass per realization, so full kappa must differ."""
    assert not np.array_equal(kappa(**SUB, subhalo_carve=True),
                              kappa(**SUB, subhalo_carve=False))


def test_mass_conserved_in_mean():
    """RNG is identical on/off, so kappa_on - kappa_off is purely the host-mass
    shift. Carving preserves the MEAN host mass (conservation in the mean), so the
    mean kappa moves only at the Jensen (sub-percent) level -- NOT by ~f_s ~ 14%,
    which is what a broken mass budget (wrong sum_i m_i or M_u) would produce."""
    on = kappa(**SUB, subhalo_carve=True)
    off = kappa(**SUB, subhalo_carve=False)
    rel = abs(on.mean() - off.mean()) / abs(off.mean())
    assert rel < 0.02, f"mean kappa shifted {rel:.3%}; expected Jensen-level (sub-percent)"


def test_no_op_when_subhalo_off():
    """subhalo = False never enters the carve branch, so the flag must not touch
    the stream (this is also the production default path)."""
    assert np.array_equal(kappa(subhalo=False, subhalo_carve=True),
                          kappa(subhalo=False, subhalo_carve=False))


def test_carve_scope_model1_nonbrute_unaffected():
    """Carve is scoped to model 3 and the brute reference only; model-1
    resolved-only (non-brute) is untouched by the flag."""
    assert np.array_equal(
        kappa(subhalo=True, subhalo_model=1, subhalo_virial=False, subhalo_carve=True),
        kappa(subhalo=True, subhalo_model=1, subhalo_virial=False, subhalo_carve=False))


def test_brute_reference_is_carved():
    """The brute truth (model 1 + subhalo_brute) is carved identically (M_u = 0,
    all clumps explicit). Small nsamples: brute resolves every clump and is slow."""
    brute = dict(z=0.5, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=200,
                 seed=17, subhalo=True, subhalo_model=1, subhalo_virial=False,
                 subhalo_brute=True)
    on = np.asarray(gw.sample_lensing_raw_ml(**brute, subhalo_carve=True)["kappa"])
    off = np.asarray(gw.sample_lensing_raw_ml(**brute, subhalo_carve=False)["kappa"])
    assert np.all(np.isfinite(on)) and np.all(np.isfinite(off))
    assert not np.array_equal(on, off)
