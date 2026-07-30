"""Tests for the filament clustering bias (lensing.h fil_bias).

Filaments collapse from a flatter first-crossing barrier than halos
(`pFCfil`, (p,q) = (0, 0.7) vs the halo `pFC` (0.3, 0.8)), so they are less
strongly biased. `fil_bias` makes the correlated field modulate the filament
counts by the PBS bias of the code's own filament mass function,
`cosmology::filbias`, instead of borrowing `halobias`.

Scope of what is gated here: the WIRING invariants (default bitwise, no-op in
the legacy layer, live in the correlated layer, present on all five entry
points, composes with the production config). The size of the physical shift is
NOT gated here -- it is below MC noise at unit-test sample counts, and is
measured on a high-N ensemble by
`scripts/convergence/fil_bias_ab.py` (see data/results/fil_bias/report.md).

The pre-change bitwise reference gate is test_cosmology_params.py
::test_backward_compat_bitwise, whose stored reference predates fil_bias.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=4000,
            seed=770_000_101)

# The production clustering configuration (ml.params.PRODUCTION_CONFIG).
PROD_BIAS = dict(bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True)


def raw(**kw):
    return np.asarray(gw.sample_lensing_raw_ml(**BASE, **kw)["kappa"])


def test_default_is_fil_bias_on():
    """The default path uses the filament bias, bit-for-bit (paper default since
    2026-07-29; the draft states the (p,q) = (0, 0.7) filament barrier, so with the
    flag off that sentence was false in the code). Was OFF before the flip."""
    for extra in (dict(bias_model=1),
                  dict(bias_model=1, bias_weak=True),
                  dict(bias_model=1, bias_window=1, bias_Rperp=20000.0),
                  PROD_BIAS):
        a = raw(**extra)
        b = raw(fil_bias=True, **extra)
        assert np.array_equal(a, b), f"default != explicit fil_bias=True for {extra}"


def test_config_reports_fil_bias():
    assert gw.get_simulator_config()["fil_bias"] is True


def test_noop_in_legacy_bias_layer():
    """bias_model=0 has no correlated field, so fil_bias must be inert there.

    ⚠ bias_weak AND bias_window must both be turned off explicitly: both default
    to the correlated-layer values since 2026-07-29 and both throw against
    bias_model=0, so a single-flag override raises instead of testing the invariant.
    This is the coupling that ml.params.LEGACY_CONFIG exists to splat in one go."""
    legacy_layer = dict(bias_model=0, bias_weak=False, bias_window=0)
    a = raw(**legacy_layer, fil_bias=False)
    b = raw(**legacy_layer, fil_bias=True)
    assert np.array_equal(a, b), "fil_bias changed the legacy iid layer"


def test_live_in_correlated_layer():
    """Guard against a dead flag: at bias_model=1 it MUST move the stream.

    Switching the filament bias changes their Poisson means, which changes the
    realized filament counts and hence the shared RNG stream downstream.
    """
    a = raw(bias_model=1, fil_bias=False)
    b = raw(bias_model=1, fil_bias=True)
    assert not np.array_equal(a, b), "fil_bias=True was a no-op at bias_model=1"


def test_live_in_production_config():
    a = raw(fil_bias=False, **PROD_BIAS)
    b = raw(fil_bias=True, **PROD_BIAS)
    assert not np.array_equal(a, b), "fil_bias=True was a no-op in the production config"


@pytest.mark.parametrize("window", [0, 1, 2])
def test_deterministic_and_finite(window):
    kw = dict(bias_model=1, bias_window=window, fil_bias=True)
    a = raw(**kw)
    b = raw(**kw)
    assert np.array_equal(a, b), "same seed gave different samples"
    assert np.all(np.isfinite(a))

    lnmu = np.asarray(gw.sample_lnmu_ml(**BASE, **kw))
    assert np.all(np.isfinite(lnmu[~np.isnan(lnmu)]))


def test_wired_on_all_entry_points():
    """All five py entry points must accept fil_bias AND act on it.

    NOTE the two argument orders: sample_lnmu / compute_lnmu_stats take
    (z, OmegaM, sigma8, h, Nreal, ...) while the *_ml family takes
    (z, h, OmegaM, sigma8, nsamples, ...). Getting this wrong silently swaps
    h and OmegaM, which kills the field power (see CLAUDE.md).
    """
    kw = dict(bias_model=1)
    # legacy positional order, keyword-explicit so it cannot be mis-ordered
    LEG = dict(z=BASE["z"], OmegaM=BASE["OmegaM"], sigma8=BASE["sigma8"],
               h=BASE["h"], Nreal=BASE["nsamples"], seed=BASE["seed"])

    a = np.asarray(gw.sample_lnmu(**LEG, fil_bias=False, **kw))
    b = np.asarray(gw.sample_lnmu(**LEG, fil_bias=True, **kw))
    assert not np.array_equal(a, b), "sample_lnmu ignored fil_bias"

    a = np.asarray(gw.sample_lnmu_ml(**BASE, fil_bias=False, **kw))
    b = np.asarray(gw.sample_lnmu_ml(**BASE, fil_bias=True, **kw))
    assert not np.array_equal(a, b), "sample_lnmu_ml ignored fil_bias"

    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        **BASE, fil_bias=False, **kw)["lnmu"])
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        **BASE, fil_bias=True, **kw)["lnmu"])
    assert not np.array_equal(a, b), "sample_lnmu_ml_with_diagnostics ignored fil_bias"

    assert not np.array_equal(raw(fil_bias=False, **kw), raw(fil_bias=True, **kw)), \
        "sample_lensing_raw_ml ignored fil_bias"

    a = gw.compute_lnmu_stats(**LEG, fil_bias=False, **kw)
    b = gw.compute_lnmu_stats(**LEG, fil_bias=True, **kw)
    assert a != b, "compute_lnmu_stats ignored fil_bias"


def test_composes_with_subhalo_model5():
    """The full production config (substructure + clustering) runs and is finite."""
    kw = dict(subhalo=True, subhalo_model=5, subhalo_carve=True,
              subhalo_kappathr_factor=0.1, kappa_anchor=1, fil_bias=True, **PROD_BIAS)
    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        **dict(BASE, nsamples=400), **kw)["lnmu"])
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        **dict(BASE, nsamples=400), **kw)["lnmu"])
    assert np.array_equal(a, b), "production config is not reproducible at fixed seed"
    assert np.all(np.isfinite(a[~np.isnan(a)]))
    assert np.isfinite(a[~np.isnan(a)]).sum() > 0
