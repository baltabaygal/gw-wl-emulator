"""Tests for the bias-field smoothing window selector (lensing.h bias_window).

  0 = transverse disk on k_perp (legacy, bitwise default)
  1 = spherical top-hat on |k|
  2 = Gaussian on |k|

The bitwise gate here is self-consistency (default == explicit 0); the
pre-change reference gate lives in test_cosmology_params.py and the
pristine-build stream comparison is recorded in
data/results/bias_window/report.md.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=4000,
            seed=950_000_301)


def raw(**kw):
    return np.asarray(gw.sample_lensing_raw_ml(**BASE, **kw)["kappa"])


def test_default_is_disk_window():
    """The default path must be the legacy disk window, bit-for-bit."""
    for extra in (dict(), dict(bias_weak=True), dict(bias_Rperp=3000.0)):
        a = raw(bias_model=1, **extra)
        b = raw(bias_model=1, bias_window=0, **extra)
        assert np.array_equal(a, b), f"default != explicit bias_window=0 for {extra}"


def test_config_reports_bias_window():
    assert gw.get_simulator_config()["bias_window"] == 0


@pytest.mark.parametrize("window", [1, 2])
def test_requires_bias_model_1(window):
    with pytest.raises(Exception):
        raw(bias_model=0, bias_window=window)


@pytest.mark.parametrize("window", [-1, 3])
def test_rejects_unknown_window(window):
    with pytest.raises(Exception):
        raw(bias_model=1, bias_window=window)


@pytest.mark.parametrize("window", [1, 2])
@pytest.mark.parametrize("weak", [False, True])
def test_deterministic_and_finite(window, weak):
    a = raw(bias_model=1, bias_window=window, bias_weak=weak)
    b = raw(bias_model=1, bias_window=window, bias_weak=weak)
    assert np.array_equal(a, b), "same seed gave different samples"
    assert np.all(np.isfinite(a))
    lnmu = np.asarray(gw.sample_lnmu_ml(**BASE, bias_model=1,
                                        bias_window=window, bias_weak=weak))
    assert np.all(np.isfinite(lnmu)) and lnmu.size == BASE["nsamples"]


@pytest.mark.parametrize("window", [1, 2])
def test_window_is_not_inert(window):
    """A selector that silently did nothing would pass every gate above."""
    assert not np.array_equal(raw(bias_model=1), raw(bias_model=1, bias_window=window))


@pytest.mark.parametrize("window", [1, 2])
def test_no_op_when_bias_off(window):
    """bias = 0 skips the field build entirely, so the window must not touch
    the stream (same invariant the bias_weak gate checks)."""
    assert np.array_equal(raw(bias_model=1, bias=False),
                          raw(bias_model=1, bias=False, bias_window=window))


# NOTE on window ordering (sigma^2_field: disk > top-hat > Gaussian at fixed R):
# that is a sharp, deterministic property of the field and is gated directly
# against the production C++ by playground/bias_field/check_cpp_vs_replica.py.
# It is deliberately NOT asserted here through Var(kappa): the clustering share
# of Var(kappa) is ~5% at z_s = 1, so the between-window differences sit far
# below the seed noise of any sample size this suite can afford (CLAUDE.md
# item 13: raw/clipped variance at <1e5 rays is seed junk — ensembles only).
