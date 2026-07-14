"""Tests for the mean-kappa anchor options (2026-07-13, batch-anchor bug).

Self-consistent (no reference files, platform-independent):
the DEFAULT path must be bit-identical to kappa_anchor=0, the robust mode
must reduce to legacy when the cut excludes nothing, and the external mode
must reproduce legacy when fed the batch's own sequential mean.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

KW = dict(z=1.0, OmegaM=0.315, sigma8=0.811, h=0.674, Nreal=8000, seed=424242)


def test_default_is_legacy_anchor():
    a = gw.sample_lnmu(**KW)
    b = gw.sample_lnmu(**KW, kappa_anchor=0)
    assert np.array_equal(a, b)


def test_robust_reduces_to_legacy_at_infinite_cut():
    a = gw.sample_lnmu(**KW)
    b = gw.sample_lnmu(**KW, kappa_anchor=1, kappa_anchor_cut=1e30)
    assert np.array_equal(a, b)


def test_external_anchor_reproduces_legacy_with_batch_mean():
    raw = gw.sample_lensing_raw_ml(
        z=KW["z"], h=KW["h"], OmegaM=KW["OmegaM"], sigma8=KW["sigma8"],
        nsamples=KW["Nreal"], seed=KW["seed"])
    mk = 0.0  # C++ sums sequentially; np.mean uses pairwise summation (differs in ULP)
    for v in raw["kappa"]:
        mk += float(v)
    mk /= len(raw["kappa"])
    a = gw.sample_lnmu(**KW)
    b = gw.sample_lnmu(**KW, kappa_anchor=2, kappa_anchor_value=mk)
    assert np.array_equal(a, b)


def test_robust_anchor_removes_monster_shift():
    """With kappa>1 rays in the batch, mode 1 shifts the body UP vs mode 0 by
    ~ +2*sum(kappa>1)/n (first order); without such rays the modes agree."""
    for seed in range(100, 200):
        raw = gw.sample_lensing_raw_ml(
            z=5.0, h=0.674, OmegaM=0.315, sigma8=0.811,
            nsamples=15000, seed=seed)
        excess = raw["kappa"][raw["kappa"] > 1.0]
        if excess.sum() > 5.0:  # a real monster batch
            break
    else:
        import pytest
        pytest.skip("no monster batch found in seed scan")
    kw = dict(z=5.0, OmegaM=0.315, sigma8=0.811, h=0.674,
              Nreal=15000, seed=seed)
    a0 = gw.sample_lnmu(**kw)
    a1 = gw.sample_lnmu(**kw, kappa_anchor=1)
    pred = 2.0 * excess.sum() / 15000
    obs = np.median(a1) - np.median(a0)
    assert obs > 0.25 * pred, (obs, pred)
    assert obs < 2.0 * pred, (obs, pred)


def test_config_reports_anchor():
    c = gw.get_simulator_config()
    assert c["kappa_anchor"] == 0
    assert c["kappa_anchor_cut"] == 1.0
    assert c["kappa_anchor_value"] == 0.0
