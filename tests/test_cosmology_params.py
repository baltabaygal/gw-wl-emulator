"""Tests for the 1+6d cosmology parameterization (A_s mode + exposed OmegaB/zeq/ns).

Reference data: tests/data/reference_lnmu_pre6d.npz, captured by
tests/capture_reference_lnmu.py from the pre-A_s module build.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "data", "reference_lnmu_pre6d.npz")

PLANCK = dict(h=0.674, OmegaM=0.315, As=2.101e-9, OmegaB=0.0493, zeq=3402.0, ns=0.9649)


def test_backward_compat_bitwise():
    """DEFAULT threshold path reproduces pre-change samples exactly.

    The DEFAULT threshold is the legacy <N>=Nhalos rule again (reverted
    2026-07-10 from the 2026-07-09 flat-1e-3 default). The no-kwarg calls
    below pin the DEFAULT itself to the pre-6d reference — do NOT add an
    explicit kappathr_flat here, or a flipped default would go unguarded
    (the guard this revert is about). The final call checks that the
    explicit legacy selector kappathr_flat=-1 picks the same path."""
    d = np.load(REF)
    for i, (z, h, om, s8) in enumerate(d["points"]):
        r = gw.sample_lnmu_ml_with_diagnostics(
            float(z), float(h), float(om), float(s8), int(d["nsamp"]), int(d["seed"]), False)
        assert np.array_equal(np.asarray(r["lnmu"]), d[f"lnmu_{i}"]), \
            f"point {i} diverged on the DEFAULT threshold path"
    z, h, om, s8 = d["points"][0]
    r = gw.sample_lnmu_ml_with_diagnostics(
        float(z), float(h), float(om), float(s8), int(d["nsamp"]), int(d["seed"]), False,
        kappathr_flat=-1.0)
    assert np.array_equal(np.asarray(r["lnmu"]), d["lnmu_0"]), \
        "explicit kappathr_flat=-1 diverged from the legacy reference"


def test_sigma8_as_round_trip_deltaH8():
    """As-mode at the sigma8-mode's derived As reproduces deltaH8 to ~1 ULP."""
    c = gw.get_simulator_config(h=0.674, OmegaM=0.315, sigma8=0.811)
    assert c["amplitude_mode"] == "sigma8"
    c2 = gw.get_simulator_config(h=0.674, OmegaM=0.315, As=c["As_derived"])
    assert c2["amplitude_mode"] == "As"
    assert c2["deltaH8"] == pytest.approx(c["deltaH8"], rel=1e-14)
    # and the inverse direction closes too
    assert c2["sigma8_derived"] == pytest.approx(0.811, rel=1e-12)


def test_sigma8_as_round_trip_samples():
    """Samples in As-mode match sigma8-mode samples (deltaH8 equal to 1 ULP; the
    lensing pipeline is smooth in deltaH8, so allow rare last-bit divergences)."""
    z, h, om, s8, n, seed = 1.0, 0.674, 0.315, 0.811, 2000, 4242
    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False)["lnmu"])
    As = gw.get_simulator_config(h=h, OmegaM=om, sigma8=s8)["As_derived"]
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False, As=As)["lnmu"])
    assert a.size == b.size
    frac_bitwise = np.mean(a == b)
    # default path is the legacy <N>=Nhalos rule again (reverted 2026-07-10);
    # kept at the relaxed 0.99 bar to tolerate As round-trip last-bit divergences
    # (max |dlnmu| ~ 5e-16), same character on either threshold rule.
    assert frac_bitwise > 0.99, f"only {frac_bitwise:.4%} bitwise-equal"
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_planck_sigma8_sanity():
    """Derived sigma8 at Planck-2018 As lands near 0.81. The code's sigma8 uses a
    smooth-k window (+4.3% vs tophat) and the EH fit + CPT growth, so allow a wide
    band; measured 0.860 on 2026-07-07."""
    c = gw.get_simulator_config(**PLANCK)
    assert 0.75 < c["sigma8_derived"] < 0.88


def test_sigma8_scales_as_sqrt_As():
    c1 = gw.get_simulator_config(As=2.0e-9)
    c2 = gw.get_simulator_config(As=8.0e-9)
    assert c2["sigma8_derived"] / c1["sigma8_derived"] == pytest.approx(2.0, rel=1e-12)


def test_As_takes_precedence_over_sigma8():
    lo = gw.get_simulator_config(sigma8=0.4, As=2.101e-9)
    hi = gw.get_simulator_config(sigma8=1.4, As=2.101e-9)
    assert lo["deltaH8"] == hi["deltaH8"]


@pytest.mark.parametrize("kw,lo,hi", [
    ("ns", 0.90, 1.02),
    ("OmegaB", 0.035, 0.065),
    ("zeq", 2500.0, 4500.0),
])
def test_new_params_affect_normalization(kw, lo, hi):
    """Each newly exposed parameter moves the derived amplitude (sigma8-mode:
    deltaH8 via the sigma integral; they all enter the transfer function)."""
    a = gw.get_simulator_config(**{kw: lo})["deltaH8"]
    b = gw.get_simulator_config(**{kw: hi})["deltaH8"]
    assert a != b


def test_new_params_affect_samples():
    """ns perturbation propagates to actual lnmu draws (same seed)."""
    z, h, om, s8, n, seed = 1.0, 0.674, 0.315, 0.811, 500, 77
    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False)["lnmu"])
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False, ns=1.01)["lnmu"])
    assert not np.array_equal(a, b)


def test_omegar_derived_from_zeq():
    c = gw.get_simulator_config(OmegaM=0.30, zeq=2999.0)
    assert c["OmegaR"] == pytest.approx(0.30 / 3000.0, rel=1e-12)
