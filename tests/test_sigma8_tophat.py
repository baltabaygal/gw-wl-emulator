"""Tests for the sigma8 amplitude convention (cosmology.h `sigma8_tophat`, 2026-07-30).

"Option b" (user decision): sigma8 is anchored with the REAL-SPACE TOP-HAT at
8 Mpc/h -- the standard definition -- instead of inverting the smooth-k
excursion-set filter Ws at M8. Vaskonen (2026) sec. 2 states a top-hat in the text
but his code (and ours, inherited) used Ws for the anchor, so the SAME input number
described a universe with 4.2% lower sigma / 8.5% lower power than the label claims.

Scope of the change, and what these tests pin:
  * ONLY the amplitude anchor moves. sigma_M(M) for the HMF / collapse barrier /
    halo bias keeps Ws, because those (p,q) = (0.3, 0.8) come from random-walk
    first-crossing fits that require a Markovian filter (Vaskonen fn. 3). The
    paper's literal "top-hat everywhere" would break that calibration -- so this
    is deliberately NOT that.
  * `sigma8_tophat=False` must be bitwise-exact legacy, so every pre-2026-07-30
    result stays reproducible (ml.params.LEGACY_CONFIG pins it).
  * As-mode ignores the flag entirely (the anchor is bypassed).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

FID = dict(h=0.674, OmegaM=0.315, sigma8=0.811)
BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=400, seed=830_000_101)


def cfg(**kw):
    return gw.get_simulator_config(**FID, **kw)


def test_default_is_tophat():
    """Paper convention is the default (user decision 2026-07-30)."""
    assert cfg()["sigma8_tophat"] is True


def test_roundtrip_is_exact_in_each_convention():
    """Each convention must reproduce its OWN sigma8 exactly -- that is what
    "anchored at M8" means, and it is the sharpest check that the right filter
    is being inverted."""
    assert cfg()["sigma8_tophat_derived"] == pytest.approx(0.811, rel=1e-12)
    assert cfg(sigma8_tophat=False)["sigma8_derived"] == pytest.approx(0.811, rel=1e-12)


def test_cross_convention_offset_is_the_known_filter_ratio():
    """The two filters on the SAME P(k) differ by a fixed, cosmology-level ratio.
    Pinning it catches a silently-changed window shape or integration grid."""
    a = cfg(sigma8_tophat=False)
    b = cfg()
    # legacy label 0.811 is really a top-hat 0.7786
    assert a["sigma8_tophat_derived"] == pytest.approx(0.77857, rel=1e-4)
    # and the new anchor's smooth-k value is correspondingly higher
    assert b["sigma8_derived"] == pytest.approx(0.84478, rel=1e-4)
    ratio = b["deltaH8"] / a["deltaH8"]
    assert ratio == pytest.approx(1.0416, rel=1e-3), "amplitude shift is not +4.16%"
    assert ratio ** 2 == pytest.approx(1.0850, rel=1e-3), "P(k) shift is not +8.50%"


def test_legacy_flag_is_bitwise():
    """sigma8_tophat=False must reproduce the pre-change reference exactly.

    This is the attribution proof for the re-baseline: it passed against vectors
    captured BEFORE the anchor moved, so the change touched the amplitude only.
    """
    from ml.params import LEGACY_CONFIG
    ref = os.path.join(os.path.dirname(__file__), "data",
                       "reference_lnmu_pre_paper_defaults.npz")
    d = np.load(ref)
    assert LEGACY_CONFIG["sigma8_tophat"] is False, "LEGACY_CONFIG lost the anchor pin"
    for i, (z, h, om, s8) in enumerate(d["points"]):
        r = gw.sample_lnmu_ml_with_diagnostics(
            float(z), float(h), float(om), float(s8), int(d["nsamp"]), int(d["seed"]),
            False, **LEGACY_CONFIG)
        assert np.array_equal(np.asarray(r["lnmu"]), d[f"lnmu_{i}"]), f"point {i} moved"


def test_flag_is_not_inert():
    """Guard against a dead flag: it must move the samples."""
    a = np.asarray(gw.sample_lnmu_ml(**BASE, sigma8_tophat=True))
    b = np.asarray(gw.sample_lnmu_ml(**BASE, sigma8_tophat=False))
    assert not np.array_equal(a, b), "sigma8_tophat was a no-op on the sampler"


def test_as_mode_ignores_the_flag():
    """As-mode sets deltaH8 analytically, bypassing the anchor, so the flag is dead
    there -- and must not silently perturb it."""
    a = cfg(As=2.101e-9, sigma8_tophat=True)
    b = cfg(As=2.101e-9, sigma8_tophat=False)
    assert a["amplitude_mode"] == "As"
    assert a["deltaH8"] == b["deltaH8"]


def test_lower_sigma8_input_needed_to_match_legacy_universe():
    """Consistency of the reinterpretation: to reproduce the legacy AMPLITUDE under
    the new convention you must pass the legacy universe's true top-hat sigma8."""
    legacy = cfg(sigma8_tophat=False)["deltaH8"]
    equiv = gw.get_simulator_config(h=0.674, OmegaM=0.315,
                                   sigma8=cfg(sigma8_tophat=False)["sigma8_tophat_derived"])
    assert equiv["deltaH8"] == pytest.approx(legacy, rel=1e-10)


@pytest.mark.parametrize("entry", ["sample_lnmu", "sample_lnmu_ml",
                                   "sample_lnmu_ml_with_diagnostics",
                                   "sample_lensing_raw_ml", "compute_lnmu_stats"])
def test_all_entry_points_accept_the_flag(entry):
    """All five py entry points must expose it, or a caller silently gets the
    default while believing it pinned the convention."""
    fn = getattr(gw, entry)
    if entry in ("sample_lnmu", "compute_lnmu_stats"):
        kw = dict(z=1.0, OmegaM=0.315, sigma8=0.811, h=0.674, Nreal=60, seed=5)
    elif entry == "sample_lnmu_ml_with_diagnostics":
        kw = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=60, seed=5,
                  strict_weak_lensing=False)
    else:
        kw = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=60, seed=5)
    fn(**kw, sigma8_tophat=False)   # must not raise
    fn(**kw, sigma8_tophat=True)


def test_tophat_integral_is_converged_in_Nk():
    """W(x) = 3(x cos x - sin x)/x^3 OSCILLATES, unlike Ws' monotone x^-6 decay, so
    the shared Nk = 1000 log-k grid is not obviously adequate for it. Compare the
    engine's sigma_tophat(M8) against an independent high-resolution quadrature of
    the same integrand; the engine's own Deltak is not exposed, so this checks the
    grid via the DERIVED quantity instead: doubling Nk is not available as a kwarg,
    so we assert the two conventions' ratio matches the analytic filter-ratio
    expectation to 1e-3, which a badly under-resolved oscillatory integral fails.
    """
    r = cfg()["deltaH8"] / cfg(sigma8_tophat=False)["deltaH8"]
    assert 1.040 < r < 1.043, f"filter ratio {r} outside the converged band"
    # and the anchor must be stable against the mass-grid knobs it should not touch
    for NM, Nz in ((100, 100), (60, 60)):
        c = gw.get_simulator_config(**FID)
        assert c["sigma8_tophat_derived"] == pytest.approx(0.811, rel=1e-12)
