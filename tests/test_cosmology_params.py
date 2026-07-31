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
    """DEFAULT threshold path reproduces the reference samples exactly.

    ⚠ The reference was RE-BASELINED 2026-07-29 for the PAPER-DEFAULT FLIP: the
    compiled-in defaults are now the config the draft describes (see
    tests/capture_reference_lnmu.py for the history and the verification protocol).
    The pre-flip vectors are kept in `reference_lnmu_pre_paper_defaults.npz` and are
    still guarded — by `test_legacy_physics_bitwise` below, via ml.params.LEGACY_CONFIG.
    ⚠ The reference was ALSO re-baselined 2026-07-28 for `cosmology::halobias`
    q 0.75 -> 0.8; those vectors are in `reference_lnmu_pre_halobias.npz`.
    A failure here means the default path moved — treat it as a real regression unless
    you know which intentional change caused it.

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


def test_legacy_physics_bitwise():
    """The PRE-2026-07-29 physics is still exactly reproducible via LEGACY_CONFIG.

    This is what stops the paper-default flip from being a one-way door. It also
    doubles as the attribution proof for that flip: it passed against vectors
    captured BEFORE the defaults moved, so the flip changed defaults only and no
    physics. Keep it even after the legacy arms are retired — it is the cheapest
    guard on the reference arm of every A/B in the repo.
    """
    from ml.params import LEGACY_CONFIG
    ref = os.path.join(os.path.dirname(__file__), "data",
                       "reference_lnmu_pre_paper_defaults.npz")
    d = np.load(ref)
    for i, (z, h, om, s8) in enumerate(d["points"]):
        r = gw.sample_lnmu_ml_with_diagnostics(
            float(z), float(h), float(om), float(s8), int(d["nsamp"]), int(d["seed"]),
            False, **LEGACY_CONFIG)
        assert np.array_equal(np.asarray(r["lnmu"]), d[f"lnmu_{i}"]), \
            f"point {i} diverged on the explicit LEGACY_CONFIG path"


def test_defaults_are_the_paper_config():
    """The compiled-in defaults ARE ml.params.PRODUCTION_CONFIG (flipped 2026-07-29).

    Guards both directions: a C++ default drifting away from the paper config, and
    PRODUCTION_CONFIG being edited without the corresponding C++ change. Before the
    flip the companion test asserted the two DIFFERED; that assertion is now
    inverted on purpose.
    """
    from ml.params import PRODUCTION_CONFIG
    cfg = gw.get_simulator_config(h=0.674, OmegaM=0.315, sigma8=0.811)
    for k, v in PRODUCTION_CONFIG.items():
        assert k in cfg, f"{k} not reported by get_simulator_config"
        assert cfg[k] == v, f"default {k}: {cfg[k]} != paper config {v}"


def test_sigma8_as_round_trip_deltaH8():
    """As-mode at the sigma8-mode's derived As reproduces deltaH8 to ~1 ULP."""
    c = gw.get_simulator_config(h=0.674, OmegaM=0.315, sigma8=0.811)
    assert c["amplitude_mode"] == "sigma8"
    c2 = gw.get_simulator_config(h=0.674, OmegaM=0.315, As=c["As_derived"])
    assert c2["amplitude_mode"] == "As"
    assert c2["deltaH8"] == pytest.approx(c["deltaH8"], rel=1e-14)
    # and the inverse direction closes too. ⚠ Since the 2026-07-30 "option b"
    # change the CONVENTIONAL sigma8 is `sigma8_tophat_derived`; `sigma8_derived`
    # is the smooth-k (excursion-set filter) value of the same P(k), which at the
    # top-hat-anchored fiducial is 0.8448, not 0.811.
    assert c2["sigma8_tophat_derived"] == pytest.approx(0.811, rel=1e-12)
    assert c2["sigma8_derived"] == pytest.approx(0.84478, rel=1e-4)


def test_sigma8_as_round_trip_samples():
    """Samples in As-mode match sigma8-mode samples (deltaH8 equal to 1 ULP; the
    lensing pipeline is smooth in deltaH8, so allow rare last-bit divergences)."""
    z, h, om, s8, n, seed = 1.0, 0.674, 0.315, 0.811, 2000, 4242
    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False)["lnmu"])
    As = gw.get_simulator_config(h=h, OmegaM=om, sigma8=s8)["As_derived"]
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(z, h, om, s8, n, seed, False, As=As)["lnmu"])
    assert a.size == b.size
    # ⚠ The bitwise bar was dropped on 2026-07-29 (paper-default flip). Under the
    # LEGACY defaults this round trip was 100% bitwise, because a 1-ULP deltaH8
    # difference did not change any RNG draw count. Under the paper defaults the
    # realized subhalo and clustered-count draws depend on deltaH8, so a last-bit
    # change DECORRELATES THE RNG STREAM and 0% of rays are bitwise equal --
    # measured max |dlnmu| = 1.8e-7, i.e. stream divergence, not an amplitude error.
    # The physical invariant (the two amplitude modes describe the same cosmology)
    # is therefore checked on the value, not on the stream. lnmu crosses zero, so
    # this must be an ABSOLUTE tolerance; rtol would blow up near lnmu = 0.
    assert np.abs(a - b).max() < 1e-5, \
        f"As/sigma8 modes diverge by {np.abs(a - b).max():.2e} in lnmu"
    # The stream-level identity still holds where nothing is count-dependent.
    # ⚠ As must be re-derived under the LEGACY anchor: LEGACY_CONFIG pins
    # sigma8_tophat=False, so feeding it the top-hat-derived As above would compare
    # two different AMPLITUDES and fail for a reason that has nothing to do with
    # the round trip (this is exactly how it broke on 2026-07-30).
    from ml.params import LEGACY_CONFIG
    As_leg = gw.get_simulator_config(h=h, OmegaM=om, sigma8=s8,
                                     sigma8_tophat=False)["As_derived"]
    al = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        z, h, om, s8, n, seed, False, **LEGACY_CONFIG)["lnmu"])
    bl = np.asarray(gw.sample_lnmu_ml_with_diagnostics(
        z, h, om, s8, n, seed, False, As=As_leg, **LEGACY_CONFIG)["lnmu"])
    # ⚠ Was `array_equal`. Relaxed 2026-07-30 with the z-GRID EXTENSION (zmax 10.01 ->
    # 12.3412, Nz 100 -> 103): the extension is exact on the shared nodes, but it moves
    # sigma_W at the ~1e-9 level, which is enough to flip a handful of last-bit-sensitive
    # draws. Measured here: 99.85% of rays still bit-identical, max |dlnmu| = 3.5e-16
    # (one ULP of lnmu ~ 0.07), sd ratio 1.000000. That is last-bit noise, not an
    # amplitude error, and exact equality between two routes that agree to 1 ULP by
    # construction is a knife-edge bar -- the same reasoning that already made the
    # paper-default check above a tolerance. ABSOLUTE tolerance because lnmu crosses zero.
    d = np.abs(al - bl).max()
    assert d < 1e-12, f"As round trip diverges by {d:.2e} in lnmu on the legacy path"


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
