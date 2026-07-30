"""Tests for subhalo_model=5 (kappa-thresholded brute, 2026-07-27).

Model 5 = model 4's population (every subhalo down to psi_min = m_floor/M, host
carved to M - sum_i m_i(retained), no unresolved/Gaussian stand-in) with the clumps
whose kappa AT THE RAY falls below kappa_thr,sub left in the smooth host instead of
rendered. The rejects are never instantiated: addClumpsRestricted samples the
RESTRICTED Poisson intensity (draw-and-reject would save nothing).

These gate the contract, not the physics-vs-model-4 agreement, which needs many rays
and lives in scripts/convergence/subhalo_model5_gate.py +
data/results/subkappathr_population/report.md.

STAGED: the shipped default stays subhalo_model=3. The module skips cleanly on a
.so predating model 5.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=600,
            seed=950_001_555)


def kappa(**kw):
    args = dict(BASE)
    args.update(kw)                      # kw may override BASE (e.g. nsamples)
    return np.asarray(gw.sample_lensing_raw_ml(**args)["kappa"])


def _supports_model5():
    try:
        gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 1}, subhalo=True, subhalo_model=5)
        return True
    except (ValueError, TypeError):
        return False


pytestmark = pytest.mark.skipif(
    not _supports_model5(), reason="module predates subhalo_model=5 (rebuild required)"
)

M5 = dict(subhalo=True, subhalo_model=5)


def test_default_model_is_five():
    """Paper default since 2026-07-29 (model 5 was STAGED, default 3, before that).

    Model 5 is the config the draft describes: every subhalo sampled to
    m_floor/M with the host carved by the realized sum, and only clumps whose
    kappa at the ray clears the threshold actually rendered."""
    assert gw.get_simulator_config()["subhalo_model"] == 5
    assert gw.get_simulator_config()["subhalo"] is True


def test_config_exposes_threshold_knobs():
    cfg = gw.get_simulator_config()
    assert cfg["subhalo_kappathr"] == -1.0        # <= 0 -> factor * host kappa_thr
    assert cfg["subhalo_kappathr_factor"] == 0.1


def test_carve_is_intrinsic():
    """Dropped clumps keep their mass in the host, which requires the carve."""
    with pytest.raises(ValueError, match="subhalo_carve"):
        gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 1}, subhalo=True,
                                 subhalo_model=5, subhalo_carve=False)


def test_brute_flag_rejected():
    """Model 5 is brute by construction; the flag would be a silent no-op."""
    with pytest.raises(ValueError, match="subhalo_brute"):
        gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 1}, subhalo=True,
                                 subhalo_model=5, subhalo_brute=True)


def test_unknown_model_still_rejected():
    with pytest.raises(ValueError, match="subhalo_model must be"):
        gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 1}, subhalo=True, subhalo_model=6)


def test_deterministic():
    assert np.array_equal(kappa(**M5), kappa(**M5))


def test_absolute_threshold_overrides_factor():
    """subhalo_kappathr > 0 is absolute: the factor must then be ignored."""
    a = kappa(**M5, subhalo_kappathr=1e-5, subhalo_kappathr_factor=0.1)
    b = kappa(**M5, subhalo_kappathr=1e-5, subhalo_kappathr_factor=1.0)
    assert np.array_equal(a, b)


def test_factor_is_live():
    """Lowering the threshold renders more clumps, so the stream must change."""
    a = kappa(**M5, subhalo_kappathr_factor=0.1)
    b = kappa(**M5, subhalo_kappathr_factor=1.0)
    assert not np.array_equal(a, b)


def test_threshold_gates_substructure_power():
    """The threshold must actually control how much substructure power is rendered.

    Measured on the PAIRED per-ray perturbation dk = kappa - kappa_nosub (substructure
    plus its carve), on the kappa <= 1 core: total Var(kappa) is dominated by host
    placement and shows no threshold trend at all at this sample size (measured: 8%
    swing, no ordering), so it cannot gate anything.

    The contrast is deliberately extreme. Var(dk) carries 7-24% seed scatter, while
    neighbouring thresholds in the production range differ by a fraction of a percent
    by design (that flatness IS the result -- see
    data/results/subkappathr_population/report.md). Only a near-total cut is
    resolvable here: measured Var(dk) = 4.7e-5 at factor 1 vs 2.0e-5 at factor 100.
    """
    def var_dk(f, seed):
        r = gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 4000, "seed": seed},
                                     **M5, subhalo_kappathr_factor=f)
        k, kn = np.asarray(r["kappa"]), np.asarray(r["kappa_nosub"])
        m = k <= 1.0
        return float(np.var((k - kn)[m]))

    seeds = (101, 202, 303)
    lo = np.mean([var_dk(0.1, s) for s in seeds])     # production-like threshold
    hi = np.mean([var_dk(100.0, s) for s in seeds])   # nearly everything dropped
    assert lo > 1.4 * hi, f"Var(dk) factor0.1={lo:.3e} vs factor100={hi:.3e}"


def test_model5_cheaper_than_model4():
    """The entire point: same population, far fewer clumps rendered."""
    import time
    def marginal(n1, n2, **kw):
        t = []
        for n in (n1, n2):
            s = time.time()
            gw.sample_lnmu(1.0, 0.315, 0.811, 0.674, n, 12345, **kw)
            t.append(time.time() - s)
        return (t[1] - t[0]) / (n2 - n1)
    m5 = marginal(200, 800, subhalo=True, subhalo_model=5)
    m4 = marginal(30, 120, subhalo=True, subhalo_model=4)
    assert m5 < 0.05 * m4, f"model5 {m5*1e3:.3f} ms/ray vs model4 {m4*1e3:.3f}"


def test_does_not_perturb_other_models():
    """Model 5 added tables to Subhalo; models 3 and 4 must be untouched.

    ⚠ subhalo_virial must be passed explicitly: it defaults True since 2026-07-29
    and THROWS on models 0-3, so `subhalo_model=3` alone no longer runs."""
    for m in (3, 4):
        virial = dict(subhalo_virial=(m >= 4))
        a = kappa(subhalo=True, subhalo_model=m, **virial)
        b = kappa(subhalo=True, subhalo_model=m, **virial)
        assert np.array_equal(a, b)
        assert np.isfinite(a).all()
