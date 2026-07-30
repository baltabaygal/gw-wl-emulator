"""Tests for subhalo_model=4 (supervisor's simplified production model, 2026-07-23).

Model 4 = every subhalo sampled individually down to the absolute floor
psi_min = m_floor/M, host carved to M - sum_i m_i, NO unresolved term (no
M_u/kappa_u/Wsub, no dynamic floor, no subhalo_factor). It is by construction the
same computation as model 1 + subhalo_brute + subhalo_carve, exposed as one named
model; the defining invariant gated here is bitwise equality with that flag combo.

Model 4 is the REFERENCE, not the default. The 2026-07-29 paper-default flip made
subhalo_model=5 the shipped default -- same population, 217x cheaper, because only
clumps whose kappa at the ray clears the threshold are rendered. Model 4 stays as the
brute truth that model 5 is gated against. The whole module skips cleanly on a
pre-2026-07-23 .so (which rejects subhalo_model=4).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=4000,
            seed=950_001_444)


def rawdict(**kw):
    return gw.sample_lensing_raw_ml(**BASE, **kw)


def kappa(**kw):
    return np.asarray(rawdict(**kw)["kappa"])


def _module_supports_model4():
    try:
        gw.sample_lensing_raw_ml(**{**BASE, "nsamples": 1}, subhalo=True, subhalo_model=4)
        return True
    except ValueError:
        return False


pytestmark = pytest.mark.skipif(
    not _module_supports_model4(),
    reason="build/gwlensing .so predates subhalo_model=4 (rebuild with `make build`)",
)


def test_default_model_is_five_not_four():
    """Model 4 is the REFERENCE, not the default. The 2026-07-29 paper-default flip
    made model 5 the default -- same population, 217x cheaper (only clumps whose
    kappa at the ray clears the threshold are rendered)."""
    assert gw.get_simulator_config()["subhalo_model"] == 5


def test_model4_equals_model1_brute_carve():
    """Defining invariant: model 4 == model 1 + brute + carve, bit-for-bit
    (kappa AND the paired no-substructure baseline)."""
    # subhalo_virial pinned OFF on BOTH arms: it defaults ON since 2026-07-29, it
    # throws on model 1, and leaving arm `a` on the default would compare a virial
    # model 4 against an M_200 model 1 and fail. The invariant under test is the
    # carve equivalence, not the virial convention.
    a = rawdict(subhalo=True, subhalo_model=4, subhalo_virial=False)
    b = rawdict(subhalo=True, subhalo_model=1, subhalo_virial=False,
                subhalo_brute=True, subhalo_carve=True)
    assert np.array_equal(np.asarray(a["kappa"]), np.asarray(b["kappa"]))
    assert np.array_equal(np.asarray(a["kappa_nosub"]), np.asarray(b["kappa_nosub"]))


def test_model4_requires_carve():
    """The carve is intrinsic to model 4; carve-off must throw, not mis-simulate."""
    with pytest.raises(ValueError):
        rawdict(subhalo=True, subhalo_model=4, subhalo_carve=False)


def test_model4_ignores_brute_and_factor():
    """subhalo_brute and subhalo_factor are dead knobs under model 4."""
    a = kappa(subhalo=True, subhalo_model=4)
    b = kappa(subhalo=True, subhalo_model=4, subhalo_brute=True)
    c = kappa(subhalo=True, subhalo_model=4, subhalo_factor=1.0e-5)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_model4_deterministic_and_finite():
    a = kappa(subhalo=True, subhalo_model=4)
    b = kappa(subhalo=True, subhalo_model=4)
    assert np.array_equal(a, b), "same seed gave different samples"
    assert np.all(np.isfinite(a))


def test_model4_not_inert():
    """Model 4 must actually add substructure scatter over the subhalo-off run
    (compare on the kappa<=1 body; raw tails are monster-ray junk)."""
    d = rawdict(subhalo=True, subhalo_model=4)
    k = np.asarray(d["kappa"])
    kns = np.asarray(d["kappa_nosub"])
    m = k <= 1.0
    dsub = (k - kns)[m]
    assert dsub.std() > 1e-4, "substructure component is inert"


def test_model4_floor_moves_substructure():
    """Raising the floor to 1e10 must remove substructure scatter (sanity that
    m_floor is the live floor under model 4, not the dynamic r_thr floor)."""
    d_lo = rawdict(subhalo=True, subhalo_model=4, m_floor=1e8)
    d_hi = rawdict(subhalo=True, subhalo_model=4, m_floor=1e10)

    def dsub_std(d):
        k = np.asarray(d["kappa"])
        kns = np.asarray(d["kappa_nosub"])
        m = k <= 1.0
        return (k - kns)[m].std()

    assert dsub_std(d_hi) < dsub_std(d_lo)


def test_subhalo_off_unaffected():
    """subhalo_model is dead when subhalo=False; must be bitwise the default run."""
    a = kappa(subhalo=False)
    b = kappa(subhalo=False, subhalo_model=4)
    assert np.array_equal(a, b)
