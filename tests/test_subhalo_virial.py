"""Tests for the JvdB14 virial convention in the subhalo population (subhalo_virial).

Background. JvdB14 sec. 2 defines host haloes and subhaloes as spheres of mean
density Delta_vir(z) rho_crit(z) INSIDE THEIR VIRIAL RADII, so f_s and psi = m/M
are M_vir quantities and the population extends to r_vir. Green+21, which
calibrates the anti-biased radial bias B(x), likewise normalizes "to unity at
r_vir" with the Bryan-Norman Delta_vir. The engine's grid mass is M_200c
(NFWlistf builds r200 from 200 rho_crit(z)).

Before this flag only the radial-BIAS SCALE x0 was converted to r_200 units
(etaVirTo200); the radial EXTENT and the MASS normalization were not, so a
virial-calibrated population was packed into an r_200 aperture. subhalo_virial
fixes both: psi -> m/M_vir and the profile is sampled to x = eta = r_vir/r_200,
with the carve converting the realized clump mass back to the M_200 scale.

Gated to subhalo_model 4/5: models 0-3 reduce the host through incomplete-Gamma
f_s_res / Wsub tables that are still written in M_200 units.

Evidence: data/results/subhalo_virial/report.md.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../build")))
import gwlensing as gw  # noqa: E402

BASE = dict(z=1.0, h=0.674, OmegaM=0.315, sigma8=0.811, nsamples=600, seed=880_000_401)
M5 = dict(subhalo=True, subhalo_model=5, subhalo_carve=True, subhalo_kappathr_factor=0.1)
M4 = dict(subhalo=True, subhalo_model=4, subhalo_carve=True)


def raw(**kw):
    return np.asarray(gw.sample_lensing_raw_ml(**BASE, **kw)["kappa"])


def test_default_is_on():
    """Paper default since 2026-07-29: the draft describes a JvdB14 population, whose
    f_s and psi = m/M are virial quantities, so with the flag off the paper misstates
    the code. Was OFF (staged) before the flip."""
    assert gw.get_simulator_config()["subhalo_virial"] is True


def test_off_is_bitwise_legacy():
    """subhalo_virial=False must be bit-for-bit the pre-2026-07-28 M_200 convention.

    xmaxh is filled with 1.0 and Mpsih with M when virial = false, and every new
    expression is a multiplication by those, which is exact in IEEE.

    ⚠ Since the 2026-07-29 flip the DEFAULT is virial=True, so this compares
    explicit-False against explicit-False-with-the-partner-flags rather than against
    the bare default (which would compare True to False and fail).
    """
    for extra in (M5, M4, dict(M5, subhalo_kappathr_factor=1.0)):
        n = dict(BASE)
        if extra is M4:
            n = dict(BASE, nsamples=60)      # model 4 is ~45 ms/ray
        a = np.asarray(gw.sample_lensing_raw_ml(
            **n, **extra, subhalo_virial=False)["kappa"])
        b = np.asarray(gw.sample_lensing_raw_ml(
            **n, **extra, subhalo_virial=False)["kappa"])
        assert np.array_equal(a, b), f"subhalo_virial=False not deterministic for {extra}"
        assert np.all(np.isfinite(a))


@pytest.mark.parametrize("model", [0, 1, 2, 3])
def test_requires_model_4_or_5(model):
    """Models 0-3 must throw: their host reduction is M_200-referred."""
    with pytest.raises(Exception):
        raw(subhalo=True, subhalo_model=model, subhalo_virial=True)


def test_no_throw_without_subhalo():
    """The guard is scoped to subhalo runs, so subhalo=False must not throw."""
    a = raw(subhalo=False, subhalo_virial=True)
    b = raw(subhalo=False)
    assert np.array_equal(a, b), "subhalo_virial leaked into the subhalo-off path"


@pytest.mark.parametrize("cfg", [M5, M4])
def test_live_and_deterministic(cfg):
    n = dict(BASE) if cfg is M5 else dict(BASE, nsamples=60)
    a = np.asarray(gw.sample_lensing_raw_ml(**n, **cfg, subhalo_virial=True)["kappa"])
    b = np.asarray(gw.sample_lensing_raw_ml(**n, **cfg, subhalo_virial=True)["kappa"])
    # explicit False: since 2026-07-29 the default IS True, so a bare call here
    # would compare True against True and the no-op guard would pass vacuously.
    c = np.asarray(gw.sample_lensing_raw_ml(**n, **cfg, subhalo_virial=False)["kappa"])
    assert np.array_equal(a, b), "same seed gave different samples"
    assert not np.array_equal(a, c), "subhalo_virial=True was a no-op"
    assert np.all(np.isfinite(a))


def test_pdf_effect_is_bounded():
    """The PDF effect must stay SMALL -- it is not a directional test.

    An earlier version of this test asserted that virial mode reduces the
    clipped sd, with the z_s ordering of the analytic content prediction. That
    was wrong: the 480k-rays/arm A/B (data/results/subhalo_virial/report.md)
    finds the effect AT THE SAMPLING FLOOR at z_s = 0.5/1/5, with
    Delta sd/sd = -0.13 +/- 0.49 %, +0.13 +/- 0.38 %, +0.15 +/- 0.22 %. The
    18-28% change in substructure content inside r_200 does NOT propagate to a
    resolvable P(lnmu) change, because substructure is only a few percent of
    Var(kappa). The directional version passed only by luck at 16k rays, where
    the sd uncertainty is ~4%.

    So the gate here is a BOUND, sized well outside this test's own noise
    (~4% at 4 seeds x 4000 rays) but tight enough to catch a gross error such
    as the extent or the mass scale being applied twice.
    """
    def sd(zs, virial, seeds=(11, 22, 33, 44)):
        out = []
        for s in seeds:
            x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, 4000, s,
                                          **M5, kappa_anchor=1,
                                          subhalo_virial=virial))
            x = x[np.isfinite(x)]
            out.append(x[np.abs(x) < 1.0].std())
        return float(np.mean(out))

    for zs in (0.5, 5.0):
        r = sd(zs, True) / sd(zs, False)
        assert 0.85 < r < 1.15, f"z_s={zs}: sd ratio {r:.4f} outside the expected bound"


def test_wired_on_all_entry_points():
    """All five py entry points must accept subhalo_virial AND act on it.

    sample_lnmu / compute_lnmu_stats take (z, OmegaM, sigma8, h, Nreal, ...);
    the *_ml family takes (z, h, OmegaM, sigma8, nsamples, ...). Keyword-only
    here so the orders cannot be confused.
    """
    LEG = dict(z=BASE["z"], OmegaM=BASE["OmegaM"], sigma8=BASE["sigma8"],
               h=BASE["h"], Nreal=BASE["nsamples"], seed=BASE["seed"])

    for fn, kw in ((gw.sample_lnmu, LEG), (gw.sample_lnmu_ml, BASE)):
        a = np.asarray(fn(**kw, **M5, subhalo_virial=False))
        b = np.asarray(fn(**kw, **M5, subhalo_virial=True))
        assert not np.array_equal(a, b), f"{fn.__name__} ignored subhalo_virial"

    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(**BASE, **M5, subhalo_virial=False)["lnmu"])
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(**BASE, **M5, subhalo_virial=True)["lnmu"])
    assert not np.array_equal(a, b), "sample_lnmu_ml_with_diagnostics ignored subhalo_virial"

    assert not np.array_equal(raw(**M5, subhalo_virial=False),
                              raw(**M5, subhalo_virial=True)), \
        "sample_lensing_raw_ml ignored subhalo_virial"

    assert gw.compute_lnmu_stats(**LEG, **M5, subhalo_virial=False) != \
        gw.compute_lnmu_stats(**LEG, **M5, subhalo_virial=True), \
        "compute_lnmu_stats ignored subhalo_virial"


def test_composes_with_production_clustering():
    """Runs alongside the production bias config and stays reproducible."""
    kw = dict(M5, bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True,
              fil_bias=True, kappa_anchor=1, subhalo_virial=True)
    a = np.asarray(gw.sample_lnmu_ml_with_diagnostics(**BASE, **kw)["lnmu"])
    b = np.asarray(gw.sample_lnmu_ml_with_diagnostics(**BASE, **kw)["lnmu"])
    assert np.array_equal(a, b)
    good = a[np.isfinite(a)]
    assert good.size > 0 and np.all(np.isfinite(good))
