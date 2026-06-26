import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import quad


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subhalo_demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("subhalo_demo_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shmf_normalization_recovers_subhalo_mass_fraction():
    demo = load_demo()

    mass_fraction, _ = quad(
        lambda lp: np.exp(lp) * demo.dN_dlnpsi(np.exp(lp), demo.g),
        np.log(demo.PSI_RES),
        np.log(1.0),
    )

    assert np.isclose(mass_fraction, demo.fs, rtol=1e-10, atol=1e-12)


def test_extending_upper_bound_restores_mass_not_number_count():
    demo = load_demo()

    count_to_01, _ = quad(
        lambda lp: demo.dN_dlnpsi(np.exp(lp), demo.g),
        np.log(demo.PSI_RES),
        np.log(0.1),
    )
    mass_to_01, _ = quad(
        lambda lp: np.exp(lp) * demo.dN_dlnpsi(np.exp(lp), demo.g),
        np.log(demo.PSI_RES),
        np.log(0.1),
    )

    assert demo.Nmean - count_to_01 < 1.0
    assert mass_to_01 < 0.8 * demo.fs


def test_poisson_draw_has_expected_mean_and_variance():
    demo = load_demo()
    rng = np.random.default_rng(12345)

    draws = rng.poisson(demo.Nmean, size=50_000)

    assert np.isclose(draws.mean(), demo.Nmean, rtol=0.01)
    assert np.isclose(draws.var(), demo.Nmean, rtol=0.03)


def test_inverse_cdf_mass_sampler_matches_analytic_bin_probabilities():
    demo = load_demo()
    rng = np.random.default_rng(12345)

    n = 200_000
    psi = np.exp(np.interp(rng.uniform(0, 1, n), demo.cdf, demo.lp))
    edges = np.logspace(-4, 0, 17)
    observed, _ = np.histogram(psi, bins=edges)
    observed = observed / observed.sum()

    expected = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        val, _ = quad(
            lambda lp: demo.dN_dlnpsi(np.exp(lp), demo.g),
            np.log(lo),
            np.log(hi),
        )
        expected.append(val / demo.Nmean)
    expected = np.array(expected)

    assert np.max(np.abs(observed - expected)) < 0.01


def test_host_mass_trend_increases_substructure_at_fixed_redshift():
    demo = load_demo()

    rows = demo.host_comparison
    masses = np.array([row[0] for row in rows])
    fs_values = np.array([row[3] for row in rows])
    nmeans = np.array([row[5] for row in rows])

    assert np.all(np.diff(masses) > 0)
    assert np.all(np.diff(fs_values) > 0)
    assert np.all(np.diff(nmeans) > 0)

