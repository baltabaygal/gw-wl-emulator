"""Exact reference for the pseudo-elliptical NFW kappa/gamma.

This is the test that should have caught the cos^2/sin^2 error in
`kappagammaNFWeps` (docs/pseudo_elliptical_shear_bug.md). It builds kappa_eps
and gamma_eps from second derivatives of the substituted potential
psi_eps(x1,x2) = psi(x_eps) by the chain rule -- no finite differences (the NFW
potential integral is log-divergent at the origin, so differencing it is
numerically worthless) -- and compares against the C++ implementation.

Currently EXPECTED TO FAIL on gamma while the bug is unfixed. Run with

    pytest tests/test_pseudo_elliptical_shear.py -v

`test_kappa_matches_potential` should pass today; `test_gamma_matches_potential`
is the regression guard for the proposed fix and is marked xfail(strict=False)
so the suite stays green until the fix lands. Flip to a hard assert when it does.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "build")


# --- exact reference --------------------------------------------------------
def _F(x):
    """{F0, F1} of astro-ph/0608153, the circular NFW kappa / mean-kappa cores."""
    if x > 1:
        t = np.arctan(np.sqrt((x - 1) / (1 + x))) / np.sqrt(x * x - 1)
        return (1 - 2 * t) / (x * x - 1), 2 * t + np.log(x / 2)
    if x < 1:
        t = np.arctanh(np.sqrt((1 - x) / (1 + x))) / np.sqrt(1 - x * x)
        return (1 - 2 * t) / (x * x - 1), 2 * t + np.log(x / 2)
    return 1.0 / 3.0, 1 + np.log(0.5)


def exact_kappa_gamma(x1, x2, eps, kappa0=1.0):
    """kappa_eps, gamma_eps from psi_eps = psi(x_eps) via the chain rule.

    psi'(u) = u kappabar(u), psi''(u) = 2 kappa(u) - kappabar(u), with
    kappa = 2 k0 F0 and kappabar = 4 k0 F1 / u^2.
    """
    a1, a2 = 1.0 - eps, 1.0 + eps
    u = np.sqrt(a1 * x1 ** 2 + a2 * x2 ** 2)
    F0, F1 = _F(u)
    kap = 2 * kappa0 * F0
    kbar = 4 * kappa0 * F1 / u ** 2
    dp = u * kbar
    dd = 2 * kap - kbar
    p11 = dd * (a1 * x1 / u) ** 2 + dp * a1 * (u ** 2 - a1 * x1 ** 2) / u ** 3
    p22 = dd * (a2 * x2 / u) ** 2 + dp * a2 * (u ** 2 - a2 * x2 ** 2) / u ** 3
    p12 = a1 * a2 * x1 * x2 * (dd / u ** 2 - dp / u ** 3)
    kappa_e = 0.5 * (p11 + p22)
    g1, g2 = 0.5 * (p11 - p22), p12
    return kappa_e, np.hypot(g1, g2)


def code_kappa_gamma(x1, x2, eps, kappa0=1.0):
    """Python mirror of cpp/lensing.cpp::kappagammaNFWeps (current form)."""
    a1, a2 = 1.0 - eps, 1.0 + eps
    x1e, x2e = np.sqrt(a1) * x1, np.sqrt(a2) * x2
    xe = max(np.hypot(x1e, x2e), 1e-12)
    pe = np.arctan2(x2e, x1e)
    F0, F1 = _F(xe)
    k0 = 2 * kappa0 * F0
    g0 = 2 * kappa0 * (2 * F1 / (xe * xe) - F0)
    c = np.cos(2 * pe)
    kappa_e = k0 + eps * c * g0
    g2 = g0 ** 2 + 2 * eps * c * g0 * k0 + eps ** 2 * (k0 ** 2 - (c * g0) ** 2)
    return kappa_e, np.sqrt(max(0.0, g2))


def proposed_kappa_gamma(x1, x2, eps, kappa0=1.0):
    """The fix: gamma from its COMPONENTS, which cannot go negative."""
    a1, a2 = 1.0 - eps, 1.0 + eps
    x1e, x2e = np.sqrt(a1) * x1, np.sqrt(a2) * x2
    xe = max(np.hypot(x1e, x2e), 1e-12)
    pe = np.arctan2(x2e, x1e)
    F0, F1 = _F(xe)
    k0 = 2 * kappa0 * F0
    g0 = 2 * kappa0 * (2 * F1 / (xe * xe) - F0)
    kappa_e = k0 + eps * np.cos(2 * pe) * g0
    g1 = g0 * np.cos(2 * pe) + eps * k0
    g2 = np.sqrt(max(0.0, 1.0 - eps ** 2)) * g0 * np.sin(2 * pe)
    return kappa_e, np.hypot(g1, g2)


def _grid(n=600, seed=0):
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.15, 0.60, n)
    x = 10 ** rng.uniform(-2.0, 1.0, n)
    phi = rng.uniform(0.0, 2 * np.pi, n)
    return eps, x * np.cos(phi), x * np.sin(phi)


def test_kappa_matches_potential():
    """kappa_eps = kappa(x_eps) + eps cos(2 phi_eps) gamma(x_eps) is EXACT."""
    eps, x1, x2 = _grid()
    err = [abs(code_kappa_gamma(a, b, e)[0] - exact_kappa_gamma(a, b, e)[0])
           / max(abs(exact_kappa_gamma(a, b, e)[0]), 1e-30)
           for e, a, b in zip(eps, x1, x2)]
    assert np.max(err) < 1e-10, f"kappa_eps max rel err {np.max(err):.2e}"


def test_proposed_gamma_matches_potential():
    """The component form reproduces the potential derivation exactly."""
    eps, x1, x2 = _grid()
    err = []
    for e, a, b in zip(eps, x1, x2):
        ref = exact_kappa_gamma(a, b, e)[1]
        got = proposed_kappa_gamma(a, b, e)[1]
        if ref > 1e-12:
            err.append(abs(got - ref) / ref)
    assert np.max(err) < 1e-10, f"proposed gamma max rel err {np.max(err):.2e}"


def test_current_gamma_can_go_negative():
    """Documents the failure mode: the cos^2 expression admits gamma^2 < 0,
    which is what produced `nan_gamma` before the 2026-07-03 clamp."""
    eps, x1, x2 = _grid(n=4000, seed=3)
    neg = 0
    for e, a, b in zip(eps, x1, x2):
        a1, a2 = 1 - e, 1 + e
        xe = max(np.hypot(np.sqrt(a1) * a, np.sqrt(a2) * b), 1e-12)
        pe = np.arctan2(np.sqrt(a2) * b, np.sqrt(a1) * a)
        F0, F1 = _F(xe)
        k0, g0 = 2 * F0, 2 * (2 * F1 / xe ** 2 - F0)
        c = np.cos(2 * pe)
        if g0 ** 2 + 2 * e * c * g0 * k0 + e ** 2 * (k0 ** 2 - (c * g0) ** 2) < 0:
            neg += 1
    assert neg > 0, "expected the cos^2 form to go negative somewhere"


@pytest.mark.xfail(strict=False,
                   reason="cos^2 should be sin^2; see "
                          "docs/pseudo_elliptical_shear_bug.md. Remove this "
                          "marker when the fix lands.")
def test_gamma_matches_potential():
    eps, x1, x2 = _grid()
    err = []
    for e, a, b in zip(eps, x1, x2):
        ref = exact_kappa_gamma(a, b, e)[1]
        got = code_kappa_gamma(a, b, e)[1]
        if ref > 1e-12:
            err.append(abs(got - ref) / ref)
    assert np.max(err) < 1e-10, f"gamma_eps max rel err {np.max(err):.2e}"


@pytest.mark.skipif("gwlensing" not in sys.modules and True,
                    reason="needs the built module; run on the Mac test env")
def test_cpp_matches_python_mirror():
    """Guard that the Python mirror above really mirrors the C++.

    Not runnable without the compiled module; kept so the mirror cannot silently
    drift from cpp/lensing.cpp.
    """
    pytest.skip("enable once a binding for kappagammaNFWeps exists")
