import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

def wasserstein_1d(u_samples, v_samples):
    u_sorted = np.sort(u_samples)
    v_sorted = np.sort(v_samples)
    if len(u_sorted) == 0 or len(v_sorted) == 0:
        return np.nan
    n = min(len(u_sorted), len(v_sorted))
    u_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)
    return np.mean(np.abs(u_interp - v_interp))

def test_deterministic_reproducibility():
    z, h, OmegaM, sigma8, nsamples = 1.0, 0.674, 0.315, 0.811, 1000
    seed = 42

    samples1 = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)
    samples2 = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)

    np.testing.assert_array_equal(samples1, samples2)

def test_redshift_monotonicity():
    h, OmegaM, sigma8, nsamples, seed = 0.674, 0.315, 0.811, 5000, 42

    zs = [0.5, 1.0, 2.0]
    variances = []

    for z in zs:
        samples = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)
        samples = samples[~np.isnan(samples)]
        variances.append(np.var(samples))

    assert variances[-1] > 1.2 * variances[0], (
        f"Broad redshift trend failed: variances={variances}"
    )

def test_sigma8_monotonicity():
    z, h, OmegaM, nsamples, seed = 1.0, 0.674, 0.315, 5000, 42

    sigma8s = [0.6, 0.8, 1.0]
    variances = []

    for s8 in sigma8s:
        samples = gw.sample_lnmu_ml(z, h, OmegaM, s8, nsamples, seed)
        samples = samples[~np.isnan(samples)]
        variances.append(np.var(samples))

    assert variances[-1] > variances[0], (
        f"Broad sigma8 trend failed: variances={variances}"
    )

def test_pdf_normalization_approximate():
    z, h, OmegaM, sigma8, nsamples, seed = 1.0, 0.674, 0.315, 0.811, 10000, 42
    samples = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)
    samples = samples[~np.isnan(samples)]

    hist, bin_edges = np.histogram(samples, bins=100, density=True)
    bin_widths = np.diff(bin_edges)
    integral = np.sum(hist * bin_widths)

    assert np.isclose(integral, 1.0, atol=1e-2), f"PDF integral {integral} is not close to 1.0"

def test_nan_fraction():
    z, h, OmegaM, sigma8, nsamples, seed = 1.0, 0.674, 0.315, 0.811, 5000, 42
    samples = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)

    nan_frac = np.sum(np.isnan(samples)) / len(samples)
    assert nan_frac < 0.1, f"NaN fraction {nan_frac} is too high"

def test_tail_stability():
    z, h, OmegaM, sigma8, nsamples = 1.0, 0.674, 0.315, 0.811, 10000

    q99s = []
    for i in range(5):
        samples = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, 100+i)
        samples = samples[~np.isnan(samples)]
        q99s.append(np.percentile(samples, 99))

    std_q99 = np.std(q99s)
    assert std_q99 < 0.1, f"99th percentile too unstable across seeds: {q99s}"

def test_wasserstein_continuity():
    z, h, OmegaM, nsamples, seed = 1.0, 0.674, 0.315, 5000, 42

    s8_1 = 0.800
    s8_2 = 0.801
    s8_3 = 0.900

    samp1 = gw.sample_lnmu_ml(z, h, OmegaM, s8_1, nsamples, seed)
    samp2 = gw.sample_lnmu_ml(z, h, OmegaM, s8_2, nsamples, seed)
    samp3 = gw.sample_lnmu_ml(z, h, OmegaM, s8_3, nsamples, seed)

    dist12 = wasserstein_1d(samp1[~np.isnan(samp1)], samp2[~np.isnan(samp2)])
    dist13 = wasserstein_1d(samp1[~np.isnan(samp1)], samp3[~np.isnan(samp3)])

    assert dist12 < dist13, f"Continuity violation: d(0.8, 0.801)={dist12} >= d(0.8, 0.9)={dist13}"

if __name__ == "__main__":
    pytest.main([__file__])
