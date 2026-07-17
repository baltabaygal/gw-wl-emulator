import numpy as np
from ml.run_interpolation_ceiling import power_law
from scipy.optimize import curve_fit

def test_power_law_curve_fit():
    # Generate mock scaling data with known parameters: A=0.04, alpha=0.6, B=0.01
    sizes = np.array([50, 100, 200, 326, 500, 750, 1000])
    A_true, alpha_true, B_true = 0.04, 0.6, 0.01
    
    # Add a tiny amount of noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1e-4, size=sizes.shape)
    
    metrics = power_law(sizes, A_true, alpha_true, B_true) + noise
    
    # Fit
    popt, pcov = curve_fit(power_law, sizes, metrics, p0=[0.05, 0.5, 0.01], bounds=(0, np.inf))
    A_fit, alpha_fit, B_fit = popt
    
    # Check that the fitted values are very close to true values
    assert np.isclose(A_fit, A_true, rtol=1e-1), f"A mismatch: {A_fit} vs {A_true}"
    assert np.isclose(alpha_fit, alpha_true, rtol=1e-1), f"alpha mismatch: {alpha_fit} vs {alpha_true}"
    assert np.isclose(B_fit, B_true, rtol=1e-1), f"B mismatch: {B_fit} vs {B_true}"
