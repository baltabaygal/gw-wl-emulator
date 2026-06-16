import numpy as np
from ml.posterior_scaling_metrics import fit_power_law, power_law

def test_phase3d_metrics_trend_fitting():
    # 1. Generate synthetic convergent data: JSD(N) = 1.0 * N^-0.5 + 0.01
    N_vals = np.array([250, 1000, 5000], dtype=float)
    A_true, alpha_true, B_true = 1.0, 0.5, 0.01
    y_vals = power_law(N_vals, A_true, alpha_true, B_true)
    
    # Fit
    popt = fit_power_law(N_vals, y_vals)
    
    # Assert popt has shape (3,) and fits reasonably
    assert len(popt) == 3
    assert popt[2] >= 0.0
    
    # Check predictions
    y_fit = power_law(N_vals, *popt)
    np.testing.assert_allclose(y_vals, y_fit, rtol=1e-2)

def test_phase3d_metrics_fit_handles_pathological_data():
    # Saturated/Flat data (no clear power-law decay)
    N_vals = np.array([250, 1000, 5000], dtype=float)
    y_vals = np.array([0.65, 0.65, 0.65])
    
    popt = fit_power_law(N_vals, y_vals)
    assert len(popt) == 3
    assert np.isfinite(popt).all()
