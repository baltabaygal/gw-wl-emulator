import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest
import torch

from ml.nsf_model import ConditionalNSF

def test_nsf_likelihood_evaluation_and_prior_bounds():
    # 1. Initialize a model with fake statistics (we don't need a trained checkpoint for bounds checks)
    model = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
    model.set_preprocessing_stats([0.0]*4, [1.0]*4, 0.0, 1.0)
    model.eval()
    
    # 2. Define MCMC log probability function wrapping NSF
    def mcmc_lnprob_nsf(theta, z_val, lnmu_mock):
        h, om, s8 = theta
        # Prior bounds
        if not (0.59 <= h <= 0.76 and 0.20 <= om <= 0.40 and 0.65 <= s8 <= 1.05):
            return -np.inf
            
        # Inside bounds: evaluate continuous NSF likelihood
        context = np.array([[z_val, h, om, s8]], dtype=np.float32)
        log_probs = model.log_prob(lnmu_mock, context)
        return float(np.sum(log_probs))
        
    z_mock = 1.0
    lnmu_mock = np.random.randn(50, 1)
    
    # 3. Verify prior bounds consistency
    theta_oob_h_low = [0.50, 0.30, 0.85]
    theta_oob_h_high = [0.80, 0.30, 0.85]
    theta_oob_om_low = [0.67, 0.15, 0.85]
    theta_oob_om_high = [0.67, 0.45, 0.85]
    theta_oob_s8_low = [0.67, 0.30, 0.60]
    theta_oob_s8_high = [0.67, 0.30, 1.10]
    
    for theta in [theta_oob_h_low, theta_oob_h_high, theta_oob_om_low, theta_oob_om_high, theta_oob_s8_low, theta_oob_s8_high]:
        val = mcmc_lnprob_nsf(theta, z_mock, lnmu_mock)
        assert val == -np.inf, f"NSF likelihood did not return -inf for OOB parameter {theta}"
        
    # 4. Verify inside-bound evaluation
    theta_in_bound = [0.67, 0.30, 0.85]
    val_in = mcmc_lnprob_nsf(theta_in_bound, z_mock, lnmu_mock)
    assert np.isfinite(val_in)
    assert val_in != -np.inf
