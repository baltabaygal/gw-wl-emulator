import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest
import torch

from ml.nsf_model import ConditionalNSF
from ml.nsf_likelihood import catalog_log_likelihood, log_prob_batch

def test_nsf_likelihood_batching_equivalence():
    model = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
    model.set_preprocessing_stats([0.5]*4, [0.2]*4, 0.05, 0.25)
    model.eval()
    
    N = 100
    M = 5
    
    catalog = np.random.randn(N, 1).astype(np.float32)
    theta_batch = np.random.uniform(0.2, 0.4, (M, 4)).astype(np.float32)
    
    # 1. Compute unbatched loop
    unbatched_liks = []
    for idx in range(M):
        log_probs = model.log_prob(catalog, theta_batch[idx])
        unbatched_liks.append(np.sum(log_probs))
    unbatched_liks = np.array(unbatched_liks)
    
    # 2. Compute batched
    batched_liks = catalog_log_likelihood(model, catalog, theta_batch)
    
    # Assert numerical equivalence
    np.testing.assert_allclose(unbatched_liks, batched_liks, rtol=1e-5, atol=1e-5)
    
    # 3. Test log_prob_batch equivalence
    x_b = np.random.randn(10, 1).astype(np.float32)
    c_b = np.random.uniform(0.2, 0.4, (10, 4)).astype(np.float32)
    
    lp_batch = log_prob_batch(model, x_b, c_b)
    lp_individual = np.array([model.log_prob(x_b[i:i+1], c_b[i:i+1])[0] for i in range(10)])
    
    np.testing.assert_allclose(lp_batch, lp_individual, rtol=1e-5, atol=1e-5)

def test_nan_inf_safety():
    model = ConditionalNSF(input_dim=1, context_dim=4, num_transforms=3, hidden_features=32, bins=8)
    model.set_preprocessing_stats([0.5]*4, [0.2]*4, 0.05, 0.25)
    model.eval()
    
    # Pass bad/infinite inputs
    catalog_with_nan = np.array([[1.0], [np.nan], [2.0]], dtype=np.float32)
    theta = np.array([[0.67, 0.30, 0.85, 0.90]], dtype=np.float32)
    
    # The code evaluates and returns NaN/Inf log_probs, but does not crash.
    # We want to verify that when NaN exists, the sum is not finite (or fails if handled)
    lik = catalog_log_likelihood(model, catalog_with_nan, theta)
    assert not np.isfinite(lik[0])
