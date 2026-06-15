import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

from ml.evaluate_nsf_tail_robustness import compute_binned_quantiles, get_config_partition

def test_compute_binned_quantiles():
    # Symmetric binned distribution
    probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 50% quantile (median) should be exactly 0.5 (midpoint of bin [0.4, 0.6])
    q50 = compute_binned_quantiles(probs, bin_edges, [0.5])[0]
    assert pytest.approx(q50) == 0.5
    
    # 10% quantile should be exactly 0.2 (boundary between first and second bin)
    q10 = compute_binned_quantiles(probs, bin_edges, [0.1])[0]
    assert pytest.approx(q10) == 0.2
    
    # 90% quantile should be exactly 0.8 (boundary between fourth and fifth bin)
    q90 = compute_binned_quantiles(probs, bin_edges, [0.9])[0]
    assert pytest.approx(q90) == 0.8
    
    # Test batch evaluation
    qs = compute_binned_quantiles(probs, bin_edges, [0.1, 0.5, 0.9])
    assert len(qs) == 3
    assert pytest.approx(qs[0]) == 0.2
    assert pytest.approx(qs[1]) == 0.5
    assert pytest.approx(qs[2]) == 0.8

def test_get_config_partition():
    # Test low_z central
    z, p = get_config_partition(z_val=0.5, om_val=0.30, s8_val=0.80)
    assert z == "low_z"
    assert p == "central_ID"
    
    # Test mid_z edge
    z, p = get_config_partition(z_val=1.5, om_val=0.22, s8_val=0.70)
    assert z == "mid_z"
    assert p == "edge_ID"
    
    # Test high_z OoD
    z, p = get_config_partition(z_val=2.5, om_val=0.45, s8_val=1.20)
    assert z == "high_z"
    assert p == "OoD"
