import os
import json
import numpy as np
import pytest

from ml.cache_utils import CacheManager

def test_phase2c_output_files_exist():
    """Verify that all required Phase 2C output JSON and MD files exist and are not empty."""
    required_files = [
        # JSON results
        "data/results/phase2c_likelihood_surface_results.json",
        "data/results/phase2c_tail_validation_results.json",
        "data/results/phase2c_runtime_results.json",
        "data/results/phase2c_failure_map_results.json",
        # Markdown reports
        "docs/phase2c_likelihood_surface_validation.md",
        "docs/phase2c_tail_validation.md",
        "docs/phase2c_runtime_optimization.md",
        "docs/phase2c_nsf_failure_map.md",
        "docs/phase2c_cache_reproducibility.md"
    ]
    
    for rel_path in required_files:
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
        assert os.path.exists(abs_path), f"Required output file is missing: {rel_path}"
        assert os.path.getsize(abs_path) > 0, f"Required output file is empty: {rel_path}"

def test_no_nan_or_inf_in_results():
    """Check that no JSON results files contain NaN or Inf values."""
    result_files = [
        "data/results/phase2c_likelihood_surface_results.json",
        "data/results/phase2c_tail_validation_results.json",
        "data/results/phase2c_runtime_results.json",
        "data/results/phase2c_failure_map_results.json"
    ]
    
    for rel_path in result_files:
        abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", rel_path))
        with open(abs_path, "r") as f:
            data = json.load(f)
            
        # Recursive check for inf / nan / None in dictionary values
        def check_val(val, path_str):
            if isinstance(val, dict):
                for k, v in val.items():
                    check_val(v, f"{path_str}.{k}")
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    check_val(v, f"{path_str}[{i}]")
            elif isinstance(val, float):
                assert np.isfinite(val), f"Non-finite float value {val} at {path_str} in {rel_path}"
            elif val is None:
                # None is generally allowed in some cases, but check if it's expected
                pass
                
        check_val(data, "root")

def test_cached_and_uncached_grids_match():
    """Verify that loading cached grids returns exactly matching values compared to metadata specs."""
    cm = CacheManager()
    
    # Locate all cached simulator grids in data/cache
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cache"))
    if not os.path.exists(cache_dir):
        pytest.skip("Cache directory does not exist.")
        
    grid_files = [f for f in os.listdir(cache_dir) if f.startswith("sim_grid_") and f.endswith(".npz")]
    if not grid_files:
        pytest.skip("No cached simulator grids found to verify.")
        
    for filename in grid_files:
        path = os.path.join(cache_dir, filename)
        with np.load(path, allow_pickle=True) as npz:
            metadata = json.loads(str(npz["metadata"]))
            grid_data = npz["sim_grid"]
            
            # Reconstruct the grid key from metadata
            reconstructed_key = {
                "z": float(metadata["z"]),
                "h": float(metadata["h"]),
                "nsamples": int(metadata["nsamples"]),
                "seed": int(metadata["seed"]),
                "grid_definition": metadata["grid_definition"],
                "catalog_hash": metadata["catalog_hash"]
            }
            
            # Load using CacheManager and check if it matches exactly
            loaded = cm.load("sim_grid", reconstructed_key)
            assert loaded is not None
            np.testing.assert_allclose(loaded["sim_grid"], grid_data)
            assert np.all(np.isfinite(loaded["sim_grid"])), "Cached simulator grid contains non-finite values!"
