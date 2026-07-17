import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

from ml.cache_utils import CacheManager, get_git_commit

def test_cache_manager_save_load_and_fail_loudly():
    cache_dir = "data/cache_test"
    cm = CacheManager(cache_dir=cache_dir)
    
    # 1. Setup sample keys and data
    key_dict = {
        "z": 1.5,
        "h": 0.67,
        "OmegaM": 0.30,
        "sigma8": 0.80,
        "nsamples": 1000,
        "seed": 42,
        "grid_definition": [20, 20]
    }
    
    data_dict = {
        "sim_grid": np.linspace(-100, 0, 400),
        "mle_idx": np.array([5, 12])
    }
    
    # Clean up test cache directory
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        
    try:
        # 2. Test saving
        cm.save("sim_grid", key_dict, data_dict)
        
        # Check that file exists
        path = cm.get_cache_path("sim_grid", key_dict)
        assert os.path.exists(path)
        
        # 3. Test loading
        loaded = cm.load("sim_grid", key_dict)
        assert loaded is not None
        np.testing.assert_allclose(loaded["sim_grid"], data_dict["sim_grid"])
        np.testing.assert_allclose(loaded["mle_idx"], data_dict["mle_idx"])
        
        # 4. Test loading with a mismatch (e.g. different seed)
        mismatched_keys = key_dict.copy()
        mismatched_keys["seed"] = 100
        
        # If we load using a different key dictionary that hashes differently,
        # it will look for a different filename and return None (since it doesn't exist).
        loaded_missing = cm.load("sim_grid", mismatched_keys)
        assert loaded_missing is None
        
        # 5. Test loader verification: if we mock a file but it has mismatched keys inside, it must fail loudly
        # To test this, let's manually write a file that has metadata mismatch
        fake_path = cm.get_cache_path("sim_grid", key_dict)
        # Save a different metadata inside the npz but on the same filename path
        bad_keys = key_dict.copy()
        bad_keys["seed"] = 1234
        
        # Override the file manually
        import json
        np.savez(fake_path, metadata=json.dumps(bad_keys), **data_dict)
        
        # Now loading with key_dict should fail loudly because metadata inside npz has seed=1234
        # but we requested seed=42!
        with pytest.raises(ValueError) as excinfo:
            cm.load("sim_grid", key_dict)
            
        assert "Cache metadata mismatch for key 'seed'" in str(excinfo.value)
        
    finally:
        # Clean up test cache directory
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

def test_git_commit_retrieval():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0
