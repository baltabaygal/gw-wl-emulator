import os
import sys
import subprocess
import shutil
import h5py
import numpy as np

def test_parallel_dataset_generation():
    test_dir = "datasets_test_parallel"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    cmd = [
        sys.executable,
        "python/generate_dataset.py",
        "--num_points", "3",
        "--nsamples", "100",
        "--seed", "42",
        "--output_dir", test_dir
    ]
    
    # Run the generation script
    # Prepend environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = "build:."
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Dataset generation failed: {res.stderr}\nStdout: {res.stdout}"
    
    # Verify the HDF5 split outputs exist and are valid
    splits = ["train", "validation", "test"]
    for s in splits:
        path = os.path.join(test_dir, s, f"dataset_{s}.h5")
        if not os.path.exists(path):
            continue
            
        with h5py.File(path, "r") as f:
            assert "samples/lnmu" in f
            assert "samples/valid_counts" in f
            assert "samples/z" in f
            assert "samples/h" in f
            assert "samples/OmegaM" in f
            assert "samples/sigma8" in f
            assert "metadata" in f
            assert f["metadata"].attrs.get("dataset_schema_version") == "1.1"
            
    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
