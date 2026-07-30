import os
import sys
import subprocess
import shutil
import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ml.params import PRODUCTION_CONFIG, PRODUCTION_CONFIG_HASH


def test_parallel_dataset_generation():
    test_dir = "datasets/test_parallel"
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
            # schema 2.1 (1+6d, sigma8-mode). 2.0 was the A_s-mode variant,
            # retired 2026-07-27 in favour of Vaskonen's convention.
            for k in ("sigma8", "OmegaB", "ns", "zeq", "As_derived"):
                assert f"samples/{k}" in f
            assert "metadata" in f
            assert f["metadata"].attrs.get("dataset_schema_version") == "2.1"
            assert f["metadata"].attrs.get("amplitude_mode") == "sigma8"

            # The physics config must be pinned from ml.params, not inherited
            # from the compiled-in C++ defaults (several of which are still the
            # legacy staged values).
            assert "metadata/physics_config" in f
            phys = f["metadata/physics_config"].attrs
            assert phys["config_hash"] == PRODUCTION_CONFIG_HASH
            for k, v in PRODUCTION_CONFIG.items():
                assert phys[k] == v, f"{k}: {phys[k]} != {v}"
            # ⚠ INVERTED 2026-07-29. This used to assert that at least one
            # production setting DIFFERED from the shipped default, guarding
            # against the pin silently becoming a no-op while the defaults were
            # staged legacy values. Since the paper-default flip the defaults ARE
            # the paper config, so the invariant is now agreement: the recorded
            # defaults must match the pin exactly. That catches the two failures
            # that actually matter now -- a C++ default drifting away from the
            # paper config, and get_simulator_config reporting stale literals
            # instead of the compiled-in struct (which it did, until 2026-07-29).
            defaults = f["metadata/simulator_defaults"].attrs
            for k, v in PRODUCTION_CONFIG.items():
                assert k in defaults, f"{k} missing from simulator_defaults"
                assert defaults[k] == v, \
                    f"compiled-in default {k}={defaults[k]} != paper config {v}"


    # Cleanup
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
