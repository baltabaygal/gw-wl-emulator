import os
import sys
import subprocess
import h5py
import pytest

def get_test_dataset_dir():
    # Helper to ensure at least a tiny dataset exists for testing
    d = "datasets_test_suite"
    if not os.path.exists(d):
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "15",
            "--nsamples", "10",
            "--seed", "123",
            "--output_dir", d
        ], check=True)
    return d

def test_metadata_completeness():
    dataset_dir = get_test_dataset_dir()
    splits = ["train", "validation", "test"]
    
    for split in splits:
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        if not os.path.exists(path):
            continue
            
        with h5py.File(path, 'r') as f:
            # Required groups/paths
            assert "metadata" in f
            assert "metadata/preprocessing" in f
            assert "metadata/invalid_stats" in f
            assert "metadata/parameter_ranges" in f
            
            # Root metadata attributes
            meta = f["metadata"]
            required_attrs = [
                "split", "seed", "dataset_version", "nsamples_per_point",
                "git_commit", "git_branch", "dataset_schema_version",
                "generation_timestamp"
            ]
            for attr in required_attrs:
                assert attr in meta.attrs, f"Missing root attribute {attr} in {split}"
                
            # Check parameter ranges
            pr = f["metadata/parameter_ranges"]
            for p in ["z", "h", "OmegaM", "sigma8"]:
                assert p in pr.attrs, f"Missing parameter range for {p} in {split}"
                rng = pr.attrs[p]
                assert len(rng) == 2
                assert rng[0] < rng[1]
