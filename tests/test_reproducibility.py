import os
import shutil
import subprocess
import sys
import h5py
import numpy as np
import pytest

def test_bitwise_reproducibility():
    dir1 = "tmp_repro_1"
    dir2 = "tmp_repro_2"
    if os.path.exists(dir1): shutil.rmtree(dir1)
    if os.path.exists(dir2): shutil.rmtree(dir2)

    try:
        # Run generation 1
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "5",
            "--nsamples", "100",
            "--seed", "42",
            "--output_dir", dir1
        ], check=True)

        # Run generation 2
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "5",
            "--nsamples", "100",
            "--seed", "42",
            "--output_dir", dir2
        ], check=True)

        # Compare files
        for split in ["train", "validation", "test"]:
            f1_path = os.path.join(dir1, split, f"dataset_{split}.h5")
            f2_path = os.path.join(dir2, split, f"dataset_{split}.h5")
            
            exists1 = os.path.exists(f1_path)
            exists2 = os.path.exists(f2_path)
            assert exists1 == exists2, f"Split {split} presence mismatch"
            if not exists1:
                continue

            with h5py.File(f1_path, 'r') as f1, h5py.File(f2_path, 'r') as f2:
                # Check samples (schema 2.0: 1+6d As-mode)
                for key in ["lnmu", "valid_counts", "z", "h", "OmegaM", "As",
                            "OmegaB", "ns", "zeq", "sigma8_derived", "split_type"]:
                    p1 = f"samples/{key}"
                    p2 = f"samples/{key}"
                    assert p1 in f1 and p2 in f2, f"Missing key {key} in split {split}"
                    
                    d1 = f1[p1][:]
                    d2 = f2[p2][:]
                    
                    if d1.dtype.kind == 'S' or d1.dtype.kind == 'O':
                        # String comparison
                        d1_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in d1]
                        d2_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in d2]
                        assert d1_str == d2_str, f"String content mismatch for {key} in split {split}"
                    else:
                        # Numeric comparison
                        np.testing.assert_array_equal(d1, d2, err_msg=f"Array mismatch for {key} in split {split}")

                # Check metadata
                m1 = f1["metadata"]
                m2 = f2["metadata"]
                for attr in ["split", "config_hash", "seed", "dataset_version", "nsamples_per_point", "git_commit", "git_branch", "dataset_schema_version"]:
                    assert attr in m1.attrs and attr in m2.attrs, f"Metadata attribute {attr} missing"
                    
                    val1 = m1.attrs[attr]
                    val2 = m2.attrs[attr]
                    if isinstance(val1, bytes): val1 = val1.decode('utf-8')
                    if isinstance(val2, bytes): val2 = val2.decode('utf-8')
                    
                    assert val1 == val2, f"Metadata attribute {attr} value mismatch"

                # Check parameter ranges group
                g1 = f1["metadata/parameter_ranges"]
                g2 = f2["metadata/parameter_ranges"]
                for p in ["z", "h", "OmegaM", "As", "OmegaB", "ns", "zeq"]:
                    assert p in g1.attrs and p in g2.attrs
                    np.testing.assert_array_equal(g1.attrs[p], g2.attrs[p])

    finally:
        if os.path.exists(dir1): shutil.rmtree(dir1)
        if os.path.exists(dir2): shutil.rmtree(dir2)
