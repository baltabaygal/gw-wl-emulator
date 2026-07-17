import os
import sys
import subprocess
import h5py
import numpy as np
import pytest

def get_test_dataset_dir():
    d = "datasets/test_suite"
    if not os.path.exists(d):
        subprocess.run([
            sys.executable, "python/generate_dataset.py",
            "--num_points", "15",
            "--nsamples", "10",
            "--seed", "123",
            "--output_dir", d
        ], check=True)
    return d

def test_split_integrity():
    dataset_dir = get_test_dataset_dir()
    splits = ["train", "validation", "test"]
    
    data = {}
    for split in splits:
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                coords = np.column_stack([
                    f["samples/z"][:],
                    f["samples/h"][:],
                    f["samples/OmegaM"][:],
                    f["samples/sigma8"][:]
                ])
                data[split] = coords

    # Verify no duplicates across splits
    for s1 in data:
        for s2 in data:
            if s1 == s2:
                continue
            c1 = data[s1]
            c2 = data[s2]
            
            if len(c1) == 0 or len(c2) == 0:
                continue

            # Compute intersection of rows
            nrows, ncols = c1.shape
            dtype = {'names': [f'f{i}' for i in range(ncols)], 'formats': [c1.dtype]*ncols}
            
            c1_struct = c1.view(dtype)
            c2_struct = c2.view(dtype)
            
            common = np.intersect1d(c1_struct, c2_struct)
            assert len(common) == 0, f"Overlap found between split {s1} and split {s2}: {common}"
