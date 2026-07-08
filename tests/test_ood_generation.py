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

def test_ood_separation():
    dataset_dir = get_test_dataset_dir()
    
    train_path = os.path.join(dataset_dir, "train", "dataset_train.h5")
    val_path = os.path.join(dataset_dir, "validation", "dataset_validation.h5")
    test_path = os.path.join(dataset_dir, "test", "dataset_test.h5")
    
    # 1. Train split checks
    if os.path.exists(train_path):
        with h5py.File(train_path, 'r') as f:
            om = f["samples/OmegaM"][:]
            s8 = f["samples/sigma8"][:]
            split_types = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f["samples/split_type"][:]]
            
            # Training OmegaM and sigma8 must be in ID region
            assert np.all((om >= 0.20) & (om <= 0.40)), "Found training OmegaM outside [0.20, 0.40]"
            assert np.all((s8 >= 0.65) & (s8 <= 1.05)), "Found training sigma8 outside [0.65, 1.05]"
            # All training split types must be 'train'
            assert all(t == 'train' for t in split_types), "Found training split_type other than 'train'"
            
    # 2. Validation / Test split checks
    for path, name in [(val_path, "validation"), (test_path, "test")]:
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                om = f["samples/OmegaM"][:]
                s8 = f["samples/sigma8"][:]
                split_types = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f["samples/split_type"][:]]
                
                for o_val, s_val, s_type in zip(om, s8, split_types):
                    # Check that intermediate points are indeed filtered out
                    is_id = (0.20 <= o_val <= 0.40) and (0.65 <= s_val <= 1.05)
                    is_ood = ((o_val > 0.40) and (s_val > 1.05)) or ((o_val < 0.20) and (s_val < 0.65))
                    
                    assert is_id or is_ood, f"Found filtered boundary point in {name}: OmegaM={o_val}, sigma8={s_val}"
                    
                    if is_id:
                        assert s_type == 'interpolation', f"Expected split_type 'interpolation' for ID point in {name}: OmegaM={o_val}, sigma8={s_val}"
                    elif is_ood:
                        assert s_type == 'ood', f"Expected split_type 'ood' for True OoD point in {name}: OmegaM={o_val}, sigma8={s_val}"
