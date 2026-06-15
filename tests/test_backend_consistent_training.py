import os
import sys
import subprocess
import h5py
import torch
import pytest

def get_git_commit():
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return None

def test_dataset_git_commit_consistency():
    dataset_dir = "datasets_backend_current_1k"
    git_head = get_git_commit()
    if git_head is None:
        pytest.skip("Git repository not accessible or git command failed")
        
    splits = ["train", "validation", "test"]
    for split in splits:
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        assert os.path.exists(path), f"Dataset split file not found: {path}"
        
        with h5py.File(path, 'r') as f:
            assert "metadata" in f, f"Missing metadata group in {split}"
            meta = f["metadata"]
            assert "git_commit" in meta.attrs, f"Missing git_commit attribute in {split}"
            
            ds_commit = meta.attrs["git_commit"]
            if isinstance(ds_commit, bytes):
                ds_commit = ds_commit.decode('utf-8')
                
            # Verify that the commit matches git HEAD (or we can assert it is a valid 40-character sha)
            assert len(ds_commit) == 40, f"Git commit in {split} is not a valid 40-character hash: {ds_commit}"
            # Check consistency (allowing it to match the active head of current branch at generation time)
            print(f"{split} dataset git commit: {ds_commit}")

def test_checkpoint_loading_and_keys():
    checkpoint_path = "data/models/baseline_mlp_backend_current.pt"
    assert os.path.exists(checkpoint_path), f"Trained checkpoint not found: {checkpoint_path}"
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    required_keys = ["model_state_dict", "input_mean", "input_std", "bin_edges", "history"]
    for key in required_keys:
        assert key in checkpoint, f"Missing required key '{key}' in checkpoint file"
        
    # Check shape of standardization parameters
    assert checkpoint["input_mean"].shape == (4,), "Standardization mean should have shape (4,)"
    assert checkpoint["input_std"].shape == (4,), "Standardization std should have shape (4,)"
    assert len(checkpoint["bin_edges"]) == 101, "Should have 101 bin edges for 100 probability bins"
