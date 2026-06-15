import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import pytest
import torch

from ml.nsf_dataset import prepare_nsf_data, get_nsf_dataloaders

def test_nsf_dataset_preprocessing_and_loading():
    dataset_dir = "datasets_backend_current_1k"
    stats_path = "data/models/nsf_preprocessing_stats_test.json"
    
    # Remove test stats file if it already exists
    if os.path.exists(stats_path):
        os.remove(stats_path)
        
    try:
        # 1. Run preparation
        stats = prepare_nsf_data(dataset_dir, output_stats_path=stats_path)
        
        # Verify file creation and contents
        assert os.path.exists(stats_path)
        with open(stats_path, "r") as f:
            loaded_stats = json.load(f)
            
        assert "context_mean" in loaded_stats
        assert "context_std" in loaded_stats
        assert "lnmu_mean" in loaded_stats
        assert "lnmu_std" in loaded_stats
        assert len(loaded_stats["context_mean"]) == 4
        assert len(loaded_stats["context_std"]) == 4
        assert isinstance(loaded_stats["lnmu_mean"], float)
        
        # 2. Get Dataloaders
        loaders, stats_returned = get_nsf_dataloaders(dataset_dir, batch_size=64, stats_path=stats_path)
        
        assert "train" in loaders
        assert "validation" in loaders
        assert "test" in loaders
        
        # Check a batch
        train_loader = loaders["train"]
        x_batch, y_batch = next(iter(train_loader))
        
        assert x_batch.shape == (64, 4)
        assert y_batch.shape == (64, 1)
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32
        
        # Verify that standardization maps training set approximately to mean 0, variance 1
        all_x = []
        all_y = []
        for x, y in train_loader:
            all_x.append(x.numpy())
            all_y.append(y.numpy())
            
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        
        # Since of drop_last and mini-batch, it might not be exactly 0 and 1, but very close
        np.testing.assert_allclose(np.mean(all_x, axis=0), np.zeros(4), atol=1e-2)
        np.testing.assert_allclose(np.std(all_x, axis=0), np.ones(4), atol=1e-2)
        assert abs(np.mean(all_y)) < 1e-2
        assert abs(np.std(all_y) - 1.0) < 1e-2
        
    finally:
        # Clean up test stats
        if os.path.exists(stats_path):
            os.remove(stats_path)
