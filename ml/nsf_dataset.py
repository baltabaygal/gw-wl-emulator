import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ml.data import load_dataset, flatten_dataset

class NSFRawDataset(Dataset):
    """PyTorch Dataset returning standardized raw samples and context."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]

def prepare_nsf_data(dataset_dir: str, output_stats_path: str = "data/models/nsf_preprocessing_stats.json") -> dict:
    """Computes preprocessing statistics using float64 to prevent float32 accumulation error."""
    dataset = load_dataset(dataset_dir)
    if "train" not in dataset:
        raise ValueError("Train split not found in dataset folder.")
    
    # Flatten the training set to get the raw samples
    X_train_raw, Y_train_raw = flatten_dataset(dataset["train"], normalize=False)
    
    # Compute mean and standard deviations in float64
    context_mean = np.mean(X_train_raw, axis=0, dtype=np.float64)
    context_std = np.std(X_train_raw, axis=0, dtype=np.float64)
    context_std[context_std == 0.0] = 1.0
    
    lnmu_mean = float(np.mean(Y_train_raw, dtype=np.float64))
    lnmu_std = float(np.std(Y_train_raw, dtype=np.float64))
    if lnmu_std == 0.0:
        lnmu_std = 1.0
        
    stats = {
        "context_mean": context_mean.tolist(),
        "context_std": context_std.tolist(),
        "lnmu_mean": lnmu_mean,
        "lnmu_std": lnmu_std
    }
    
    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
    with open(output_stats_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Preprocessing statistics written to {output_stats_path}")
    return stats

def get_nsf_dataloaders(
    dataset_dir: str,
    batch_size: int = 16384,
    stats_path: str = "data/models/nsf_preprocessing_stats.json"
) -> tuple[dict[str, DataLoader], dict]:
    """Prepares standardizing loaders for all available dataset splits using float64 for intermediate calculation."""
    if not os.path.exists(stats_path):
        stats = prepare_nsf_data(dataset_dir, stats_path)
    else:
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
    context_mean = np.array(stats["context_mean"], dtype=np.float64)
    context_std = np.array(stats["context_std"], dtype=np.float64)
    lnmu_mean = np.float64(stats["lnmu_mean"])
    lnmu_std = np.float64(stats["lnmu_std"])
    
    dataset = load_dataset(dataset_dir)
    loaders = {}
    
    for split in ["train", "validation", "test"]:
        if split in dataset:
            X_raw, Y_raw = flatten_dataset(dataset[split], normalize=False)
            
            # Normalize using float64 to ensure mean=0 and std=1, then convert to float32
            X_norm = ((X_raw.astype(np.float64) - context_mean) / context_std).astype(np.float32)
            Y_norm = ((Y_raw.astype(np.float64) - lnmu_mean) / lnmu_std).astype(np.float32)
            
            ds = NSFRawDataset(X_norm, Y_norm)
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                drop_last=(split == "train")
            )
            
    return loaders, stats
