import os
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


def load_split(path: str) -> Dict[str, np.ndarray]:
    """Load one HDF5 split into memory as numpy arrays."""
    with h5py.File(path, "r") as f:
        out = {
            "lnmu": f["samples/lnmu"][:],
            "valid_counts": f["samples/valid_counts"][:],
            "z": f["samples/z"][:],
            "h": f["samples/h"][:],
            "OmegaM": f["samples/OmegaM"][:],
            "sigma8": f["samples/sigma8"][:],
            "metadata": dict(f["metadata"].attrs.items()),
            "preprocessing": dict(f["metadata/preprocessing"].attrs.items()),
        }
    return out


def load_dataset(dataset_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all available splits from dataset_dir."""
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        if os.path.exists(path):
            result[split] = load_split(path)
    return result


def flatten_dataset(
    split_data: Dict[str, np.ndarray],
    normalize: bool = False,
    lnmu_mean: Optional[float] = None,
    lnmu_std: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten padded lnmu rows into ML-ready (X, Y) arrays.

    Returns:
      X: float32, shape (N_valid, 4), columns [z, h, OmegaM, sigma8]
      Y: float32, shape (N_valid, 1), values lnmu
    """
    lnmu = split_data["lnmu"]
    counts = split_data["valid_counts"].astype(np.int64)
    z = split_data["z"]
    h = split_data["h"]
    om = split_data["OmegaM"]
    s8 = split_data["sigma8"]

    total_valid = int(np.sum(counts))
    X = np.empty((total_valid, 4), dtype=np.float32)
    Y = np.empty((total_valid, 1), dtype=np.float32)

    if normalize:
        if lnmu_mean is None or lnmu_std is None:
            pre = split_data.get("preprocessing", {})
            lnmu_mean = float(pre.get("lnmu_mean", 0.0))
            lnmu_std = float(pre.get("lnmu_std", 1.0))
        if lnmu_std == 0:
            lnmu_std = 1.0

    cursor = 0
    for i in range(lnmu.shape[0]):
        n = int(counts[i])
        if n <= 0:
            continue
        row = lnmu[i, :n]

        X[cursor:cursor + n, 0] = z[i]
        X[cursor:cursor + n, 1] = h[i]
        X[cursor:cursor + n, 2] = om[i]
        X[cursor:cursor + n, 3] = s8[i]

        if normalize:
            Y[cursor:cursor + n, 0] = (row - lnmu_mean) / lnmu_std
        else:
            Y[cursor:cursor + n, 0] = row

        cursor += n

    return X, Y
