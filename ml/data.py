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
        if "samples/split_type" in f:
            out["split_type"] = f["samples/split_type"][:]
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


def get_recommended_bin_edges() -> np.ndarray:
    """Returns the recommended 100-bin hybrid (linear-log) bin edges (101 elements)."""
    neg_edges = np.linspace(-0.5, -0.1, 15, endpoint=False)
    peak_edges = np.linspace(-0.1, 0.2, 65, endpoint=False)
    pos_edges = np.geomspace(0.2 + 1.0, 2.5 + 1.0, 21) - 1.0
    return np.concatenate([neg_edges, peak_edges, pos_edges])


def load_histogram_dataset(
    dataset_dir: str,
    bin_edges: np.ndarray,
) -> dict:
    """Loads dataset splits and preprocesses lnmu samples into normalized histogram probability vectors.

    Returns:
      dict mapping split name ("train", "validation", "test") to a tuple:
        X: shape (C, 4) - parameters [z, h, OmegaM, sigma8]
        Y: shape (C, 100) - normalized bin probabilities (sum to 1.0)
        split_types: list of len C - config split types ('train', 'interpolation', 'ood')
    """
    raw_data = load_dataset(dataset_dir)
    result = {}

    for split_name, split_data in raw_data.items():
        lnmu = split_data["lnmu"]
        counts = split_data["valid_counts"]
        z = split_data["z"]
        h = split_data["h"]
        om = split_data["OmegaM"]
        s8 = split_data["sigma8"]

        num_configs = lnmu.shape[0]
        X = np.empty((num_configs, 4), dtype=np.float32)
        Y = np.empty((num_configs, len(bin_edges) - 1), dtype=np.float32)

        split_types = []
        if "split_type" in split_data:
            split_types = [
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in split_data["split_type"]
            ]
        else:
            # Fallback for older datasets
            for i in range(num_configs):
                o_val = om[i]
                s_val = s8[i]
                is_id = (0.20 <= o_val <= 0.40) and (0.65 <= s_val <= 1.05)
                is_ood = ((o_val > 0.40) and (s_val > 1.05)) or ((o_val < 0.20) and (s_val < 0.65))
                if split_name == "train":
                    st = "train"
                else:
                    st = "interpolation" if is_id else ("ood" if is_ood else "boundary")
                split_types.append(st)

        for i in range(num_configs):
            X[i, 0] = z[i]
            X[i, 1] = h[i]
            X[i, 2] = om[i]
            X[i, 3] = s8[i]

            n = int(counts[i])
            if n > 0:
                row = lnmu[i, :n]
                # Clamp outliers to ensure they fall within the outermost bins
                clamped_row = np.clip(row, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
                counts_hist, _ = np.histogram(clamped_row, bins=bin_edges)
                total_c = np.sum(counts_hist)
                if total_c > 0:
                    probs = counts_hist / total_c
                else:
                    probs = np.ones(len(bin_edges) - 1, dtype=np.float32) / (len(bin_edges) - 1)
                Y[i] = probs.astype(np.float32)
            else:
                Y[i] = np.ones(len(bin_edges) - 1, dtype=np.float32) / (len(bin_edges) - 1)

        result[split_name] = (X, Y, split_types)

    return result
