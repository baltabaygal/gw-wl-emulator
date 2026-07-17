import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import hashlib
import subprocess
import numpy as np

def get_git_commit() -> str:
    """Retrieves the current Git commit hash of the repository."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown_commit"

class CacheManager:
    """Manager for metadata-safe caching of expensive simulator objects."""
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self.likelihood_version = "v1"

    def _clean_dict(self, d):
        """Recursively converts NumPy types in a dict/list to Python native types for JSON sorting."""
        if isinstance(d, dict):
            return {k: self._clean_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [self._clean_dict(x) for x in d]
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif isinstance(d, (np.integer, np.floating)):
            return d.item()
        return d

    def get_cache_hash(self, key_dict: dict) -> str:
        """Computes a SHA256 hash of a serialized, key-sorted key dictionary."""
        cleaned = self._clean_dict(key_dict)
        serialized = json.dumps(cleaned, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_cache_path(self, category: str, key_dict: dict) -> str:
        """Returns the absolute file path for a cached category file, injecting default keys."""
        full_keys = key_dict.copy()
        if "simulator_git_commit" not in full_keys:
            full_keys["simulator_git_commit"] = get_git_commit()
        if "likelihood_version" not in full_keys:
            full_keys["likelihood_version"] = self.likelihood_version
            
        cache_hash = self.get_cache_hash(full_keys)
        return os.path.join(self.cache_dir, f"{category}_{cache_hash}.npz")

    def load(self, category: str, key_dict: dict) -> dict | None:
        """Loads data from cache if it exists and keys match exactly. Fails loudly on mismatch."""
        path = self.get_cache_path(category, key_dict)
        if not os.path.exists(path):
            return None

        # Prepare expected keys (with injected defaults)
        expected_keys = key_dict.copy()
        if "simulator_git_commit" not in expected_keys:
            expected_keys["simulator_git_commit"] = get_git_commit()
        if "likelihood_version" not in expected_keys:
            expected_keys["likelihood_version"] = self.likelihood_version
        cleaned_expected = self._clean_dict(expected_keys)

        print(f"Loading cached {category} from {path}...")
        try:
            with np.load(path, allow_pickle=True) as npz:
                metadata = json.loads(str(npz["metadata"]))
                
                # Check all expected keys match metadata inside npz file
                for k, val in cleaned_expected.items():
                    if k not in metadata:
                        raise ValueError(f"Cache metadata key '{k}' missing from cached file.")
                    cached_val = metadata[k]
                    if cached_val != val:
                        raise ValueError(
                            f"Cache metadata mismatch for key '{k}': "
                            f"Expected {val}, found {cached_val} in cache."
                        )
                
                # Retrieve all array items except metadata
                data = {}
                for file_name in npz.files:
                    if file_name != "metadata":
                        data[file_name] = npz[file_name]
                return data
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Failed to read cache file {path}: {str(e)}") from e

    def save(self, category: str, key_dict: dict, data_dict: dict):
        """Saves data dict and metadata to cache."""
        path = self.get_cache_path(category, key_dict)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare metadata (with injected defaults)
        full_keys = key_dict.copy()
        if "simulator_git_commit" not in full_keys:
            full_keys["simulator_git_commit"] = get_git_commit()
        if "likelihood_version" not in full_keys:
            full_keys["likelihood_version"] = self.likelihood_version
            
        cleaned_metadata = self._clean_dict(full_keys)
        np.savez(path, metadata=json.dumps(cleaned_metadata), **data_dict)
        print(f"Saved {category} to cache at {path}")
