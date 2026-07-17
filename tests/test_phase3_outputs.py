import json

import numpy as np
import pytest

from ml.cache_utils import CacheManager
from ml.phase3_common import save_npz_with_metadata


def test_phase3_cache_refuses_stale_metadata(tmp_path):
    cm = CacheManager(cache_dir=str(tmp_path))
    key = {"catalog_hash": "abc", "grid_definition": {"n": 2}, "prior_bounds": {"h": [0.59, 0.76]}}
    cm.save("phase3_unit", key, {"x": np.array([1.0])})
    path = cm.get_cache_path("phase3_unit", key)
    np.savez(path, metadata=json.dumps({**key, "catalog_hash": "stale"}), x=np.array([1.0]))
    with pytest.raises(ValueError):
        cm.load("phase3_unit", key)


def test_phase3_required_output_npz_metadata_roundtrip(tmp_path):
    path = tmp_path / "grid.npz"
    metadata = {
        "catalog_hash": "abc",
        "grid_definition": {"grid_type": "2d"},
        "prior_bounds": {"h": [0.59, 0.76], "OmegaM": [0.20, 0.40], "sigma8": [0.65, 1.05]},
        "simulator_git_commit": "commit",
        "likelihood_version": "phase3_v1",
    }
    save_npz_with_metadata(path, {"sim_log_likelihood": np.zeros((2, 2))}, metadata)
    assert path.exists()
    with np.load(path, allow_pickle=True) as npz:
        loaded = json.loads(str(npz["metadata"]))
    for key in metadata:
        assert key in loaded
