import json

import numpy as np

from ml.phase3_common import catalog_hash, make_redshifts


def test_phase3_redshift_generation_is_reproducible():
    z1, desc1 = make_redshifts("mixed_uniform", 32, 123)
    z2, desc2 = make_redshifts("mixed_uniform", 32, 123)
    np.testing.assert_allclose(z1, z2)
    assert desc1 == desc2


def test_phase3_catalog_metadata_hash_contains_truth_values():
    z = np.array([0.5, 1.0, 1.5])
    lnmu = np.array([0.01, -0.02, 0.03])
    metadata = {
        "catalog_id": "unit",
        "catalog_type": "mixed_uniform",
        "z_distribution": {"type": "uniform"},
        "N": 3,
        "seed": 7,
        "true_h": 0.67,
        "true_OmegaM": 0.30,
        "true_sigma8": 0.85,
        "simulator_git_commit": "abc",
        "generation_timestamp": "2026-06-15T00:00:00Z",
    }
    h1 = catalog_hash(z, lnmu, metadata)
    metadata["true_sigma8"] = 0.86
    h2 = catalog_hash(z, lnmu, metadata)
    assert h1 != h2
