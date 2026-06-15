import sys
import os
import argparse
import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import DEFAULT_MIN_VALID_FRACTION

def validate_dataset(filepath, min_valid_fraction=DEFAULT_MIN_VALID_FRACTION):
    print(f"Validating {filepath}...")
    try:
        with h5py.File(filepath, 'r') as f:
            required_paths = [
                "samples/lnmu",
                "samples/valid_counts",
                "samples/z",
                "samples/h",
                "samples/OmegaM",
                "samples/sigma8",
                "samples/split_type",
                "metadata",
                "metadata/preprocessing",
                "metadata/invalid_stats",
                "metadata/parameter_ranges",
            ]
            for path in required_paths:
                assert path in f, f"Missing required path: {path}"

            num_points = len(f['samples/z'])
            assert len(f['samples/h']) == num_points, "samples/h length mismatch"
            assert len(f['samples/OmegaM']) == num_points, "samples/OmegaM length mismatch"
            assert len(f['samples/sigma8']) == num_points, "samples/sigma8 length mismatch"
            assert len(f['samples/split_type']) == num_points, "samples/split_type length mismatch"

            lnmu = f['samples/lnmu']
            assert lnmu.ndim == 2, "samples/lnmu must be 2D"
            assert lnmu.shape[0] == num_points, "samples/lnmu first dimension mismatch"

            counts = f['samples/valid_counts'][:]
            assert counts.shape == (num_points,), "samples/valid_counts shape mismatch"
            assert np.all(counts >= 0), "Found negative valid_counts"
            assert np.all(counts <= lnmu.shape[1]), "Found valid_counts larger than lnmu row length"

            total_elements = int(np.prod(lnmu.shape))
            total_valid = int(np.sum(counts))
            total_nan = 0

            for i in range(num_points):
                row = lnmu[i]
                valid_count = int(counts[i])
                assert np.all(np.isfinite(row[:valid_count])), f"Non-finite values in valid region at row {i}"
                assert np.all(np.isnan(row[valid_count:])), f"Non-NaN values in padded region at row {i}"
                if valid_count == 0:
                    assert np.all(np.isnan(row)), f"Expected full-NaN row for valid_count==0 at row {i}"
                total_nan += int(np.sum(np.isnan(row)))

            nan_frac = total_nan / total_elements if total_elements > 0 else 0.0
            valid_fraction = total_valid / total_elements if total_elements > 0 else 0.0
            print(f"  Shape: {lnmu.shape}")
            print(f"  Valid counts range: [{np.min(counts)}, {np.max(counts)}]")
            print(f"  Global NaN fraction: {nan_frac:.4f}")
            print(f"  Valid fraction: {valid_fraction:.4f}")
            assert valid_fraction >= min_valid_fraction, (
                f"Valid fraction {valid_fraction:.4f} below threshold {min_valid_fraction:.4f}"
            )

            # Check metadata completeness
            meta = f['metadata']
            split_name = meta.attrs.get("split")
            
            required_meta_attrs = [
                "split", "config_hash", "seed", "dataset_version", 
                "nsamples_per_point", "git_commit", "git_branch",
                "dataset_schema_version", "generation_timestamp"
            ]
            for attr in required_meta_attrs:
                assert attr in meta.attrs, f"Missing metadata attribute: {attr}"

            pre = f['metadata/preprocessing']
            for attr in ["lnmu_mean", "lnmu_std", "lnmu_min", "lnmu_max"]:
                assert attr in pre.attrs, f"Missing preprocessing attribute: {attr}"

            inv = f['metadata/invalid_stats']
            for attr in [
                "total_samples", "valid_samples", "invalid_samples",
                "negative_detA", "nonfinite_mu", "negative_mu",
                "nan_kappa", "nan_gamma", "overflow_mu", "invalid_logmu",
                "invalid_fraction", "detA_min", "detA_max", "detA_mean",
            ]:
                assert attr in inv.attrs, f"Missing invalid_stats attribute: {attr}"

            # Check parameter ranges group
            pr = f['metadata/parameter_ranges']
            for attr in ["z", "h", "OmegaM", "sigma8"]:
                assert attr in pr.attrs, f"Missing parameter ranges attribute: {attr}"
                val = pr.attrs[attr]
                assert len(val) == 2
                assert val[0] < val[1]

            # Check split types dataset consistency
            split_types = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['samples/split_type'][:]]
            if split_name == "train":
                assert all(t == "train" for t in split_types), f"Found non-train split_type in train split: {set(split_types)}"
            elif split_name in ["validation", "test"]:
                assert all(t in ["interpolation", "ood"] for t in split_types), f"Invalid split_type in {split_name}: {set(split_types)}"
                
                # Validate the boundaries
                om = f['samples/OmegaM'][:]
                s8 = f['samples/sigma8'][:]
                for t, o_val, s_val in zip(split_types, om, s8):
                    is_id = (0.20 <= o_val <= 0.40) and (0.65 <= s_val <= 1.05)
                    is_ood = ((o_val > 0.40) and (s_val > 1.05)) or ((o_val < 0.20) and (s_val < 0.65))
                    if t == "interpolation":
                        assert is_id, f"split_type 'interpolation' but parameter outside ID range: OmegaM={o_val}, sigma8={s_val}"
                    elif t == "ood":
                        assert is_ood, f"split_type 'ood' but parameter outside True OoD range: OmegaM={o_val}, sigma8={s_val}"

            # Check duplicates using only conditioning parameters
            params = np.column_stack([f['samples/z'][:], f['samples/h'][:], f['samples/OmegaM'][:], f['samples/sigma8'][:]])
            unique_params = np.unique(params, axis=0)
            assert len(unique_params) == num_points, "Duplicate cosmologies found."

        print("  [OK] Dataset is valid.")
        return True
    except Exception as e:
        print(f"  [FAIL] Validation failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--min_valid_fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    args = parser.parse_args()

    all_valid = True
    for split in ['train', 'validation', 'test']:
        path = os.path.join(args.dataset_dir, split, f"dataset_{split}.h5")
        if os.path.exists(path):
            if not validate_dataset(path, min_valid_fraction=args.min_valid_fraction):
                all_valid = False

    sys.exit(0 if all_valid else 1)
