import sys
import os
import argparse
import h5py
import numpy as np

def validate_dataset(filepath):
    print(f"Validating {filepath}...")
    try:
        with h5py.File(filepath, 'r') as f:
            # Check shape consistency
            num_points = len(f['samples/z'])
            assert len(f['samples/h']) == num_points
            assert len(f['samples/OmegaM']) == num_points
            assert len(f['samples/sigma8']) == num_points

            lnmu = f['samples/lnmu']
            counts = f['samples/valid_counts'][:]
            assert lnmu.shape[0] == num_points

            # Check NaN fractions
            total_elements = np.prod(lnmu.shape)
            nan_count = np.sum(np.isnan(lnmu[:]))
            nan_frac = nan_count / total_elements
            print(f"  Shape: {lnmu.shape}")
            print(f"  Valid counts range: [{np.min(counts)}, {np.max(counts)}]")
            print(f"  Global NaN fraction: {nan_frac:.4f}")
            assert nan_frac < 0.99, "Catastrophic NaN fraction detected."

            # Check metadata completeness
            meta = f['metadata']
            assert 'split' in meta.attrs
            assert 'config_hash' in meta.attrs

            pre = f['metadata/preprocessing']
            assert 'lnmu_mean' in pre.attrs

            # Check duplicates roughly
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
    args = parser.parse_args()

    all_valid = True
    for split in ['train', 'validation', 'test']:
        path = os.path.join(args.dataset_dir, split, f"dataset_{split}.h5")
        if os.path.exists(path):
            if not validate_dataset(path):
                all_valid = False

    sys.exit(0 if all_valid else 1)
