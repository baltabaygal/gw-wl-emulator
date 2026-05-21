import sys
import os
import argparse
import time
import subprocess
import h5py
import numpy as np
import hashlib
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

try:
    from pyDOE3 import lhs
except ImportError:
    try:
        from pyDOE2 import lhs
    except ImportError:
        from pyDOE import lhs

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return commit.decode('utf-8').strip()
    except Exception:
        return "unknown"

def partition_samples(params_dict, num_points):
    """
    Partitions generated parameters into train, val, and test using an alternating interval approach.
    We'll bin sigma8 into 10 intervals.
    Even intervals -> Train
    Odd intervals -> split into Val / Test
    """
    sigma8 = params_dict['sigma8']
    min_s8, max_s8 = np.min(sigma8), np.max(sigma8)
    # create 10 bins
    bins = np.linspace(min_s8, max_s8, 11)
    indices = np.digitize(sigma8, bins) - 1
    # fix edge case where digitize returns 10
    indices[indices == 10] = 9

    train_mask = (indices % 2 == 0)
    non_train_mask = ~train_mask

    val_mask = np.zeros_like(train_mask, dtype=bool)
    test_mask = np.zeros_like(train_mask, dtype=bool)

    # of the non_train, assign half to val and half to test randomly or by another parameter
    non_train_idx = np.where(non_train_mask)[0]
    np.random.shuffle(non_train_idx)
    mid = len(non_train_idx) // 2
    val_mask[non_train_idx[:mid]] = True
    test_mask[non_train_idx[mid:]] = True

    return train_mask, val_mask, test_mask

def generate_dataset_split(params, split_name, output_file, nsamples_per_point, seed, config):
    num_points = len(params['z'])
    if num_points == 0:
        return

    print(f"\n--- Generating {split_name} split ({num_points} points) ---")
    start_time = time.time()

    chunk_size = 100 # write in chunks to save memory

    with h5py.File(output_file, 'w') as f:
        g_samples = f.create_group("samples")

        # We need VLEN for lnmu because rows might have different valid lengths (muj > 0)
        # Using HDF5 variable-length data type
        vlen_type = h5py.vlen_dtype(np.float64)
        ds_lnmu = g_samples.create_dataset("lnmu", (num_points,), dtype=vlen_type, compression="gzip")

        ds_z = g_samples.create_dataset("z", data=params['z'], compression="gzip")
        ds_h = g_samples.create_dataset("h", data=params['h'], compression="gzip")
        ds_OmegaM = g_samples.create_dataset("OmegaM", data=params['OmegaM'], compression="gzip")
        ds_sigma8 = g_samples.create_dataset("sigma8", data=params['sigma8'], compression="gzip")

        # Preprocessing metadata accumulators
        sum_lnmu = 0.0
        sum_sq_lnmu = 0.0
        total_valid_samples = 0

        for i in range(num_points):
            if i > 0 and i % 10 == 0:
                print(f"  Processed {i}/{num_points} configurations...")

            sim_seed = None if seed is None else seed + i

            lnmu = gw.sample_lnmu_ml(
                params['z'][i],
                params['h'][i],
                params['OmegaM'][i],
                params['sigma8'][i],
                nsamples_per_point,
                sim_seed
            )

            # Remove nans or invalid
            lnmu = lnmu[~np.isnan(lnmu)]

            ds_lnmu[i] = lnmu

            if len(lnmu) > 0:
                sum_lnmu += np.sum(lnmu)
                sum_sq_lnmu += np.sum(lnmu**2)
                total_valid_samples += len(lnmu)

        # Issue 4: Preprocessing stats
        g_pre = f.create_group("metadata/preprocessing")
        if total_valid_samples > 0:
            global_mean = sum_lnmu / total_valid_samples
            global_var = (sum_sq_lnmu / total_valid_samples) - global_mean**2
            global_std = np.sqrt(global_var) if global_var > 0 else 0.0
        else:
            global_mean = 0.0
            global_std = 0.0

        g_pre.attrs["global_mean"] = global_mean
        g_pre.attrs["global_std"] = global_std
        g_pre.attrs["recommended_transform"] = "standardize"

        # Issue 6, 7: Metadata
        g_meta = f["metadata"]
        g_meta.attrs["split"] = split_name
        g_meta.attrs["split_label"] = "train" if split_name == "train" else ("interpolation" if split_name in ["validation", "test"] else "OoD")
        g_meta.attrs["seed"] = seed if seed is not None else "stochastic"
        g_meta.attrs["git_commit"] = get_git_commit()
        g_meta.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        g_meta.attrs["nsamples_per_point"] = nsamples_per_point
        g_meta.attrs["dataset_version"] = "1.0"

        sim_config = gw.get_simulator_config()
        g_meta.attrs["filaments"] = sim_config["filaments"]
        g_meta.attrs["bias"] = sim_config["bias"]
        g_meta.attrs["ell"] = sim_config["ell"]
        g_meta.attrs["Nhalos"] = sim_config["Nhalos"]

        cfg_str = json.dumps({k: params[k].tolist() for k in params}) + str(seed) + str(nsamples_per_point)
        g_meta.attrs["config_hash"] = hashlib.md5(cfg_str.encode()).hexdigest()

    end_time = time.time()
    print(f"Simulation for {split_name} completed in {end_time - start_time:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--nsamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    bounds = {
        'h': (0.59, 0.76),
        'OmegaM': (0.15, 0.47),
        'sigma8': (0.4, 1.4),
        'z': (0.01, 10.0)
    }

    keys = ['h', 'OmegaM', 'sigma8', 'z']

    # Generate all LHS points
    # Need to generate enough points so splits get populated
    total_points = args.num_points * 2
    lhs_samples = lhs(4, samples=total_points)

    params = {}
    for i, k in enumerate(keys):
        low, high = bounds[k]
        params[k] = low + lhs_samples[:, i] * (high - low)

    train_mask, val_mask, test_mask = partition_samples(params, total_points)

    splits = {
        'train': train_mask,
        'validation': val_mask,
        'test': test_mask
    }

    for split_name, mask in splits.items():
        if np.sum(mask) == 0:
            continue
        split_params = {k: v[mask] for k, v in params.items()}
        out_file = os.path.join(args.output_dir, split_name, f"dataset_{split_name}.h5")
        split_seed = None if args.seed is None else args.seed + hash(split_name) % 10000
        generate_dataset_split(split_params, split_name, out_file, args.nsamples, split_seed, args)

if __name__ == "__main__":
    main()
