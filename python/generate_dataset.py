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

SPLIT_SEED_OFFSETS = {
    "train": 1000,
    "validation": 2000,
    "test": 3000,
}

def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return commit.decode('utf-8').strip()
    except Exception:
        return "unknown"

def partition_samples(params_dict, num_points):
    """
    Partitions generated parameters into train, val, and test using
    a multidimensional checkerboard pattern on (z, h, OmegaM, sigma8).
    """
    keys = ["z", "h", "OmegaM", "sigma8"]
    checker_index_sum = np.zeros(num_points, dtype=np.int32)

    for key in keys:
        values = params_dict[key]
        bins = np.linspace(np.min(values), np.max(values), 5)
        idx = np.digitize(values, bins) - 1
        idx[idx == 4] = 3
        checker_index_sum += idx

    # Checkerboard: sum of indices is even -> Train, odd -> Val/Test
    checker = (checker_index_sum % 2) == 0

    train_mask = checker
    non_train_mask = ~checker

    val_mask = np.zeros_like(train_mask, dtype=bool)
    test_mask = np.zeros_like(train_mask, dtype=bool)

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

    start_time = time.time()
    print(f"\n--- Generating {split_name} split ({num_points} points) ---")

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1e6
        cpu_count = psutil.cpu_count()
    except ImportError:
        psutil = None
        process = None
        mem_before = np.nan
        cpu_count = os.cpu_count()

    sum_lnmu = 0.0
    sum_sq_lnmu = 0.0
    total_valid_samples = 0
    min_lnmu = None
    max_lnmu = None

    with h5py.File(output_file, 'w') as f:
        g_samples = f.create_group("samples")

        ds_lnmu = g_samples.create_dataset(
            "lnmu",
            shape=(num_points, nsamples_per_point),
            dtype="float32",
            compression="gzip",
            chunks=(1, nsamples_per_point),
            fillvalue=np.nan,
        )
        ds_counts = g_samples.create_dataset(
            "valid_counts",
            shape=(num_points,),
            dtype="int32",
            compression="gzip",
        )

        g_samples.create_dataset("z", data=params['z'], compression="gzip")
        g_samples.create_dataset("h", data=params['h'], compression="gzip")
        g_samples.create_dataset("OmegaM", data=params['OmegaM'], compression="gzip")
        g_samples.create_dataset("sigma8", data=params['sigma8'], compression="gzip")

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
            valid_len = min(len(lnmu), nsamples_per_point)

            ds_counts[i] = valid_len
            if valid_len > 0:
                row = lnmu[:valid_len]
                ds_lnmu[i, :valid_len] = row.astype(np.float32)

                row_min = float(np.min(row))
                row_max = float(np.max(row))
                min_lnmu = row_min if min_lnmu is None else min(min_lnmu, row_min)
                max_lnmu = row_max if max_lnmu is None else max(max_lnmu, row_max)

                sum_lnmu += float(np.sum(row))
                sum_sq_lnmu += float(np.sum(row**2))
                total_valid_samples += valid_len

        sim_time = time.time() - start_time
        mem_after = process.memory_info().rss / 1e6 if process is not None else np.nan

        # Preprocessing metadata
        g_pre = f.create_group("metadata/preprocessing")
        if total_valid_samples > 0:
            global_mean = sum_lnmu / total_valid_samples
            global_var = (sum_sq_lnmu / total_valid_samples) - global_mean**2
            global_std = np.sqrt(max(0.0, global_var))
        else:
            global_mean = 0.0
            global_std = 0.0

        g_pre.attrs["lnmu_mean"] = global_mean
        g_pre.attrs["lnmu_std"] = global_std
        g_pre.attrs["lnmu_min"] = float(min_lnmu) if min_lnmu is not None else 0.0
        g_pre.attrs["lnmu_max"] = float(max_lnmu) if max_lnmu is not None else 0.0

        for k in ['z', 'h', 'OmegaM', 'sigma8']:
            g_pre.attrs[f"{k}_mean"] = float(np.mean(params[k]))
            g_pre.attrs[f"{k}_std"] = float(np.std(params[k]))
            g_pre.attrs[f"{k}_min"] = float(np.min(params[k]))
            g_pre.attrs[f"{k}_max"] = float(np.max(params[k]))

        g_pre.attrs["recommended_transform"] = "standardize"

        # Resources metadata
        g_res = f.create_group("metadata/resources")
        g_res.attrs["generation_time_sec"] = sim_time
        g_res.attrs["throughput_samples_per_sec"] = total_valid_samples / sim_time if sim_time > 0 else 0
        g_res.attrs["mem_before_mb"] = mem_before
        g_res.attrs["mem_after_mb"] = mem_after
        g_res.attrs["peak_memory_proxy_mb"] = np.nanmax([mem_before, mem_after]) if not (np.isnan(mem_before) and np.isnan(mem_after)) else np.nan
        g_res.attrs["cpu_count"] = cpu_count if cpu_count is not None else -1
        g_res.attrs["total_valid_samples"] = int(total_valid_samples)
        g_res.attrs["valid_fraction"] = float(total_valid_samples / (num_points * nsamples_per_point)) if num_points > 0 and nsamples_per_point > 0 else 0.0

        # General metadata
        g_meta = f["metadata"]
        g_meta.attrs["split"] = split_name
        g_meta.attrs["split_label"] = "train" if split_name == "train" else ("interpolation" if split_name in ["validation", "test"] else "OoD")
        g_meta.attrs["seed"] = seed if seed is not None else "stochastic"
        g_meta.attrs["git_commit"] = get_git_commit()
        g_meta.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        g_meta.attrs["nsamples_per_point"] = nsamples_per_point
        g_meta.attrs["dataset_version"] = "1.1"

        sim_config = gw.get_simulator_config()
        g_meta.attrs["filaments"] = sim_config["filaments"]
        g_meta.attrs["bias"] = sim_config["bias"]
        g_meta.attrs["ell"] = sim_config["ell"]
        g_meta.attrs["Nhalos"] = sim_config["Nhalos"]

        cfg_str = json.dumps({k: params[k].tolist() for k in params}) + str(seed) + str(nsamples_per_point)
        g_meta.attrs["config_hash"] = hashlib.md5(cfg_str.encode()).hexdigest()

    print(f"Simulation for {split_name} completed in {sim_time:.2f} seconds.")

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
        split_seed = None if args.seed is None else args.seed + SPLIT_SEED_OFFSETS[split_name]
        generate_dataset_split(split_params, split_name, out_file, args.nsamples, split_seed, args)

if __name__ == "__main__":
    main()
