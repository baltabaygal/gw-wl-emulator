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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gwlensing as gw
from ml.params import PRIOR_6D, WIDE_6D, PARAM_KEYS_6D

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

def get_git_branch():
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL)
        return branch.decode('utf-8').strip()
    except Exception:
        return "unknown"

def simulate_config_worker(args_tuple):
    z, h, om, sigma8, ob, ns, zeq, nsamples_per_point, seed, i = args_tuple
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
    import gwlensing as gw
    import numpy as np

    sim_seed = None if seed is None else seed + i
    # sigma8-mode: As is left at its default (-1.0), so the amplitude is set by
    # the sigma8 positional via deltaH8 = sigma8/sigmaC(M8, 1). This is the
    # upstream Vaskonen convention (halos main_lensing.cpp).
    result = gw.sample_lnmu_ml_with_diagnostics(
        z, h, om, sigma8, nsamples_per_point, sim_seed, False,
        OmegaB=ob, ns=ns, zeq=zeq
    )
    return i, list(result["lnmu"]), dict(result["invalid_stats"])


def partition_samples(params_dict, num_points, seed=None):
    """
    Partitions generated parameters into train, val, and test.
    - In-Distribution (ID): OmegaM and sigma8 inside the PRIOR_6D box
    - True Out-of-Distribution (OoD): both beyond the box on the same side
    - Boundary/intermediate points are filtered out.
    """
    omega_m = params_dict['OmegaM']
    s8 = params_dict['sigma8']
    om_lo, om_hi = PRIOR_6D['Om']
    a_lo, a_hi = PRIOR_6D['sigma8']

    id_mask = (omega_m >= om_lo) & (omega_m <= om_hi) & (s8 >= a_lo) & (s8 <= a_hi)
    ood_mask = ((omega_m > om_hi) & (s8 > a_hi)) | ((omega_m < om_lo) & (s8 < a_lo))

    train_mask = np.zeros(num_points, dtype=bool)
    val_mask = np.zeros(num_points, dtype=bool)
    test_mask = np.zeros(num_points, dtype=bool)
    split_types = ["" for _ in range(num_points)]

    keys = ["z"] + list(PARAM_KEYS_6D)
    checker_index_sum = np.zeros(num_points, dtype=np.int32)

    for key in keys:
        values = params_dict[key]
        bins = np.linspace(np.min(values), np.max(values), 5)
        idx = np.digitize(values, bins) - 1
        idx[idx == 4] = 3
        checker_index_sum += idx

    checker = (checker_index_sum % 2) == 0

    # ID points partition
    id_indices = np.where(id_mask)[0]
    id_checker = checker[id_indices]

    id_train_idx = id_indices[id_checker]
    train_mask[id_train_idx] = True
    for idx in id_train_idx:
        split_types[idx] = "train"

    id_val_test_idx = id_indices[~id_checker]
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(id_val_test_idx)
    else:
        np.random.shuffle(id_val_test_idx)

    mid_id = len(id_val_test_idx) // 2
    id_val_idx = id_val_test_idx[:mid_id]
    id_test_idx = id_val_test_idx[mid_id:]

    val_mask[id_val_idx] = True
    test_mask[id_test_idx] = True
    for idx in id_val_idx:
        split_types[idx] = "interpolation"
    for idx in id_test_idx:
        split_types[idx] = "interpolation"

    # True OoD points partition
    ood_indices = np.where(ood_mask)[0]
    if seed is not None:
        rng_ood = np.random.default_rng(seed + 1)
        rng_ood.shuffle(ood_indices)
    else:
        np.random.shuffle(ood_indices)

    mid_ood = len(ood_indices) // 2
    ood_val_idx = ood_indices[:mid_ood]
    ood_test_idx = ood_indices[mid_ood:]

    val_mask[ood_val_idx] = True
    test_mask[ood_test_idx] = True
    for idx in ood_val_idx:
        split_types[idx] = "ood"
    for idx in ood_test_idx:
        split_types[idx] = "ood"

    return train_mask, val_mask, test_mask, split_types


def generate_dataset_split(params, split_name, output_file, nsamples_per_point, seed, config, split_types, bounds):
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
    invalid_totals = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "negative_detA": 0,
        "nonfinite_mu": 0,
        "negative_mu": 0,
        "nan_kappa": 0,
        "nan_gamma": 0,
        "overflow_mu": 0,
        "invalid_logmu": 0,
        "strict_weak_lensing_rejects": 0,
        "detA_near_zero_count": 0,
    }
    detA_min = None
    detA_max = None
    detA_weighted_sum = 0.0

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
        for k in PARAM_KEYS_6D:
            g_samples.create_dataset(k, data=params[k], compression="gzip")
        # derived A_s per config (diagnostic; the amplitude input is sigma8).
        # sigma8 itself is already stored via PARAM_KEYS_6D above.
        As_derived = np.array([
            gw.get_simulator_config(h=params['h'][i], OmegaM=params['OmegaM'][i],
                                    sigma8=params['sigma8'][i], OmegaB=params['OmegaB'][i],
                                    zeq=params['zeq'][i], ns=params['ns'][i])["As_derived"]
            for i in range(num_points)
        ])
        g_samples.create_dataset("As_derived", data=As_derived, compression="gzip")

        # Write split type dataset
        dt = h5py.special_dtype(vlen=str)
        ds_split_type = g_samples.create_dataset(
            "split_type",
            shape=(num_points,),
            dtype=dt,
            compression="gzip",
        )
        ds_split_type[:] = np.array(split_types, dtype=object)

        # Prepare parallel simulation tasks
        tasks = [
            (params['z'][i], params['h'][i], params['OmegaM'][i], params['sigma8'][i],
             params['OmegaB'][i], params['ns'][i], params['zeq'][i],
             nsamples_per_point, seed, i)
            for i in range(num_points)
        ]

        from multiprocessing import Pool
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 16)
        print(f"  Simulating {num_points} configurations in parallel using {num_workers} workers...")

        with Pool(processes=num_workers) as pool:
            parallel_results = pool.map(simulate_config_worker, tasks)

        # Write results sequentially to HDF5
        for i, lnmu_list, inv in sorted(parallel_results, key=lambda x: x[0]):
            lnmu = np.array(lnmu_list)

            for k in invalid_totals:
                invalid_totals[k] += int(inv.get(k, 0))
            row_detA_min = float(inv.get("detA_min", np.nan))
            row_detA_max = float(inv.get("detA_max", np.nan))
            if np.isfinite(row_detA_min):
                detA_min = row_detA_min if detA_min is None else min(detA_min, row_detA_min)
            if np.isfinite(row_detA_max):
                detA_max = row_detA_max if detA_max is None else max(detA_max, row_detA_max)
            detA_weighted_sum += float(inv.get("detA_mean", 0.0)) * float(inv.get("total_samples", 0))

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

        for k in ['z'] + list(PARAM_KEYS_6D):
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

        g_inv = f.create_group("metadata/invalid_stats")
        for k, v in invalid_totals.items():
            g_inv.attrs[k] = int(v)
        total_samples = invalid_totals["total_samples"]
        g_inv.attrs["invalid_fraction"] = float(invalid_totals["invalid_samples"] / total_samples) if total_samples > 0 else 0.0
        g_inv.attrs["detA_min"] = float(detA_min) if detA_min is not None else 0.0
        g_inv.attrs["detA_max"] = float(detA_max) if detA_max is not None else 0.0
        g_inv.attrs["detA_mean"] = float(detA_weighted_sum / total_samples) if total_samples > 0 else 0.0
        g_inv.attrs["detA_near_zero_fraction"] = (
            float(invalid_totals["detA_near_zero_count"] / total_samples) if total_samples > 0 else 0.0
        )
        g_inv.attrs["huge_threshold"] = 1.0e12

        # General metadata
        g_meta = f["metadata"]
        g_meta.attrs["split"] = split_name
        g_meta.attrs["seed"] = seed if seed is not None else "stochastic"
        g_meta.attrs["git_commit"] = get_git_commit()
        g_meta.attrs["git_branch"] = get_git_branch()
        g_meta.attrs["generation_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        g_meta.attrs["nsamples_per_point"] = nsamples_per_point
        # 2.1 = 1+6d sigma8-mode (z; h, Om, sigma8, Ob, ns, zeq). 2.0 was the
        # A_s-mode variant, retired 2026-07-27 in favour of Vaskonen's convention.
        g_meta.attrs["dataset_schema_version"] = "2.1"
        g_meta.attrs["dataset_version"] = "2.1"
        g_meta.attrs["amplitude_mode"] = "sigma8"

        sim_config = gw.get_simulator_config()
        g_meta.attrs["filaments"] = sim_config["filaments"]
        g_meta.attrs["bias"] = sim_config["bias"]
        g_meta.attrs["ell"] = sim_config["ell"]
        g_meta.attrs["Nhalos"] = sim_config["Nhalos"]

        # Parameter ranges metadata
        g_ranges = f.create_group("metadata/parameter_ranges")
        for k in ['z'] + list(PARAM_KEYS_6D):
            g_ranges.attrs[k] = bounds[k]

        cfg_str = json.dumps({k: params[k].tolist() for k in params}) + str(seed) + str(nsamples_per_point)
        g_meta.attrs["config_hash"] = hashlib.md5(cfg_str.encode()).hexdigest()

    print(f"Simulation for {split_name} completed in {sim_time:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=100)
    parser.add_argument("--nsamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="datasets")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--log_z", action="store_true", help="Sample z log-uniformly instead of uniformly (denser low-z coverage).")
    parser.add_argument("--z_min", type=float, default=0.01, help="Lower z bound for sampling.")
    parser.add_argument("--z_max", type=float, default=10.0, help="Upper z bound for sampling.")
    args = parser.parse_args()

    if args.dataset_dir is not None:
        args.output_dir = args.dataset_dir

    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)

    # LHS over the wide 6d box (Om, sigma8 extend past the ID box for OoD corners).
    bounds = {
        'h': WIDE_6D['h'],
        'OmegaM': WIDE_6D['Om'],
        'sigma8': WIDE_6D['sigma8'],
        'OmegaB': WIDE_6D['Ob'],
        'ns': WIDE_6D['ns'],
        'zeq': WIDE_6D['zeq'],
        'z': (args.z_min, args.z_max)
    }

    keys = ['h', 'OmegaM', 'sigma8', 'OmegaB', 'ns', 'zeq', 'z']

    # Generate all LHS points
    # Need to generate enough points so splits get populated (scale by 4 due to filtering)
    total_points = args.num_points * 4

    # Check if seed parameter is supported (pyDOE3)
    try:
        lhs_samples = lhs(len(keys), samples=total_points, seed=args.seed)
    except TypeError:
        # Fallback if pyDOE version doesn't support seed keyword
        lhs_samples = lhs(len(keys), samples=total_points)

    params = {}
    for i, k in enumerate(keys):
        low, high = bounds[k]
        if k == "z" and args.log_z:
            # Log-uniform z: equal density per decade so the steep low-z regime
            # is resolved as well as the slow high-z regime.
            params[k] = np.exp(np.log(low) + lhs_samples[:, i] * (np.log(high) - np.log(low)))
        else:
            params[k] = low + lhs_samples[:, i] * (high - low)


    train_mask, val_mask, test_mask, split_types = partition_samples(params, total_points, args.seed)

    splits = {
        'train': train_mask,
        'validation': val_mask,
        'test': test_mask
    }

    for split_name, mask in splits.items():
        if np.sum(mask) == 0:
            continue
        split_params = {k: v[mask] for k, v in params.items()}
        split_types_filtered = [split_types[idx] for idx in np.where(mask)[0]]
        out_file = os.path.join(args.output_dir, split_name, f"dataset_{split_name}.h5")
        
        split_seed = None if args.seed is None else args.seed + SPLIT_SEED_OFFSETS[split_name]
        generate_dataset_split(split_params, split_name, out_file, args.nsamples, split_seed, args, split_types_filtered, bounds)


if __name__ == "__main__":
    main()
