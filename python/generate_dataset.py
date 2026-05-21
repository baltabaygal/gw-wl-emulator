import sys
import os
import argparse
import time
import subprocess
import h5py
import numpy as np

# Ensure build directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

try:
    from pyDOE3 import lhs
except ImportError:
    try:
        from pyDOE2 import lhs
    except ImportError:
        try:
            from pyDOE import lhs
        except ImportError:
            print("Please install pyDOE3, pyDOE2, or pyDOE for Latin Hypercube Sampling.")
            sys.exit(1)


def get_git_commit():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL)
        return commit.decode('utf-8').strip()
    except Exception:
        return "unknown"

def generate_dataset(num_points, nsamples_per_point, output_file, split_type='train', seed=None):
    # Parameter ranges
    # h in [0.59, 0.76]
    # OmegaM in [0.15, 0.47]
    # sigma8 in [0.4, 1.4]
    # z in [0, 10]

    if seed is not None:
        np.random.seed(seed)


    if split_type == 'train':
        bounds = {
            'h': (0.59, 0.72),
            'OmegaM': (0.15, 0.40),
            'sigma8': (0.4, 1.2),
            'z': (0.01, 8.0)
        }
    elif split_type == 'validation':
        bounds = {
            'h': (0.72, 0.74),
            'OmegaM': (0.40, 0.44),
            'sigma8': (1.2, 1.3),
            'z': (8.0, 9.0)
        }
    else: # test
        bounds = {
            'h': (0.74, 0.76),
            'OmegaM': (0.44, 0.47),
            'sigma8': (1.3, 1.4),
            'z': (9.0, 10.0)
        }


    keys = ['h', 'OmegaM', 'sigma8', 'z']

    print(f"Generating LHS parameters for {num_points} configurations...")
    lhs_samples = lhs(4, samples=num_points)

    params = {}
    for i, k in enumerate(keys):
        low, high = bounds[k]
        params[k] = low + lhs_samples[:, i] * (high - low)

    lnmu_array = np.full((num_points, nsamples_per_point), np.nan)

    print("Simulating ln(mu) samples...")
    start_time = time.time()
    for i in range(num_points):
        if i > 0 and i % 10 == 0:
            print(f"  Processed {i}/{num_points} configurations...")

        z = params['z'][i]
        h = params['h'][i]
        omega_m = params['OmegaM'][i]
        sigma8 = params['sigma8'][i]

        # Determine simulation seed for this config
        sim_seed = None if seed is None else seed + i

        # sample_lnmu_ml(z, h, OmegaM, sigma8, nsamples, seed)
        lnmu = gw.sample_lnmu_ml(
            z,
            h,
            omega_m,
            sigma8,
            nsamples_per_point,
            sim_seed
        )

        lnmu_array[i, :len(lnmu)] = lnmu

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    print(f"Saving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Create dataset structure
        g_samples = f.create_group("samples")
        g_samples.create_dataset("lnmu", data=lnmu_array, compression="gzip")
        g_samples.create_dataset("z", data=params['z'], compression="gzip")
        g_samples.create_dataset("h", data=params['h'], compression="gzip")
        g_samples.create_dataset("OmegaM", data=params['OmegaM'], compression="gzip")
        g_samples.create_dataset("sigma8", data=params['sigma8'], compression="gzip")

        # Store metadata
        g_meta = f.create_group("metadata")
        g_meta.attrs["seed"] = seed if seed is not None else "stochastic"
        g_meta.attrs["git_commit"] = get_git_commit()
        g_meta.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        g_meta.attrs["nsamples_per_point"] = nsamples_per_point

        # physics toggles from sample_lnmu_ml defaults
        g_meta.attrs["filaments"] = True
        g_meta.attrs["bias"] = True
        g_meta.attrs["ell"] = True
        g_meta.attrs["Nhalos"] = 100

def main():
    parser = argparse.ArgumentParser(description="Generate weak-lensing dataset.")
    parser.add_argument("--num_points", type=int, default=100, help="Number of parameter configurations per split.")
    parser.add_argument("--nsamples", type=int, default=50000, help="Number of ln(mu) samples per configuration.")
    parser.add_argument("--seed", type=int, default=None, help="Global seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Output directory.")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    # Deterministic dataset splits based on seed if provided
    splits = [
        ('train', args.num_points, 0),
        ('validation', max(1, args.num_points // 5), 100000), # Ensure at least 1 point
        ('test', max(1, args.num_points // 5), 200000)        # Ensure at least 1 point
    ]

    for split_name, n_pts, offset in splits:
        print(f"\n--- Generating {split_name} split ({n_pts} points) ---")
        split_seed = None if args.seed is None else args.seed + offset
        out_file = os.path.join(args.output_dir, split_name, f"dataset_{split_name}.h5")
        generate_dataset(n_pts, args.nsamples, out_file, split_type=split_name, seed=split_seed)

if __name__ == "__main__":
    main()
