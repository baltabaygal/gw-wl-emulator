import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

def wasserstein_1d(u_samples, v_samples):
    u_sorted = np.sort(u_samples)
    v_sorted = np.sort(v_samples)
    if len(u_sorted) == 0 or len(v_sorted) == 0:
        return np.nan

    # interpolate to same size if different
    if len(u_sorted) != len(v_sorted):
        n = min(len(u_sorted), len(v_sorted))
        u_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
        v_sorted = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)

    return np.mean(np.abs(u_sorted - v_sorted))

def run_smoothness_diagnostics(output_dir):
    nsamples = 10000
    seed = 42

    # Fixed base parameters
    z_base = 1.0
    h_base = 0.674
    OmegaM_base = 0.315
    sigma8_base = 0.811

    # Vary sigma8
    sigma8_range = np.linspace(0.6, 1.0, 10)
    samples_list = []

    print("Running Smoothness Diagnostics (varying sigma8)...")
    for s8 in sigma8_range:
        lnmu = gw.sample_lnmu_ml(z_base, h_base, OmegaM_base, s8, nsamples, seed)
        samples_list.append(lnmu[~np.isnan(lnmu)])

    distances = []
    for i in range(len(sigma8_range) - 1):
        dist = wasserstein_1d(samples_list[i], samples_list[i+1])
        distances.append(dist)

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(sigma8_range[:-1], distances, marker='o')
    plt.xlabel('sigma8')
    plt.ylabel('Wasserstein Distance to Next Step')
    plt.title('Parameter Smoothness (sigma8)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_sigma8.png'), dpi=300)
    plt.close()

    print(f"Smoothness plot saved to {output_dir}/smoothness_sigma8.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="ml/plots")
    args = parser.parse_args()
    run_smoothness_diagnostics(args.output_dir)
