import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))
import gwlensing as gw

def compute_stats(samples):
    samples = samples[~np.isnan(samples)]
    if len(samples) < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean = np.mean(samples)
    var = np.var(samples)
    std = np.sqrt(var)

    # Skewness
    m3 = np.mean((samples - mean)**3)
    skew = m3 / (std**3) if std > 0 else 0

    # Kurtosis
    m4 = np.mean((samples - mean)**4)
    kurt = m4 / (std**4) - 3 if std > 0 else 0

    # Quantiles
    q1 = np.percentile(samples, 1)
    q99 = np.percentile(samples, 99)

    return mean, var, skew, kurt, q1, q99

def run_mc_diagnostics(output_dir):
    nsamples_list = [1000, 5000, 10000, 50000]
    num_seeds = 10

    # Fixed cosmology
    z = 1.0
    h = 0.674
    OmegaM = 0.315
    sigma8 = 0.811

    results = {n: {'mean': [], 'var': [], 'skew': [], 'kurt': [], 'q1': [], 'q99': []} for n in nsamples_list}

    print("Running Monte Carlo Diagnostics...")
    for n in nsamples_list:
        print(f"Testing nsamples = {n}...")
        for i in range(num_seeds):
            seed = 123 + i
            lnmu = gw.sample_lnmu_ml(z, h, OmegaM, sigma8, n, seed)
            stats = compute_stats(lnmu)

            results[n]['mean'].append(stats[0])
            results[n]['var'].append(stats[1])
            results[n]['skew'].append(stats[2])
            results[n]['kurt'].append(stats[3])
            results[n]['q1'].append(stats[4])
            results[n]['q99'].append(stats[5])

    os.makedirs(output_dir, exist_ok=True)

    # Plotting
    metrics = ['mean', 'var', 'skew', 'kurt', 'q1', 'q99']
    titles = ['Mean', 'Variance', 'Skewness', 'Kurtosis', '1st Percentile', '99th Percentile']

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        means = [np.mean(results[n][metric]) for n in nsamples_list]
        stds = [np.std(results[n][metric]) for n in nsamples_list]

        ax.errorbar(nsamples_list, means, yerr=stds, fmt='-o', capsize=5)
        ax.set_xscale('log')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel(titles[i])
        ax.set_title(f'Convergence of {titles[i]}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mc_convergence.png'), dpi=300)
    plt.close()

    # Estimate minimum viable samples based on variance of kurtosis (sensitive to tails)
    kurt_stds = [np.std(results[n]['kurt']) for n in nsamples_list]
    print("\nConvergence Summary (Kurtosis Std Dev):")
    for n, std in zip(nsamples_list, kurt_stds):
        print(f"  {n} samples: std = {std:.4f}")

    print(f"\nPlots saved to {output_dir}/mc_convergence.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="ml/plots")
    args = parser.parse_args()

    run_mc_diagnostics(args.output_dir)
