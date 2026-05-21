import sys
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_lnmu_histograms(dataset_path, output_dir):
    with h5py.File(dataset_path, 'r') as f:
        lnmu = f['samples/lnmu'][:]
        z = f['samples/z'][:]

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot histogram for a few redshift slices
    num_plots = min(5, len(z))
    indices = np.linspace(0, len(z)-1, num_plots, dtype=int)

    for idx in indices:
        samples = lnmu[idx]
        samples = samples[~np.isnan(samples)]

        if len(samples) > 0:
            plt.hist(samples, bins=50, density=True, histtype='step',
                     label=f"z = {z[idx]:.2f}", lw=2)

    plt.xlabel(r'$\ln(\mu)$', fontsize=14)
    plt.ylabel('PDF', fontsize=14)
    plt.title('ln(mu) Distribution', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lnmu_histograms.png'), dpi=300)
    plt.close()

def plot_moments_vs_redshift(dataset_path, output_dir):
    with h5py.File(dataset_path, 'r') as f:
        lnmu = f['samples/lnmu'][:]
        z = f['samples/z'][:]

    means = []
    variances = []
    skewnesses = []
    valid_z = []

    for i in range(len(z)):
        samples = lnmu[i]
        samples = samples[~np.isnan(samples)]
        if len(samples) > 2:
            mean = np.mean(samples)
            var = np.var(samples)

            # manual skewness
            m3 = np.mean((samples - mean)**3)
            skew = m3 / (var**1.5) if var > 0 else 0

            means.append(mean)
            variances.append(var)
            skewnesses.append(skew)
            valid_z.append(z[i])

    # Sort by redshift
    sorted_idx = np.argsort(valid_z)
    valid_z = np.array(valid_z)[sorted_idx]
    means = np.array(means)[sorted_idx]
    variances = np.array(variances)[sorted_idx]
    skewnesses = np.array(skewnesses)[sorted_idx]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].scatter(valid_z, means, s=10, alpha=0.7)
    axes[0].set_ylabel('Mean ln(mu)')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(valid_z, variances, s=10, alpha=0.7)
    axes[1].set_ylabel('Variance')
    axes[1].grid(alpha=0.3)

    axes[2].scatter(valid_z, skewnesses, s=10, alpha=0.7)
    axes[2].set_ylabel('Skewness')
    axes[2].set_xlabel('Redshift (z)')
    axes[2].grid(alpha=0.3)

    plt.suptitle('Moments vs Redshift', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moments_vs_redshift.png'), dpi=300)
    plt.close()

def plot_tail_diagnostics(dataset_path, output_dir):
    with h5py.File(dataset_path, 'r') as f:
        lnmu = f['samples/lnmu'][:]
        z = f['samples/z'][:]

    plt.figure(figsize=(10, 6))

    num_plots = min(5, len(z))
    indices = np.linspace(0, len(z)-1, num_plots, dtype=int)

    for idx in indices:
        samples = lnmu[idx]
        samples = samples[~np.isnan(samples)]

        if len(samples) > 0:
            # Sort samples and compute CDF
            sorted_samples = np.sort(samples)
            cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

            # Plot 1-CDF (survival function) on log scale to see heavy tails
            plt.plot(sorted_samples, 1 - cdf, label=f"z = {z[idx]:.2f}")

    plt.yscale('log')
    plt.xlabel(r'$\ln(\mu)$', fontsize=14)
    plt.ylabel('1 - CDF (Survival Function)', fontsize=14)
    plt.title('Tail Diagnostics', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tail_diagnostics.png'), dpi=300)
    plt.close()

def plot_sample_counts(dataset_path, output_dir):
    with h5py.File(dataset_path, 'r') as f:
        lnmu = f['samples/lnmu'][:]
        z = f['samples/z'][:]

    counts = [np.sum(~np.isnan(row)) for row in lnmu]

    plt.figure(figsize=(8, 5))
    plt.scatter(z, counts, alpha=0.7)
    plt.xlabel('Redshift (z)')
    plt.ylabel('Valid Samples Count')
    plt.title('Sample Count Diagnostics')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_counts.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Validation plotting utilities.")
    parser.add_argument("--dataset", type=str, default="datasets/train/dataset_train.h5", help="Path to HDF5 dataset.")
    parser.add_argument("--output_dir", type=str, default="ml/plots", help="Directory to save plots.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.dataset):
        print(f"Dataset {args.dataset} not found. Please run generate_dataset.py first.")
        sys.exit(1)

    print(f"Generating validation plots for {args.dataset}...")
    plot_lnmu_histograms(args.dataset, args.output_dir)
    plot_moments_vs_redshift(args.dataset, args.output_dir)
    plot_tail_diagnostics(args.dataset, args.output_dir)
    plot_sample_counts(args.dataset, args.output_dir)
    print(f"Plots saved to {args.output_dir}.")

if __name__ == "__main__":
    main()
