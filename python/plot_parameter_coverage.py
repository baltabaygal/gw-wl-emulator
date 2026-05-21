import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_parameter_coverage(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'validation', 'test']
    colors = {'train': 'blue', 'validation': 'green', 'test': 'red'}
    markers = {'train': 'o', 'validation': 's', 'test': '^'}

    params_data = {s: {} for s in splits}

    for split in splits:
        path = os.path.join(dataset_dir, split, f"dataset_{split}.h5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, 'r') as f:
            for k in ['h', 'OmegaM', 'sigma8', 'z']:
                params_data[split][k] = f['samples'][k][:]

    # Pairwise plots
    keys = ['h', 'OmegaM', 'sigma8', 'z']
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Histogram
                for split in splits:
                    if keys[i] in params_data[split]:
                        ax.hist(params_data[split][keys[i]], bins=15, alpha=0.5,
                                color=colors[split], label=split)
                if i == 0 and j == 0:
                    ax.legend()
            else:
                # Scatter
                for split in splits:
                    if keys[i] in params_data[split] and keys[j] in params_data[split]:
                        ax.scatter(params_data[split][keys[j]], params_data[split][keys[i]],
                                   c=colors[split], marker=markers[split], alpha=0.6, s=20)

            if j == 0:
                ax.set_ylabel(keys[i])
            if i == n - 1:
                ax.set_xlabel(keys[j])

    plt.suptitle("Parameter Space Coverage (Interpolation Splits)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_coverage.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="ml/plots")
    args = parser.parse_args()
    plot_parameter_coverage(args.dataset_dir, args.output_dir)
    print(f"Parameter coverage plot saved to {args.output_dir}")
