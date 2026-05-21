import sys
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_ood_regions(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train', 'validation', 'test']

    data = {}
    for s in splits:
        path = os.path.join(dataset_dir, s, f"dataset_{s}.h5")
        if os.path.exists(path):
            with h5py.File(path, 'r') as f:
                data[s] = {
                    'sigma8': f['samples/sigma8'][:],
                    'OmegaM': f['samples/OmegaM'][:]
                }

    if not data:
        print("No datasets found to plot OoD regions.")
        return

    plt.figure(figsize=(8, 6))

    if 'train' in data:
        plt.scatter(data['train']['OmegaM'], data['train']['sigma8'],
                    c='blue', label='Train (In-Distribution)', alpha=0.5, marker='o')

    if 'validation' in data:
        plt.scatter(data['validation']['OmegaM'], data['validation']['sigma8'],
                    c='green', label='Validation (Interpolation/OoD)', alpha=0.7, marker='s')

    if 'test' in data:
        plt.scatter(data['test']['OmegaM'], data['test']['sigma8'],
                    c='red', label='Test (Interpolation/OoD)', alpha=0.7, marker='^')

    plt.xlabel('OmegaM')
    plt.ylabel('sigma8')
    plt.title('Out-of-Distribution / Interpolation Regions')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ood_regions.png'), dpi=300)
    plt.close()

    print(f"OoD visualization saved to {output_dir}/ood_regions.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="ml/plots")
    args = parser.parse_args()
    plot_ood_regions(args.dataset_dir, args.output_dir)
