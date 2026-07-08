import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ml.data import load_histogram_dataset, get_recommended_bin_edges

def main() -> None:
    parser = argparse.ArgumentParser(description="Perform PCA representation analysis on weak-lensing PDFs.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/medium")
    parser.add_argument("--output_dir", type=str, default="plots/figures/phase2_representation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    bin_edges = get_recommended_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_histogram_dataset(args.dataset_dir, bin_edges)
    
    # Pool all available ID configurations to have a rich matrix
    # ID configurations have split_type == 'train' or 'interpolation'
    all_pdfs = []
    configs = []
    
    for split_name, (X, Y, split_types) in dataset.items():
        for i in range(X.shape[0]):
            if split_types[i] in ["train", "interpolation"]:
                all_pdfs.append(Y[i])
                configs.append((X[i], Y[i], split_name, split_types[i]))
                
    Y_matrix = np.array(all_pdfs) # shape (C, 100)
    print(f"Constructed PDF matrix with shape: {Y_matrix.shape}")
    
    if Y_matrix.shape[0] == 0:
        raise SystemExit("No In-Distribution configurations found to perform PCA.")

    # 2. Perform PCA using SVD
    C = Y_matrix.shape[0]
    mean_pdf = np.mean(Y_matrix, axis=0)
    Y_centered = Y_matrix - mean_pdf

    # SVD: Y_centered = U * S * V^T
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Variance explained
    eigenvalues = (S ** 2) / (C - 1) if C > 1 else S ** 2
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("\n--- PCA Explained Variance Summary ---")
    print(f"Eigenvalues: {list(np.round(eigenvalues[:10], 6))}")
    print(f"Explained Variance Ratio: {list(np.round(explained_variance_ratio[:10], 6))}")
    print(f"Cumulative Variance: {list(np.round(cumulative_variance[:10], 6))}")

    # Determine required PCs
    thresholds = [0.90, 0.95, 0.99, 0.999]
    required_pcs = {}
    for th in thresholds:
        idx = np.where(cumulative_variance >= th)[0]
        required_pcs[th] = int(idx[0] + 1) if len(idx) > 0 else len(cumulative_variance)
        print(f"Number of PCs required for {th*100:.1f}% explained variance: {required_pcs[th]}")

    # 3. Visualize first 5 principal components
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(5, Vt.shape[0])):
        ax.plot(bin_centers, Vt[i], label=f"PC {i+1} ({explained_variance_ratio[i]*100:.2f}%)", lw=2)
    ax.set_xlabel(r"$\ln\mu$")
    ax.set_ylabel("Component Weight (Eigenvector value)")
    ax.set_title("First 5 Principal Components of lensing PDF family")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "pca_components.png"), dpi=200)
    plt.close(fig)

    # 4. Reconstruction examples
    # Select 3 representative configurations (low, mid, high redshift)
    # Sort configs by redshift
    configs_sorted_z = sorted(configs, key=lambda x: x[0][0])
    sel_configs = [
        configs_sorted_z[0],
        configs_sorted_z[len(configs_sorted_z)//2],
        configs_sorted_z[-1]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    pc_selections = [2, 5, 10]
    colors_recon = {2: '#ff7f0e', 5: '#2ca02c', 10: '#d62728'}
    
    for idx, (x_params, y_true, split_name, stype) in enumerate(sel_configs):
        ax_obj = axes[idx]
        
        # True PDF density
        y_true_density = y_true / bin_widths
        x_steps = np.repeat(bin_edges, 2)[1:-1]
        y_true_steps = np.repeat(y_true_density, 2)
        
        ax_obj.plot(x_steps, y_true_steps, color='black', lw=2.5, label='True PDF')
        ax_obj.fill_between(x_steps, y_true_steps, color='black', alpha=0.1)

        # Centered target
        y_centered = y_true - mean_pdf
        
        for n_comp in pc_selections:
            # Project: scores = y_centered * V
            # Vt is shape (K, 100), V is shape (100, K)
            V_sub = Vt[:n_comp].T # shape (100, n_comp)
            scores = np.dot(y_centered, V_sub)
            
            # Reconstruct: y_recon = scores * Vt[:n_comp] + mean
            y_recon = np.dot(scores, Vt[:n_comp]) + mean_pdf
            
            # Ensure probabilities sum to 1.0 (some PCA reconstructions might have negative values, we normalize)
            y_recon = np.maximum(y_recon, 0.0)
            total_sum = np.sum(y_recon)
            if total_sum > 0:
                y_recon = y_recon / total_sum
            
            y_recon_density = y_recon / bin_widths
            y_recon_steps = np.repeat(y_recon_density, 2)
            ax_obj.plot(x_steps, y_recon_steps, color=colors_recon[n_comp], lw=1.8, linestyle='--',
                        label=f'{n_comp} PCs')
            
        ax_obj.set_xlabel(r"$\ln\mu$")
        ax_obj.set_ylabel("Probability Density")
        ax_obj.set_xlim(-0.35, 1.2)
        ax_obj.set_title(f"Reconstruction at z={x_params[0]:.2f}\n($\\sigma_8$={x_params[3]:.2f}, $\\Omega_M$={x_params[2]:.2f})")
        ax_obj.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "pca_reconstruction.png"), dpi=200)
    plt.close(fig)
    print("PCA components and reconstruction plots generated successfully!")

if __name__ == "__main__":
    main()
