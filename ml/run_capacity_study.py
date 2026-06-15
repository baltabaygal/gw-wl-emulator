import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.data import load_histogram_dataset, get_recommended_bin_edges
from ml.baselines import (
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance
)

# Define models dynamically
class CustomMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[128, 128, 128], output_dim=100):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train_custom_model(hidden_dims, X_train, Y_train, X_val, Y_val, epochs=500, lr=1e-3, batch_size=32, patience=30):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = CustomMLP(input_dim=4, hidden_dims=hidden_dims, output_dim=100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    val_x_t = torch.tensor(X_val_norm, dtype=torch.float32)
    val_y_t = torch.tensor(Y_val, dtype=torch.float32)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(x_b)
            loss = loss_fn(torch.log(pred + 1e-12), y_b)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            pred_val = model(val_x_t)
            val_loss = loss_fn(torch.log(pred_val + 1e-12), val_y_t).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    model.eval()
    with torch.no_grad():
        preds = model(val_x_t).numpy()
    return preds

def main() -> None:
    parser = argparse.ArgumentParser(description="Run capacity study on MLP emulator.")
    parser.add_argument("--dataset_dir", type=str, default="datasets_large")
    parser.add_argument("--output_dir", type=str, default="plots/figures/phase2_representation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bin_edges = get_recommended_bin_edges()
    
    print(f"Loading dataset from {args.dataset_dir}...")
    dataset = load_histogram_dataset(args.dataset_dir, bin_edges)
    
    # Pool all In-Distribution configurations across splits
    all_x = []
    all_y = []
    
    for split_name, (X, Y, split_types) in dataset.items():
        for i in range(X.shape[0]):
            if split_types[i] in ["train", "interpolation"]:
                all_x.append(X[i])
                all_y.append(Y[i])
                
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Shuffle deterministically
    rng = np.random.default_rng(42)
    indices = np.arange(all_x.shape[0])
    rng.shuffle(indices)
    
    all_x = all_x[indices]
    all_y = all_y[indices]
    
    val_size = min(50, all_x.shape[0] // 4)
    X_val = all_x[:val_size]
    Y_val = all_y[:val_size]
    
    X_pool = all_x[val_size:]
    Y_pool = all_y[val_size:]

    model_sizes = {
        'Small [64,64]': [64, 64],
        'Baseline [128,128,128]': [128, 128, 128],
        'Large [256,256,256,256]': [256, 256, 256, 256]
    }
    
    train_sizes = [25, 100]
    if X_pool.shape[0] > 100:
        train_sizes.append(X_pool.shape[0])
        
    results = {m_name: {} for m_name in model_sizes}
    
    for N in train_sizes:
        print(f"\n=================== Evaluating Training Size N={N} ===================")
        X_train = X_pool[:N]
        Y_train = Y_pool[:N]
        
        for m_name, hidden_dims in model_sizes.items():
            print(f"Training Model: {m_name}...")
            preds = train_custom_model(hidden_dims, X_train, Y_train, X_val, Y_val)
            
            kls = compute_kl_divergence(Y_val, preds)
            jsds = compute_js_divergence(Y_val, preds)
            wasses = compute_wasserstein_distance(Y_val, preds, bin_edges)
            
            results[m_name][N] = {
                'kl': np.mean(kls),
                'jsd': np.mean(jsds),
                'was': np.mean(wasses)
            }
            print(f"  {m_name} | JSD: {results[m_name][N]['jsd']:.6f} | Wasserstein: {results[m_name][N]['was']:.6f}")

    # Generate capacity study plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    x_indices = np.arange(len(train_sizes))
    width = 0.25
    
    model_colors = {
        'Small [64,64]': '#1f77b4',
        'Baseline [128,128,128]': '#2ca02c',
        'Large [256,256,256,256]': '#d62728'
    }
    
    for idx_m, m_name in enumerate(model_sizes):
        jsd_y = [results[m_name][N]['jsd'] for N in train_sizes]
        was_y = [results[m_name][N]['was'] for N in train_sizes]
        
        offset = (idx_m - 1) * width
        ax1.bar(x_indices + offset, jsd_y, width, label=m_name, color=model_colors[m_name], alpha=0.85)
        ax2.bar(x_indices + offset, was_y, width, label=m_name, color=model_colors[m_name], alpha=0.85)
        
    ax1.set_ylabel("Average JSD")
    ax1.set_xlabel("Training Set Size (N)")
    ax1.set_title("JSD Comparison across Architectures")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"N={N}" if N != X_pool.shape[0] else f"Full (N={N})" for N in train_sizes])
    ax1.legend()
    
    ax2.set_ylabel("Average Wasserstein Distance")
    ax2.set_xlabel("Training Set Size (N)")
    ax2.set_title("Wasserstein Distance Comparison across Architectures")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"N={N}" if N != X_pool.shape[0] else f"Full (N={N})" for N in train_sizes])
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "capacity_study.png"), dpi=200)
    plt.close(fig)
    print("Capacity study plots generated successfully!")

if __name__ == "__main__":
    main()
