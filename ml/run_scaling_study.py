import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ml.data import load_histogram_dataset, get_recommended_bin_edges
from ml.baselines import (
    BaselineMLP,
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance
)

def train_model(X_train, Y_train, X_val, Y_val, epochs=500, lr=1e-3, batch_size=32, patience=30):
    # Standardize inputs
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    train_ds = TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = BaselineMLP(input_dim=4, hidden_dim=128, output_dim=100)
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
    parser = argparse.ArgumentParser(description="Run scaling study on MLP emulator.")
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
    
    print(f"Total pooled In-Distribution configs: {all_x.shape[0]}")
    
    # Shuffle deterministically
    rng = np.random.default_rng(42)
    indices = np.arange(all_x.shape[0])
    rng.shuffle(indices)
    
    all_x = all_x[indices]
    all_y = all_y[indices]
    
    # Hold out 50 configurations for validation
    val_size = min(50, all_x.shape[0] // 4)
    X_val = all_x[:val_size]
    Y_val = all_y[:val_size]
    
    X_pool = all_x[val_size:]
    Y_pool = all_y[val_size:]
    
    print(f"Validation size: {X_val.shape[0]}")
    print(f"Training pool size: {X_pool.shape[0]}")

    # Sizes to evaluate
    eval_sizes = [25, 50, 100, 250, 500]
    eval_sizes = [s for s in eval_sizes if s <= X_pool.shape[0]]
    if X_pool.shape[0] not in eval_sizes:
        eval_sizes.append(X_pool.shape[0])
        
    results = {}
    
    for N in eval_sizes:
        print(f"\n--- Training baseline MLP with N={N} configurations ---")
        X_train = X_pool[:N]
        Y_train = Y_pool[:N]
        
        preds = train_model(X_train, Y_train, X_val, Y_val)
        
        # Calculate metrics
        kls = compute_kl_divergence(Y_val, preds)
        jsds = compute_js_divergence(Y_val, preds)
        wasses = compute_wasserstein_distance(Y_val, preds, bin_edges)
        
        results[N] = {
            'kl': np.mean(kls),
            'jsd': np.mean(jsds),
            'was': np.mean(wasses)
        }
        print(f"Results N={N} | KL: {results[N]['kl']:.6f} | JSD: {results[N]['jsd']:.6f} | Wasserstein: {results[N]['was']:.6f}")

    # Generate scaling study plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    sizes = sorted(results.keys())
    jsd_vals = [results[s]['jsd'] for s in sizes]
    was_vals = [results[s]['was'] for s in sizes]
    
    ax1.plot(sizes, jsd_vals, marker='o', color='#1f77b4', lw=2.5, ms=8)
    ax1.set_xlabel("Training Set Size (N configurations)")
    ax1.set_ylabel("Average JSD")
    ax1.set_title("Jensen-Shannon Divergence Scaling")
    ax1.set_xscale('log')
    ax1.set_xticks(sizes)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax2.plot(sizes, was_vals, marker='s', color='#d62728', lw=2.5, ms=8)
    ax2.set_xlabel("Training Set Size (N configurations)")
    ax2.set_ylabel("Average Wasserstein Distance")
    ax2.set_title("Wasserstein Distance Scaling")
    ax2.set_xscale('log')
    ax2.set_xticks(sizes)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "scaling_study.png"), dpi=200)
    plt.close(fig)
    print("Scaling study plots generated successfully!")

if __name__ == "__main__":
    main()
