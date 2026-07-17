import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from ml.nsf_dataset import get_nsf_dataloaders
from ml.nsf_model import ConditionalNSF

def main():
    parser = argparse.ArgumentParser(description="Train Conditional NSF Weak Lensing Emulator.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/backend_current_1k")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--num_transforms", type=int, default=6)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--output_model", type=str, default="data/models/conditional_nsf_backend_current.pt")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase2b_nsf_training")
    parser.add_argument("--output_report", type=str, default="docs/phase2/phase2b_nsf_training_report.md")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)

    device = torch.device(args.device)
    print(f"Training on device: {device}")

    # 1. Load data loaders and standardization stats
    print(f"Loading data from {args.dataset_dir}...")
    loaders, stats = get_nsf_dataloaders(args.dataset_dir, batch_size=args.batch_size)
    
    train_loader = loaders["train"]
    val_loader = loaders.get("validation", None)
    
    print(f"Dataset loaded. Train steps per epoch: {len(train_loader)}")
    if val_loader:
        print(f"Val steps per epoch: {len(val_loader)}")

    # 2. Initialize Model (context dim follows the dataset: 7 for 1+6d, 4 legacy)
    model = ConditionalNSF(
        input_dim=1,
        context_dim=len(np.asarray(stats["context_mean"]).ravel()),
        num_transforms=args.num_transforms,
        hidden_features=args.hidden_features,
        bins=args.bins
    ).to(device)
    
    # Store standardization stats inside the model buffers
    model.set_preprocessing_stats(
        stats["context_mean"],
        stats["context_std"],
        stats["lnmu_mean"],
        stats["lnmu_std"]
    )

    # 3. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training history
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    print("Beginning training...")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Zuko log_prob: flow(context).log_prob(y)
            log_prob = model.flow(x_batch).log_prob(y_batch)
            loss = -log_prob.mean()
            
            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss encountered at epoch {epoch}")
                
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
            optimizer.step()
            
            train_loss_sum += loss.item() * x_batch.shape[0]
            train_samples += x_batch.shape[0]
            
        epoch_train_loss = train_loss_sum / train_samples
        history["train_loss"].append(epoch_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_loss_sum = 0.0
            val_samples = 0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    log_prob_val = model.flow(x_val).log_prob(y_val)
                    loss_val = -log_prob_val.mean()
                    val_loss_sum += loss_val.item() * x_val.shape[0]
                    val_samples += x_val.shape[0]
            epoch_val_loss = val_loss_sum / val_samples
            history["val_loss"].append(epoch_val_loss)
        else:
            epoch_val_loss = 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train NLL: {epoch_train_loss:.6f} | Val NLL: {epoch_val_loss:.6f}")

        # Early stopping logic
        if val_loader:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}. Reloading best model.")
                    break

    t_end = time.time()
    total_time = t_end - t_start
    print(f"Training completed in {total_time:.2f} s")

    # Load best state
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save model checkpoint
    model.save_checkpoint(args.output_model)
    
    # Save training history
    with open(args.output_model.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train NLL", lw=2)
    if val_loader:
        ax.plot(history["val_loss"], label="Val NLL", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative Log Likelihood")
    ax.set_title("Conditional NSF Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.plots_dir, "training_loss.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Training loss plot saved to {plot_path}")

    # Generate training report
    report_md = f"""# Phase 2B — NSF Training Diagnostic Report

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

## 1. Model Configuration
- Flow Type: Rational Quadratic Neural Spline Flow (Zuko NSF)
- Transforms: {args.num_transforms}
- Hidden Features: {args.hidden_features} x 2
- Spline Bins: {args.bins}
- Input Dimension (lnmu): 1
- Context Dimension (z, h, OmegaM, sigma8): 4

## 2. Preprocessing Statistics
- Context Mean: `{stats["context_mean"]}`
- Context Std: `{stats["context_std"]}`
- lnmu Mean: `{stats["lnmu_mean"]:.6f}`
- lnmu Std: `{stats["lnmu_std"]:.6f}`

## 3. Training Details
- Optimizer: AdamW
- Batch Size: {args.batch_size}
- Initial Learning Rate: {args.lr}
- Weight Decay: {args.weight_decay}
- Epochs Completed: {epoch} (Early stopping patience: {args.patience})
- Total Training Time: {total_time:.2f} seconds
- Best Validation NLL: {best_val_loss:.6f}

## 4. Diagnostics Curves
![Training Loss](figures/phase2b_nsf_training/training_loss.png)
"""
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Training report written to {args.output_report}")

if __name__ == "__main__":
    main()
