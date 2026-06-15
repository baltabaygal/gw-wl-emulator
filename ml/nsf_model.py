import os
# Workaround for macOS duplicate OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import zuko

class ConditionalNSF(nn.Module):
    """Conditional Neural Spline Flow emulator for 1D density estimation."""
    def __init__(
        self,
        input_dim: int = 1,
        context_dim: int = 4,
        num_transforms: int = 6,
        hidden_features: int = 128,
        bins: int = 8
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        
        # Instantiate Zuko's NSF flow
        self.flow = zuko.flows.NSF(
            features=input_dim,
            context=context_dim,
            transforms=num_transforms,
            hidden_features=[hidden_features] * 2,
            bins=bins
        )
        
        # Register standardization stats as buffers for auto-device management and serialization
        self.register_buffer("context_mean", torch.zeros(context_dim))
        self.register_buffer("context_std", torch.ones(context_dim))
        self.register_buffer("lnmu_mean", torch.zeros(1))
        self.register_buffer("lnmu_std", torch.ones(1))

    def set_preprocessing_stats(self, context_mean, context_std, lnmu_mean, lnmu_std):
        """Sets the standardization statistics as buffers."""
        device = self.context_mean.device
        self.register_buffer("context_mean", torch.tensor(context_mean, dtype=torch.float32, device=device))
        self.register_buffer("context_std", torch.tensor(context_std, dtype=torch.float32, device=device))
        self.register_buffer("lnmu_mean", torch.tensor(lnmu_mean, dtype=torch.float32, device=device))
        self.register_buffer("lnmu_std", torch.tensor(lnmu_std, dtype=torch.float32, device=device))

    def log_prob(self, x, context) -> torch.Tensor | np.ndarray:
        """Evaluates the log probability of raw input x given raw context.
        
        Supports both numpy arrays and torch tensors. Returns matching type.
        """
        is_numpy = isinstance(x, np.ndarray) or isinstance(context, np.ndarray)
        
        # Convert to torch tensor if numpy
        if isinstance(x, np.ndarray):
            x_t = torch.tensor(x, dtype=torch.float32, device=self.context_mean.device)
        else:
            x_t = x.to(self.context_mean.device)
            
        if isinstance(context, np.ndarray):
            context_t = torch.tensor(context, dtype=torch.float32, device=self.context_mean.device)
        else:
            context_t = context.to(self.context_mean.device)
            
        # Standardize inputs
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(-1)
        if context_t.ndim == 1:
            context_t = context_t.unsqueeze(0)
            
        # If context is (1, 4) and x is (N, 1), repeat context to match x
        if context_t.shape[0] == 1 and x_t.shape[0] > 1:
            context_t = context_t.repeat(x_t.shape[0], 1)
            
        x_norm = (x_t - self.lnmu_mean) / self.lnmu_std
        context_norm = (context_t - self.context_mean) / self.context_std
        
        # Evaluate log probability under flow
        log_prob_norm = self.flow(context_norm).log_prob(x_norm)
        
        # Apply Jacobian correction for normalization: ln p(x) = ln p(x_norm) - ln(std_lnmu)
        log_prob_raw = log_prob_norm - torch.log(self.lnmu_std)
        
        # Reshape to match the input shape's batch dimension if needed
        if log_prob_raw.ndim > 1 and log_prob_raw.shape[-1] == 1:
            log_prob_raw = log_prob_raw.squeeze(-1)
            
        if is_numpy:
            return log_prob_raw.detach().cpu().numpy()
        return log_prob_raw

    def sample(self, context, nsamples: int) -> torch.Tensor | np.ndarray:
        """Draws samples from the flow given raw context.
        
        Supports both numpy arrays and torch tensors. Returns matching type.
        """
        is_numpy = isinstance(context, np.ndarray)
        
        if isinstance(context, np.ndarray):
            context_t = torch.tensor(context, dtype=torch.float32, device=self.context_mean.device)
        else:
            context_t = context.to(self.context_mean.device)
            
        if context_t.ndim == 1:
            context_t = context_t.unsqueeze(0)
            
        # Standardize context
        context_norm = (context_t - self.context_mean) / self.context_std
        
        # Draw samples from flow: shape (nsamples, batch_size, 1) or (nsamples, 1)
        samples_norm = self.flow(context_norm).sample((nsamples,))
        
        # Un-standardize samples
        samples_raw = samples_norm * self.lnmu_std + self.lnmu_mean
        
        # Squeeze unnecessary dimensions
        # If context has shape (1, 4), samples_raw has shape (nsamples, 1, 1), squeeze to (nsamples, 1)
        if samples_raw.ndim == 3 and samples_raw.shape[1] == 1:
            samples_raw = samples_raw.squeeze(1)
            
        if is_numpy:
            return samples_raw.detach().cpu().numpy()
        return samples_raw

    def density_grid(self, context, x_grid) -> torch.Tensor | np.ndarray:
        """Evaluates density (not log prob) on a grid of physical x values for a single context."""
        is_numpy = isinstance(x_grid, np.ndarray)
        
        if isinstance(x_grid, np.ndarray):
            x_grid_t = torch.tensor(x_grid, dtype=torch.float32, device=self.context_mean.device)
        else:
            x_grid_t = x_grid.to(self.context_mean.device)
            
        if isinstance(context, np.ndarray):
            context_t = torch.tensor(context, dtype=torch.float32, device=self.context_mean.device)
        else:
            context_t = context.to(self.context_mean.device)
            
        grid_size = len(x_grid_t)
        
        # Ensure context is shape (grid_size, context_dim)
        if context_t.ndim == 1:
            context_t = context_t.unsqueeze(0)
        if context_t.shape[0] == 1:
            context_t = context_t.repeat(grid_size, 1)
            
        # Compute log probs and take exp
        log_probs = self.log_prob(x_grid_t, context_t)
        density = torch.exp(log_probs)
        
        if is_numpy:
            return density.detach().cpu().numpy()
        return density

    def save_checkpoint(self, path: str):
        """Saves model weights and preprocessing stats."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "context_mean": self.context_mean.cpu().numpy().tolist(),
            "context_std": self.context_std.cpu().numpy().tolist(),
            "lnmu_mean": float(self.lnmu_mean.cpu().item()),
            "lnmu_std": float(self.lnmu_std.cpu().item())
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads model weights and preprocessing stats."""
        checkpoint = torch.load(path, map_location=self.context_mean.device)
        self.set_preprocessing_stats(
            checkpoint["context_mean"],
            checkpoint["context_std"],
            checkpoint["lnmu_mean"],
            checkpoint["lnmu_std"]
        )
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {path}")
