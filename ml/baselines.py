from typing import Optional, Tuple

import numpy as np


SCIPY_AVAILABLE = True
try:
    from scipy.stats import gaussian_kde
except Exception:
    SCIPY_AVAILABLE = False
    gaussian_kde = None


def histogram_pdf(samples: np.ndarray, bins: int = 100, density: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Histogram baseline PDF estimate.

    Returns:
      centers, pdf_values
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.array([]), np.array([])

    hist, edges = np.histogram(samples, bins=bins, density=density)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist


def gaussian_kde_pdf(
    samples: np.ndarray,
    grid: Optional[np.ndarray] = None,
    points: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Gaussian KDE baseline. Gracefully disabled when scipy is unavailable."""
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return np.array([]), np.array([])

    if not SCIPY_AVAILABLE:
        return np.array([]), None

    if grid is None:
        lo = float(np.min(samples))
        hi = float(np.max(samples))
        if lo == hi:
            lo -= 1e-6
            hi += 1e-6
        grid = np.linspace(lo, hi, points)

    kde = gaussian_kde(samples)
    pdf = kde(grid)
    return grid, pdf


# PyTorch MLP Emulator and Target Metrics
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

if torch is not None:
    class BaselineMLP(nn.Module):
        """Baseline Multi-Layer Perceptron emulator for binned magnification probabilities.

        Inputs:
          z, h, OmegaM, sigma8 (4 dimensions)
        Outputs:
          Probability vector of length 100 (sums to 1.0)
        """
        def __init__(self, input_dim: int = 4, hidden_dim: int = 128, output_dim: int = 100):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class BaselineMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available.")


def compute_binned_moments(p: np.ndarray, bin_centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes Mean, Variance, Skewness, and Excess Kurtosis from binned probabilities.

    Assumes p has shape (N, Bins) or (Bins,) and bin_centers has shape (Bins,).
    """
    p = np.asarray(p)
    bin_centers = np.asarray(bin_centers)
    is_1d = (p.ndim == 1)
    if is_1d:
        p = np.expand_dims(p, axis=0)

    # Mean
    mean = np.sum(p * bin_centers, axis=-1)

    # Variance
    mean_expanded = np.expand_dims(mean, axis=-1)
    var = np.sum(p * (bin_centers - mean_expanded)**2, axis=-1)
    std = np.sqrt(np.maximum(var, 1e-12))

    # Skewness
    skew = np.sum(p * (bin_centers - mean_expanded)**3, axis=-1) / (std**3)

    # Kurtosis (excess)
    kurt = np.sum(p * (bin_centers - mean_expanded)**4, axis=-1) / (std**4) - 3.0

    if is_1d:
        return mean[0], var[0], skew[0], kurt[0]
    return mean, var, skew, kurt


def compute_characteristic_function(p: np.ndarray, bin_centers: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Computes binned characteristic function P_hat(k) = sum p_i e^{i k x_i}.

    p: shape (N, Bins) or (Bins,)
    bin_centers: shape (Bins,)
    k_grid: shape (K_vals,)

    Returns:
      complex array of shape (N, K_vals) or (K_vals,)
    """
    p = np.asarray(p)
    bin_centers = np.asarray(bin_centers)
    k_grid = np.asarray(k_grid)
    is_1d = (p.ndim == 1)
    if is_1d:
        p = np.expand_dims(p, axis=0)

    # exponent shape: (K_vals, Bins)
    exponent = 1j * np.outer(k_grid, bin_centers)
    exp_factor = np.exp(exponent) # (K_vals, Bins)

    # Result shape: (N, K_vals)
    res = np.matmul(p, exp_factor.T)

    if is_1d:
        return res[0]
    return res


def compute_kl_divergence(p_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes KL divergence sum(p_true * log(p_true / p_pred)).

    p_true, p_pred: shape (N, Bins) or (Bins,)
    """
    p_true = np.asarray(p_true)
    p_pred = np.asarray(p_pred)
    is_1d = (p_true.ndim == 1)
    if is_1d:
        p_true = np.expand_dims(p_true, axis=0)
        p_pred = np.expand_dims(p_pred, axis=0)

    kl = np.sum(p_true * np.log((p_true + eps) / (p_pred + eps)), axis=-1)

    if is_1d:
        return kl[0]
    return kl


def compute_js_divergence(p_true: np.ndarray, p_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes Jensen-Shannon divergence between two binned distributions.

    p_true, p_pred: shape (N, Bins) or (Bins,)
    """
    p_true = np.asarray(p_true)
    p_pred = np.asarray(p_pred)
    is_1d = (p_true.ndim == 1)
    if is_1d:
        p_true = np.expand_dims(p_true, axis=0)
        p_pred = np.expand_dims(p_pred, axis=0)

    m = 0.5 * (p_true + p_pred)
    kl_true_m = p_true * np.log((p_true + eps) / (m + eps))
    kl_pred_m = p_pred * np.log((p_pred + eps) / (m + eps))
    jsd = 0.5 * np.sum(kl_true_m, axis=-1) + 0.5 * np.sum(kl_pred_m, axis=-1)

    if is_1d:
        return jsd[0]
    return jsd


def compute_wasserstein_distance(p_true: np.ndarray, p_pred: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Computes 1D Earth Mover's Distance (Wasserstein distance) using the CDF.

    p_true, p_pred: shape (N, Bins) or (Bins,)
    bin_edges: shape (Bins + 1,)
    """
    p_true = np.asarray(p_true)
    p_pred = np.asarray(p_pred)
    bin_edges = np.asarray(bin_edges)
    is_1d = (p_true.ndim == 1)
    if is_1d:
        p_true = np.expand_dims(p_true, axis=0)
        p_pred = np.expand_dims(p_pred, axis=0)

    # Cumulative distribution: shape (N, Bins)
    cdf_true = np.cumsum(p_true, axis=-1)
    cdf_pred = np.cumsum(p_pred, axis=-1)

    # Integrand: diff in CDFs up to N-1
    cdf_diff = np.abs(cdf_true[:, :-1] - cdf_pred[:, :-1]) # (N, Bins-1)
    widths = np.diff(bin_edges)[:-1] # (Bins-1,)

    w1 = np.sum(cdf_diff * widths, axis=-1)

    if is_1d:
        return w1[0]
    return w1
