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
