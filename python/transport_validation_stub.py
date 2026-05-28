"""Initial transport-spectrum validation scaffold.

This module bootstraps the numerical validation program for the
weak-lensing transport framework.

Phase implemented:
- synthetic propagator generation
- FFT characteristic-function extraction
- effective Lambda reconstruction
- roundtrip consistency checks

The goal is to validate the numerics before touching real PDFs.
"""

from __future__ import annotations

import numpy as np


def make_x_grid(xmax: float = 4.0, n: int = 4096):
    dx = 2.0 * xmax / n
    x = (np.arange(n) - n // 2) * dx
    return x, dx


def k_grid(n: int, dx: float):
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def exponential_jump_ft(k, xi0=0.25):
    return 1.0 / (1.0 + 1j * k * xi0)


def lambda_model(k, v=0.02, eps=0.01, rate=0.7, xi0=0.25):
    return -1j * v * k - eps * k**2 + rate * (exponential_jump_ft(k, xi0) - 1.0)


def synthetic_phi(k, z, **kwargs):
    return np.exp(z * lambda_model(k, **kwargs))


def reconstruct_lambda(phi, z, floor=1e-12):
    phi = phi.copy()
    small = np.abs(phi) < floor
    phi[small] = floor * np.exp(1j * np.angle(phi[small]))
    return np.log(phi) / z


def synthetic_roundtrip(z=1.0, xmax=4.0, n=4096):
    x, dx = make_x_grid(xmax=xmax, n=n)
    k = k_grid(n, dx)

    lam_true = lambda_model(k)
    phi = synthetic_phi(k, z)

    p = np.fft.fftshift(np.fft.ifft(phi)).real / dx
    p = np.maximum(p, 0.0)
    p /= np.trapz(p, x)

    phi_rec = np.fft.fft(np.fft.ifftshift(p)) * dx
    lam_rec = reconstruct_lambda(phi_rec, z)

    mask = np.abs(phi_rec) > 2e-2

    rel = np.abs(lam_rec - lam_true) / np.maximum(np.abs(lam_true), 1e-12)

    return {
        "mean_mu": float(np.trapz(np.exp(x) * p, x)),
        "phi0": complex(phi_rec[0]),
        "reliable_modes": int(mask.sum()),
        "median_relative_error": float(np.median(rel[mask])),
        "max_relative_error": float(np.max(rel[mask])),
    }


if __name__ == "__main__":
    out = synthetic_roundtrip()
    print("Synthetic transport validation")
    for k, v in out.items():
        print(f"{k}: {v}")
