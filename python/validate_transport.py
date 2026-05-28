import json
import numpy as np

try:
    import gwlensing as gw
except Exception:
    gw = None


def make_x_grid(xmax=4.0, n=4096):
    dx = 2.0 * xmax / n
    x = (np.arange(n) - n // 2) * dx
    return x, dx


def sample_lnmu(z, OmegaM=0.315, sigma8=0.811, h=0.674, Nreal=50000, seed=123,
                filaments=True, bias=True, ell=True, Nhalos=100):
    if gw is None:
        raise RuntimeError("gwlensing extension is not available")
    return np.asarray(gw.sample_lnmu(z=z, OmegaM=OmegaM, sigma8=sigma8, h=h,
                                      Nreal=Nreal, seed=seed, filaments=filaments,
                                      bias=bias, ell=ell, Nhalos=Nhalos), dtype=float)


def lnmu_to_px(lnmu, x_grid):
    dx = x_grid[1] - x_grid[0]
    edges = np.concatenate(([x_grid[0] - dx / 2.0], 0.5 * (x_grid[:-1] + x_grid[1:]), [x_grid[-1] + dx / 2.0]))
    hist, _ = np.histogram(lnmu, bins=edges, density=True)
    p = hist.astype(float)
    norm = np.trapz(p, x_grid)
    if norm > 0:
        p /= norm
    return p


def fft_phi(p_x, dx):
    return np.fft.fft(np.fft.ifftshift(p_x)) * dx


def lambda_eff(phi, z, floor=1e-12):
    phi = np.asarray(phi, dtype=complex)
    safe = phi.copy()
    m = np.abs(safe) < floor
    safe[m] = floor * np.exp(1j * np.angle(safe[m]))
    return np.log(safe) / z


def validate_redshift(z, **sampler_kwargs):
    x, dx = make_x_grid()
    lnmu = sample_lnmu(z, **sampler_kwargs)
    p_x = lnmu_to_px(lnmu, x)
    phi = fft_phi(p_x, dx)
    lam = lambda_eff(phi, z)
    mu = np.exp(x)
    return {
        "z": float(z),
        "n_samples": int(lnmu.size),
        "norm": float(np.trapz(p_x, x)),
        "mean_mu": float(np.trapz(mu * p_x, x)),
        "phi0_real": float(np.real(phi[0])),
        "phi0_imag": float(np.imag(phi[0])),
        "reliable_modes": int(np.count_nonzero(np.abs(phi) > 2e-2)),
        "lambda_median_abs": float(np.median(np.abs(lam[np.abs(phi) > 2e-2])) if np.any(np.abs(phi) > 2e-2) else np.inf),
    }


if __name__ == "__main__":
    reports = [validate_redshift(z) for z in [0.5, 1.0, 2.0, 5.0]]
    print(json.dumps(reports, indent=2, sort_keys=True))
