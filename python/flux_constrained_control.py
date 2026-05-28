import numpy as np


def make_x_grid(xmax=4.0, n=4096):
    dx = 2.0 * xmax / n
    x = (np.arange(n) - n // 2) * dx
    return x, dx


def k_grid(n, dx):
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def exp_jump_ft(k, xi0=0.25):
    return 1.0 / (1.0 + 1j * k * xi0)


def lambda_model(k, v=0.0, eps=0.01, rate=0.7, xi0=0.25):
    return -1j * v * k - eps * k**2 + rate * (exp_jump_ft(k, xi0) - 1.0)


def flux_conserving_v(eps=0.01, rate=0.7, xi0=0.25):
    # Enforce Lambda(-i)=0 so that <mu>=1 for all z.
    w_tilde_minus_i = exp_jump_ft(-1j, xi0)
    return eps + rate * (w_tilde_minus_i - 1.0)


def synthetic_phi(k, z, v=None, eps=0.01, rate=0.7, xi0=0.25):
    if v is None:
        v = flux_conserving_v(eps=eps, rate=rate, xi0=xi0)
    return np.exp(z * lambda_model(k, v=v, eps=eps, rate=rate, xi0=xi0))


def synthetic_pdf(z, xmax=4.0, n=4096, v=None, eps=0.01, rate=0.7, xi0=0.25):
    x, dx = make_x_grid(xmax=xmax, n=n)
    k = k_grid(n, dx)
    phi = synthetic_phi(k, z, v=v, eps=eps, rate=rate, xi0=xi0)
    p = np.fft.fftshift(np.fft.ifft(phi)).real / dx
    p = np.maximum(p, 0.0)
    p /= np.trapz(p, x)
    return x, p, phi


def check_flux(z_values=(0.5, 1.0, 2.0, 5.0), xmax=4.0, n=4096, eps=0.01, rate=0.7, xi0=0.25):
    v = flux_conserving_v(eps=eps, rate=rate, xi0=xi0)
    out = []
    for z in z_values:
        x, p, phi = synthetic_pdf(z, xmax=xmax, n=n, v=v, eps=eps, rate=rate, xi0=xi0)
        mu = np.exp(x)
        out.append({
            "z": float(z),
            "v": complex(v),
            "norm": float(np.trapz(p, x)),
            "mean_mu": float(np.trapz(mu * p, x)),
            "phi0": complex(phi[0]),
            "ln_mean_mu_over_z": float(np.log(np.trapz(mu * p, x)) / z),
        })
    return out


if __name__ == "__main__":
    results = check_flux()
    for row in results:
        print(row)
