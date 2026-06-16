"""Overlay the magnification PDF dP/dmu vs mu from the C++ simulator (MCMC) and
the NSF emulator, for several redshifts at a fixed cosmology.

The flow models density in ln(mu); we apply the Jacobian dP/dmu = p(lnmu)/mu
(mu = exp(lnmu)) to plot in mu space.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ml.phase3_common import import_gwlensing, load_nsf_model
from ml.run_phase3_posterior_grid import load_bin_edges


def sim_lnmu(z, theta, nsim, seed=100):
    gw = import_gwlensing()
    h, om, s8 = theta
    res = gw.sample_lnmu_ml_with_diagnostics(float(z), float(h), float(om), float(s8), int(nsim), int(seed), False)
    lnmu = np.asarray(res["lnmu"], dtype=np.float64)
    return lnmu[np.isfinite(lnmu)]


def nsf_density_lnmu(model, z, theta, lnmu_grid):
    dev = model.context_mean.device
    x = torch.tensor(lnmu_grid, dtype=torch.float32, device=dev).reshape(-1, 1)
    ctx = torch.tensor([[z, *theta]], dtype=torch.float32, device=dev).repeat(len(lnmu_grid), 1)
    with torch.no_grad():
        xn = (x - model.lnmu_mean) / model.lnmu_std
        cn = (ctx - model.context_mean) / model.context_std
        lp = model.flow(cn).log_prob(xn) - torch.log(model.lnmu_std)
        return torch.exp(lp).cpu().numpy().ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, nargs="+", default=[0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--h", type=float, default=0.67)
    ap.add_argument("--OmegaM", type=float, default=0.30)
    ap.add_argument("--sigma8", type=float, default=0.85)
    ap.add_argument("--nsim", type=int, default=200000)
    ap.add_argument("--model_path", default="data/models/conditional_nsf_backend_current.pt")
    ap.add_argument("--out", default="plots/figures/dpdmu_comparison.png")
    ap.add_argument("--xmin", type=float, default=0.6)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--logy", action="store_true", help="log y-axis to reveal the high-mu lensing tail")
    args = ap.parse_args()

    theta = (args.h, args.OmegaM, args.sigma8)
    bin_edges = load_bin_edges()                      # lnmu bins, e.g. [-1, 1]
    model = load_nsf_model(args.model_path)

    lnmu_grid = np.linspace(bin_edges[0], bin_edges[-1], 500)
    mu_grid = np.exp(lnmu_grid)
    mu_edges = np.exp(bin_edges)

    n = len(args.zs)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)

    for i, z in enumerate(args.zs):
        ax = axes[i // ncol][i % ncol]
        # simulator: histogram in mu space -> dP/dmu
        lnmu = sim_lnmu(z, theta, args.nsim)
        mu = np.exp(lnmu)
        ax.hist(mu, bins=mu_edges, density=True, histtype="stepfilled",
                color="#1f77b4", alpha=0.35, label="C++ simulator (MCMC)")
        ax.hist(mu, bins=mu_edges, density=True, histtype="step", color="#1f77b4", lw=1.0)
        # emulator: dP/dmu = p(lnmu)/mu
        p_lnmu = nsf_density_lnmu(model, z, theta, lnmu_grid)
        p_mu = p_lnmu / mu_grid
        ax.plot(mu_grid, p_mu, color="#d62728", lw=2.0, label="NSF emulator")
        ax.set_title(f"z = {z}")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(r"$dP/d\mu$")
        ax.set_xlim(args.xmin, args.xmax)
        if args.logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, None)
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=8)

    # hide unused panels
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(rf"Magnification PDF $dP/d\mu$ — simulator vs NSF emulator "
                 rf"($h={args.h},\ \Omega_M={args.OmegaM},\ \sigma_8={args.sigma8}$)", y=1.00)
    fig.tight_layout()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
