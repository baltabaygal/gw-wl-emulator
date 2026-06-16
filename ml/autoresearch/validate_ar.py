"""Validate the autoresearch winner against the production NSF.

Two honest checks, both on points NOT used to drive the metric:
  1. Held-out generalization: simulator survival vs production vs winner at
     cosmologies and redshifts outside the metric panel.
  2. dP/dmu overlay plot at the reported failure cosmology (high-structure corner).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AR_DIR = Path(__file__).resolve().parent
REPO = AR_DIR.parents[1]
MAIN_REPO = Path("/Users/baltabay/Desktop/gw-wl-emulator")
WINNER = AR_DIR / "models" / "flow_ar.pt"
PROD = MAIN_REPO / "data" / "models" / "conditional_nsf_backend_current.pt"

from ml.autoresearch import train_ar
from ml.autoresearch.prepare_ar import _import_simulator

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
THRESH = (2.0, 3.0, 4.0, 5.0)

# Held-out panel: cosmologies and redshifts NOT in the metric cache.
HELDOUT_COSMO = {"locorner": (0.60, 0.22, 0.70), "hicorner": (0.74, 0.39, 1.03)}
HELDOUT_ZS = (1.5, 2.5, 4.0, 6.5)


def winner_fn():
    ckpt = torch.load(WINNER, map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ckpt["config"])
    flow.load_state_dict(ckpt["state_dict"])
    return train_ar.make_log_prob_fn(flow, ckpt["stats"], torch.device("cpu")), ckpt["config"]


def prod_fn():
    sys.path.insert(0, str(MAIN_REPO))
    from ml.nsf_model import ConditionalNSF
    m = ConditionalNSF(input_dim=1, context_dim=4)
    m.load_checkpoint(str(PROD))
    m.eval()

    def fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, dtype=np.float32)
        ctx = np.array([[z, *theta]], dtype=np.float32)
        return m.log_prob(lnmu, ctx)

    return fn


def survival(fn, z, theta):
    grid = np.linspace(-3.0, 7.0, 5000)
    p = np.exp(np.asarray(fn(grid, float(z), tuple(map(float, theta))), dtype=np.float64))
    return np.array([float(_trapz(p[grid > np.log(t)], grid[grid > np.log(t)])) for t in THRESH])


def sim_survival(gw, z, theta, n=400_000):
    h, om, s8 = theta
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), float(h), float(om), float(s8), int(n), 100, False)
    lnmu = np.asarray(r["lnmu"], float); lnmu = lnmu[np.isfinite(lnmu)]; mu = np.exp(lnmu)
    return np.array([float(np.mean(mu > t)) for t in THRESH]), lnmu


def tlse(s_model, s_sim):
    eps = 1e-6
    terms = [abs(np.log10(max(m, eps)) - np.log10(max(s, eps)))
             for m, s in zip(s_model, s_sim) if s > 0]
    return float(np.mean(terms)) if terms else float("nan")


def heldout_table():
    gw = _import_simulator()
    wfn, wcfg = winner_fn(); pfn = prod_fn()
    print(f"\n=== HELD-OUT generalization (points NOT in metric panel) ===")
    print(f"winner config: bins={wcfg['bins']} bound={wcfg['bound']} base={wcfg['base']} "
          f"df={wcfg['df']} tail_weight={wcfg['tail_weight']}")
    print(f"{'point':18s} | {'mu>t':>4} | {'S_sim':>9} {'S_prod':>9} {'S_winner':>9}")
    prod_terms, win_terms = [], []
    for cname, theta in HELDOUT_COSMO.items():
        for z in HELDOUT_ZS:
            s_sim, _ = sim_survival(gw, z, theta)
            s_p = survival(pfn, z, theta); s_w = survival(wfn, z, theta)
            prod_terms.append(tlse(s_p, s_sim)); win_terms.append(tlse(s_w, s_sim))
            for j, t in enumerate(THRESH):
                lead = f"{cname}_z{z}" if j == 0 else ""
                print(f"{lead:18s} | {t:4.1f} | {s_sim[j]:9.5f} {s_p[j]:9.5f} {s_w[j]:9.5f}")
    print("-" * 60)
    print(f"HELD-OUT mean TLSE   production={np.nanmean(prod_terms):.4f}   "
          f"winner={np.nanmean(win_terms):.4f}")


def dpdmu_plot(out=AR_DIR / "validation_dpdmu.png"):
    gw = _import_simulator()
    wfn, _ = winner_fn(); pfn = prod_fn()
    theta = (0.72, 0.38, 1.00)  # the reported high-structure failure cosmology
    zs = [2.0, 3.5, 5.0, 8.0]
    grid = np.linspace(-1.0, 3.0, 600); mu_grid = np.exp(grid)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    for i, z in enumerate(zs):
        ax = axes[i // 2][i % 2]
        _, lnmu = sim_survival(gw, z, theta)
        mu = np.exp(lnmu)
        edges = np.exp(np.linspace(-1.0, 3.0, 60))
        ax.hist(mu, bins=edges, density=True, histtype="stepfilled",
                color="#1f77b4", alpha=0.30, label="simulator")
        for fn, c, lab in ((pfn, "#d62728", "production NSF"), (wfn, "#2ca02c", "winner (AR)")):
            p = np.exp(np.asarray(fn(grid, z, theta), float)) / mu_grid
            ax.plot(mu_grid, p, color=c, lw=2.0, label=lab)
        ax.set_yscale("log"); ax.set_ylim(1e-3, None); ax.set_xlim(0.6, 9)
        ax.set_title(f"z = {z}"); ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$dP/d\mu$")
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle(r"$dP/d\mu$ — simulator vs production vs autoresearch winner "
                 r"($h=0.72,\ \Omega_M=0.38,\ \sigma_8=1.00$)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    heldout_table()
    dpdmu_plot()
