"""Fast tail-shape plot from the cached simulator samples (no new simulator calls).

Overlays dP/dmu (log-y) of the simulator (cached body samples) vs the current
flow model, plus the benchmark power-law slope, at the stress high-z points.
Shows whether the model tail is smooth and follows the power law.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ml.autoresearch import train_ar

AR = Path(__file__).resolve().parent
CACHE = AR / "cache" / "groundtruth.npz"
POINTS = [("stress_z2.0", 2.0), ("stress_z3.5", 3.5), ("stress_z5.0", 5.0), ("stress_z8.0", 8.0)]
THETA = (0.72, 0.38, 1.00)
ALPHA = 3.4


def main():
    d = np.load(CACHE, allow_pickle=True)
    tags = [str(t) for t in d["tags"]]
    ckpt = torch.load(AR / "models" / "flow_ar.pt", map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ckpt["config"]); flow.load_state_dict(ckpt["state_dict"])
    fn = train_ar.make_log_prob_fn(flow, ckpt["stats"], torch.device("cpu"))

    MU_LO, MU_HI = 1.3, 13.0   # tail-focused log-log window
    lnmu_grid = np.linspace(np.log(MU_LO), np.log(MU_HI), 400)
    mu_grid = np.exp(lnmu_grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for k, (tag, z) in enumerate(POINTS):
        ax = axes[k // 2][k % 2]
        # simulator as points (bin centers), only bins with enough counts (suppress MC noise)
        mu = np.exp(np.asarray(d[f"body_{tag}"], float))
        edges = np.exp(np.linspace(np.log(MU_LO), np.log(MU_HI), 26))
        cnt, _ = np.histogram(mu, bins=edges)
        ctr = np.sqrt(edges[:-1] * edges[1:]); bw = np.diff(edges)
        dens = cnt / (mu.size * bw)
        m = cnt >= 10
        ax.scatter(ctr[m], dens[m], s=22, color="#1f77b4", zorder=3, label="simulator")
        # flow curve
        p_mu = np.exp(np.asarray(fn(lnmu_grid, z, THETA), float)) / mu_grid
        ax.plot(mu_grid, p_mu, color="#2ca02c", lw=2.4, label="flow (slope-reg)")
        # benchmark power-law guide anchored at mu=3
        i0 = np.argmin(np.abs(mu_grid - 3.0))
        guide = p_mu[i0] * (mu_grid / 3.0) ** (-ALPHA)
        gm = mu_grid >= 2.0
        ax.plot(mu_grid[gm], guide[gm], "k--", lw=1.3, alpha=0.8, label=fr"$\mu^{{-{ALPHA}}}$ guide")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-4, 5); ax.set_xlim(MU_LO, MU_HI)
        ax.set_title(f"{tag}"); ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$dP/d\mu$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if k == 0:
            ax.legend(fontsize=9)
    fig.suptitle(r"Tail (log-log): simulator vs slope-regularized flow ($h=0.72,\Omega_M=0.38,\sigma_8=1.00$)")
    fig.tight_layout()
    out = AR / "tail_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
