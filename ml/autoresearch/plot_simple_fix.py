"""Plot the SIMPLE bound-fixed model (Normal-base NSF, bound widened 5->16, plain NLL)
vs the simulator, in the original 6-panel dP/dmu style. No splice, no StudentT, no penalty.

Tests whether just removing the spline-bound cutoff artifact gives nice plots.
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
from ml.autoresearch import train_ar

AR = Path(__file__).resolve().parent
REPO = AR.parents[1]
THETA = (0.72, 0.38, 1.00)
ZS = [0.5, 1.0, 2.0, 3.5, 5.0, 8.0]
NSIM = 300_000


def sim_lnmu(gw, z):
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, THETA), NSIM, 100, False)
    x = np.asarray(r["lnmu"], float); return x[np.isfinite(x)]


def main():
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    ck = torch.load(AR / "models" / "flow_ar.pt", map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ck["config"]); flow.load_state_dict(ck["state_dict"])
    fn = train_ar.make_log_prob_fn(flow, ck["stats"], torch.device("cpu"))

    lg = np.linspace(np.log(0.6), np.log(12.0), 500); mug = np.exp(lg)
    edges = np.exp(np.linspace(np.log(0.6), np.log(12.0), 60)); mue = edges
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for i, z in enumerate(ZS):
        ax = axes[i // 3][i % 3]
        mu = np.exp(sim_lnmu(gw, z))
        ax.hist(mu, bins=mue, density=True, histtype="stepfilled", color="#1f77b4",
                alpha=0.35, label="C++ simulator (MCMC)")
        ax.hist(mu, bins=mue, density=True, histtype="step", color="#1f77b4", lw=1.0)
        p_mu = np.exp(np.asarray(fn(lg, z, THETA), float)) / mug
        ax.plot(mug, p_mu, color="#d62728", lw=2.0, label="NSF emulator (bound-fixed)")
        ax.set_yscale("log"); ax.set_ylim(1e-3, 2e2); ax.set_xlim(0, 12)
        ax.set_title(f"z = {z}"); ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$dP/d\mu$")
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle(r"Magnification PDF $dP/d\mu$ — simulator vs SIMPLE bound-fixed NSF "
                 r"($h=0.72,\ \Omega_M=0.38,\ \sigma_8=1.0$)", y=1.0)
    fig.tight_layout()
    out = AR / "simple_fix_check.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
