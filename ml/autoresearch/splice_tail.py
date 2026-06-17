"""R1: body-flow + conditional analytic power-law (GPD) tail splice.

The flow models the body (it's faithful there). Above a threshold mu_u we replace
the tail with an analytic power law dP/dmu ~ mu^-alpha:
  - SLOPE fixed to the universal benchmark (alpha ~ 3.4).
  - AMPLITUDE pinned by continuity to the flow's own density at mu_u -> conditional
    on (z, cosmology) for free, no global tilt, smooth, correct per regime.

In lnmu the tail is exponential: p(lnmu) = p_flow(lnmu_u) * exp(-(alpha-1)(lnmu-lnmu_u)).
The spliced density is renormalized so it integrates to 1.

Exposes a log_prob_fn(lnmu, z, theta) compatible with prepare_ar.evaluate.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.autoresearch import train_ar, prepare_ar

AR = Path(__file__).resolve().parent
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def load_body_fn(path=AR / "models" / "flow_ar.pt"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ckpt["config"]); flow.load_state_dict(ckpt["state_dict"])
    return train_ar.make_log_prob_fn(flow, ckpt["stats"], torch.device("cpu")), ckpt


def make_spliced_log_prob_fn(body_fn, mu_u=2.0, alpha=3.4, anchor="survival", lnmu_min=-4.0):
    """Body flow below mu_u; analytic mu^-alpha tail above.

    anchor="survival" (recommended): tail carries the flow's survival mass S_flow(mu_u)
      (reliable at moderate mu_u, data-rich) redistributed as a power law. Auto-normalized.
      S_spliced(mu>t) = S_flow(mu_u) * (t/mu_u)^-(alpha-1).
    anchor="density": match the flow density at mu_u (continuity; sensitive to far-tail
      undershoot of the body flow).
    """
    lnu = float(np.log(mu_u)); k = float(alpha - 1.0)
    gb = np.linspace(lnmu_min, lnu, 2000)               # grid for body CDF up to threshold

    def fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, dtype=np.float64).ravel()
        pb = np.exp(np.asarray(body_fn(gb, z, theta), np.float64))
        cdf_u = float(_trapz(pb, gb))                   # P(lnmu <= lnu) from the flow body
        if anchor == "survival":
            s_tail = max(1.0 - cdf_u, 1e-12)            # flow's mass above mu_u (reliable)
            p_u = k * s_tail                            # power-law density at lnu (mass-matched)
            total = 1.0                                 # cdf_u + s_tail = 1 already
        else:                                           # density continuity
            p_u = float(np.exp(np.asarray(body_fn(np.array([lnu]), z, theta), np.float64))[0])
            total = cdf_u + p_u / k
        out = np.empty_like(lnmu)
        below = lnmu <= lnu
        if below.any():
            out[below] = np.exp(np.asarray(body_fn(lnmu[below], z, theta), np.float64))
        if (~below).any():
            out[~below] = p_u * np.exp(-k * (lnmu[~below] - lnu))
        return np.log(np.maximum(out / total, 1e-300))

    return fn


def plot(body_fn, spliced_fn, mu_u, alpha, out=AR / "splice_check.png"):
    d = np.load(prepare_ar.CACHE_PATH, allow_pickle=True)
    tags = [str(t) for t in d["tags"]]
    pts = [("stress_z2.0", 2.0), ("stress_z3.5", 3.5), ("stress_z5.0", 5.0), ("stress_z8.0", 8.0)]
    theta = (0.72, 0.38, 1.00)
    MU_LO, MU_HI = 1.3, 13.0
    lg = np.linspace(np.log(MU_LO), np.log(MU_HI), 400); mug = np.exp(lg)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for k, (tag, z) in enumerate(pts):
        ax = axes[k // 2][k % 2]
        mu = np.exp(np.asarray(d[f"body_{tag}"], float))
        edges = np.exp(np.linspace(np.log(MU_LO), np.log(MU_HI), 26))
        cnt, _ = np.histogram(mu, bins=edges); ctr = np.sqrt(edges[:-1] * edges[1:]); bw = np.diff(edges)
        m = cnt >= 10
        ax.scatter(ctr[m], cnt[m] / (mu.size * bw[m]), s=22, color="#1f77b4", zorder=3, label="simulator")
        ax.plot(mug, np.exp(np.asarray(spliced_fn(lg, z, theta), float)) / mug,
                color="#2ca02c", lw=2.4, label="spliced (body+GPD tail)")
        ax.axvline(mu_u, color="gray", ls=":", lw=1, label=f"threshold μ={mu_u}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 5); ax.set_xlim(MU_LO, MU_HI)
        ax.set_title(tag); ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$dP/d\mu$")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(fr"Body-flow + conditional $\mu^{{-{alpha}}}$ tail splice (threshold $\mu_u={mu_u}$)")
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu_u", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=3.4)
    args = ap.parse_args()
    body_fn, ckpt = load_body_fn()
    spliced = make_spliced_log_prob_fn(body_fn, mu_u=args.mu_u, alpha=args.alpha)
    print(f"=== splice mu_u={args.mu_u} alpha={args.alpha} (body cfg bins={ckpt['config']['bins']} "
          f"bound={ckpt['config']['bound']} tilt={ckpt['config'].get('tail_tilt')}) ===")
    print("--- BODY flow alone ---")
    mb = prepare_ar.evaluate(body_fn, verbose=False)
    print({k: round(v, 4) for k, v in mb.items()})
    print("--- SPLICED ---")
    ms = prepare_ar.evaluate(spliced, verbose=True)
    print({k: round(v, 4) for k, v in ms.items()})
    plot(body_fn, spliced, args.mu_u, args.alpha)


if __name__ == "__main__":
    main()
