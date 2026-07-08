"""Phase 3 density-accuracy check (the emulator's actual job).

Compares the NSF predicted lnmu PMF against a HIGH-statistics simulator PMF at
a handful of (z, theta) points. This is converged and clean -- it isolates
emulation fidelity from the N=1000 posterior over-concentration that breaks the
smoke gate.

Metrics per (z, theta): total variation, KL(sim||nsf), and the catalog-support
mean log-density residual.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ml.phase3_common import ensure_dir, import_gwlensing, load_nsf_model
from ml.run_phase3_posterior_grid import load_bin_edges


def sim_pmf(z, theta, bin_edges, nsim, seed=100):
    gw = import_gwlensing()
    h, om, s8 = [float(x) for x in theta]
    res = gw.sample_lnmu_ml_with_diagnostics(float(z), h, om, s8, int(nsim), int(seed), False)
    lnmu = np.asarray(res["lnmu"], dtype=np.float64)
    lnmu = lnmu[np.isfinite(lnmu)]
    clamped = np.clip(lnmu, bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
    counts, _ = np.histogram(clamped, bins=bin_edges)
    p = counts / max(float(counts.sum()), 1.0)
    return p


def nsf_pmf(model, z, theta, bin_edges):
    import torch
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = np.diff(bin_edges)
    dev = model.context_mean.device
    x = torch.tensor(centers, dtype=torch.float32, device=dev).reshape(-1, 1)
    ctx = torch.tensor([[z, *theta]], dtype=torch.float32, device=dev).repeat(len(centers), 1)
    with torch.no_grad():
        xn = (x - model.lnmu_mean) / model.lnmu_std
        cn = (ctx - model.context_mean) / model.context_std
        lp = model.flow(cn).log_prob(xn) - torch.log(model.lnmu_std)
        dens = torch.exp(lp).cpu().numpy()
    p = dens * widths
    p = p / p.sum()
    return p


def tv_kl(p, q):
    eps = 1e-12
    tv = float(0.5 * np.sum(np.abs(p - q)))
    kl = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))  # KL(p||q), p=sim
    return tv, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=200000)
    ap.add_argument("--model_path", default="data/models/conditional_nsf_backend_current.pt")
    ap.add_argument("--out_json", default="data/results/phase3_density_accuracy.json")
    ap.add_argument("--out_md", default="docs/phase3/phase3_density_accuracy.md")
    args = ap.parse_args()

    bin_edges = load_bin_edges()
    model = load_nsf_model(args.model_path)

    truth = (0.67, 0.30, 0.85)
    points = {
        "truth": truth,
        "sim_smoke_map": (0.67, 0.335, 0.747),
        "nsf_corner": (0.67, 0.20, 1.05),
        "mid_grid": (0.67, 0.27, 0.92),
        "high_OmegaM": (0.67, 0.40, 0.65),
    }
    zs = [0.5, 1.5, 2.5]

    rows = []
    for label, theta in points.items():
        for z in zs:
            ps = sim_pmf(z, theta, bin_edges, args.nsim)
            pn = nsf_pmf(model, z, theta, bin_edges)
            tv, kl = tv_kl(ps, pn)
            # mean log-density residual over bins where sim has support
            sup = ps > 0
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            widths = np.diff(bin_edges)
            sim_logdens = np.log(ps[sup] / widths[sup] + 1e-30)
            nsf_logdens = np.log(pn[sup] / widths[sup] + 1e-30)
            mean_resid = float(np.mean(nsf_logdens - sim_logdens))
            rows.append({"point": label, "theta": list(theta), "z": z,
                         "tv": round(tv, 5), "kl_sim_nsf": round(kl, 5),
                         "mean_logdens_resid": round(mean_resid, 4)})
            print(f"{label:14s} z={z}  TV={tv:.4f}  KL(sim||nsf)={kl:.4f}  meanLogDensResid={mean_resid:+.3f}", flush=True)

    ensure_dir(Path(args.out_json).parent)
    Path(args.out_json).write_text(json.dumps({"nsim": args.nsim, "rows": rows}, indent=2))

    L = ["# Phase 3 Density Accuracy (NSF vs high-statistics simulator)", "",
         f"Simulator statistics: nsim = {args.nsim} per (z, theta). 100-bin PMF over lnmu in [-1, 1].", "",
         "TV is total-variation between the two PMFs (0 = identical). KL(sim||nsf) in nats.", "",
         "| Point | theta (h,OmegaM,sigma8) | z | TV | KL(sim‖nsf) | mean logdens resid |",
         "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        th = ",".join(f"{x:.3f}" for x in r["theta"])
        L.append(f"| {r['point']} | {th} | {r['z']} | {r['tv']} | {r['kl_sim_nsf']} | {r['mean_logdens_resid']:+.3f} |")
    L.append("")
    ensure_dir(Path(args.out_md).parent)
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
