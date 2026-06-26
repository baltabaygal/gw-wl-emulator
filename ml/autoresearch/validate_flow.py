"""Overlay the current flow (flow_ar.pt) on a CLEAN 10M full-tail reference (cached once).
Shows image-plane (raw, dP/dmu ~ mu^-2) and source-plane (x 1/mu -> mu^-3) in log-log.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ml.autoresearch import train_ar

AR = Path(__file__).resolve().parent; REPO = AR.parents[1]
THETA = (0.72, 0.38, 1.00); ZS = [2.0, 3.5, 5.0, 8.0]
NPER, NWORK = 1_000_000, 10
REF = AR / "cache" / "clean_tail_ref.npz"
EDGES = np.geomspace(1.0, 300.0, 45)

def _worker(args):
    z, seed = args; sys.path.insert(0, str(REPO/"build")); import gwlensing as gw
    r = gw.sample_lnmu_ml_with_diagnostics(float(z), *map(float, THETA), int(NPER), int(seed), False)
    x = np.asarray(r["lnmu"], float); return x[np.isfinite(x)]

def build_ref():
    if REF.exists():
        d = np.load(REF); return {k: d[k] for k in d.files}
    out = {"edges": EDGES}
    for z in ZS:
        with Pool(NWORK) as p: parts = p.map(_worker, [(z, 100+j) for j in range(NWORK)])
        mu = np.exp(np.concatenate(parts)); c, _ = np.histogram(mu, bins=EDGES)
        out[f"c_{z}"] = c; out[f"N_{z}"] = mu.size
        print(f"ref z={z}: N={mu.size} maxmu={mu.max():.0f}")
    REF.parent.mkdir(parents=True, exist_ok=True); np.savez(REF, **out); return out

def main():
    ref = build_ref()
    ck = torch.load(AR/"models/flow_ar.pt", map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ck["config"]); flow.load_state_dict(ck["state_dict"]); flow.eval()
    fn = train_ar.make_log_prob_fn(flow, ck["stats"], torch.device("cpu"))
    edges = ref["edges"]; ctr = np.sqrt(edges[:-1]*edges[1:]); bw = np.diff(edges)
    lg = np.linspace(np.log(0.6), np.log(300), 600); mug = np.exp(lg)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for i, z in enumerate(ZS):
        a = ax[i//2][i%2]; c = ref[f"c_{z}"]; N = int(ref[f"N_{z}"]); m = c >= 20
        a.scatter(ctr[m], c[m]/(N*bw[m]), s=14, color="#1f77b4", label="simulator (10M, image-plane)")
        p_img = np.exp(fn(lg, z, THETA))/mug
        a.plot(mug, p_img, color="#2ca02c", lw=2.2, label="flow (image-plane)")
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-7, 5); a.set_xlim(0.6, 300)
        a.set_title(f"z={z}"); a.set_xlabel(r"$\mu$"); a.set_ylabel(r"$dP/d\mu$"); a.grid(True, which="both", ls=":", alpha=.4)
        if i == 0: a.legend(fontsize=8)
    fig.suptitle(r"slope$=-1$ flow vs clean 10M simulator (image-plane $\mu^{-2}$) — log-log"); fig.tight_layout()
    fig.savefig(AR/"validate_flow.png", dpi=140, bbox_inches="tight"); print("wrote", AR/"validate_flow.png")

if __name__ == "__main__": main()
