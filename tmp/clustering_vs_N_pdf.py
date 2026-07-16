"""Per-<N> PDF view of the clustering signature (z_s = 1, field R_perp = 8.44
Mpc vs matched no-bias), <N> in {100, 300, 1000}. Extends the N-arms to 32
shards = 480k (the <N>=100 pair uses the 64-shard/960k cache), then plots one
column per <N>: PDF pair on top, ratio + seed-half noise band below.

Run: $PY tmp/clustering_vs_N_pdf.py   -> plots/clustering_vs_N_pdf.png
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "scan", REPO / "scripts" / "convergence" / "rperp_pdf_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

ZS = 1.0
R844 = {name: kw for name, kw in scan.arms()}["m1e14"]["bias_Rperp"]
# (<N>, nobias tag, field tag, nobias arm_idx, field arm_idx, n shards)
CFG = [(100, "nobias", "m1e14", None, None, 64),
       (300, "nob_N300", "f844_N300", 91, 92, 32),
       (1000, "nob_N1000", "f844_N1000", 93, 94, 32)]


def ensure(jobs=8):
    todo = []
    for N, nob, fld, ai_n, ai_f, nsh in CFG:
        if ai_n is None:
            continue
        for tag, ai, kw in ((nob, ai_n, dict(bias=False, Nhalos=N)),
                            (fld, ai_f, dict(bias_model=1, bias_Rperp=R844, Nhalos=N))):
            for s in range(nsh):
                p = scan.shard_path(ZS, tag, s)
                if p.exists():
                    continue
                todo.append([sys.executable,
                             str(REPO / "scripts" / "convergence" / "rperp_pdf_scan.py"),
                             "shard", "--zs", str(ZS),
                             "--seed", str(scan.shard_seed(ZS, ai, s)),
                             "--n", str(scan.NPERSHARD), "--out", str(p),
                             "--kwargs", json.dumps(kw)])
    if not todo:
        return
    print(f"{len(todo)} shards to generate")
    t0, running = time.time(), []
    while todo or running:
        while todo and len(running) < jobs:
            running.append(subprocess.Popen(todo.pop(0)))
        running = [p for p in running if p.poll() is None]
        time.sleep(0.3)
    print(f"done in {time.time()-t0:.0f}s")


def load(tag, nsh, half=None):
    ss = {None: range(nsh), "A": range(nsh // 2), "B": range(nsh // 2, nsh)}[half]
    return np.concatenate([np.load(scan.shard_path(ZS, tag, s))["lnmu"]
                           for s in ss]).astype(np.float64)


def cstd(x):
    q = np.quantile(x, [0.005, 0.995])
    return np.std(x[(x > q[0]) & (x < q[1])])


def main():
    ensure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5),
                             gridspec_kw=dict(height_ratios=[1.5, 1]))
    for jc, (N, nob_t, fld_t, _, _, nsh) in enumerate(CFG):
        nob, fld = load(nob_t, nsh), load(fld_t, nsh)
        sn, sf = cstd(nob), cstd(fld)
        share = 100.0 * (sf ** 2 - sn ** 2) / sf ** 2
        lo, hi = np.quantile(fld, 5e-4) - 0.02, np.quantile(fld, 1 - 2e-3)
        bedges = np.linspace(lo, hi, 181)
        bctr = 0.5 * (bedges[1:] + bedges[:-1])
        redges = np.linspace(lo, hi, 37)
        rctr = 0.5 * (redges[1:] + redges[:-1])

        ax = axes[0, jc]
        ax.plot(bctr, np.histogram(fld, bins=bedges, density=True)[0],
                color="#2a78d6", lw=1.2, label=f"field 8.44 Mpc ($\\sigma_c$={sf:.4f})")
        ax.plot(bctr, np.histogram(nob, bins=bedges, density=True)[0],
                color="#e34948", ls=":", lw=1.4, label=f"no bias ($\\sigma_c$={sn:.4f})")
        ax.set_title(f"$\\langle N\\rangle = {N}$:  clustering share {share:.1f}%")
        ax.set_ylabel(r"$P(\ln\mu)$" if jc == 0 else None)
        ax.legend(fontsize=8.5, frameon=False)

        ax = axes[1, jc]
        hn = np.histogram(nob, bins=redges, density=True)[0]
        hf = np.histogram(fld, bins=redges, density=True)[0]
        ratio = np.divide(hf, hn, out=np.full_like(hf, np.nan), where=hn > 0)
        rA = np.histogram(load(fld_t, nsh, "A"), bins=redges, density=True)[0] / \
             np.maximum(np.histogram(load(nob_t, nsh, "A"), bins=redges, density=True)[0], 1e-300)
        rB = np.histogram(load(fld_t, nsh, "B"), bins=redges, density=True)[0] / \
             np.maximum(np.histogram(load(nob_t, nsh, "B"), bins=redges, density=True)[0], 1e-300)
        band = 0.5 * np.abs(rA - rB)
        ax.fill_between(rctr, ratio - band, ratio + band, color="#2a78d6",
                        alpha=0.2, lw=0)
        ax.plot(rctr, ratio, color="#2a78d6", lw=1.4)
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_ylim(0.82, 1.3)
        ax.set_xlabel(r"$\ln\mu$")
        ax.set_ylabel(r"$P_{\rm field}/P_{\rm nobias}$" if jc == 0 else None)
    fig.suptitle("Clustering signature vs $\\langle N\\rangle$ at $z_s=1$ "
                 "(field $R_\\perp$=8.44 Mpc vs matched no-bias; 480k–960k/arm; "
                 "band = seed-half spread; $\\sigma_c$ = 99%-clipped std)", y=0.99)
    fig.tight_layout()
    out = REPO / "plots" / "clustering_vs_N_pdf.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
