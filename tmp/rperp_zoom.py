"""One-off zoom: P(lnmu) for R_perp = 8.44 / 10 / 18.2 Mpc (bias_model=1),
z_s = {1,5}, reusing the rperp_pdf_scan cache + protocol, EXTENDED to 64
shards = 960k realizations/arm for the four plotted arms (2026-07-16, smoother
curves; the scan's own arms keep their first 16 shards untouched). The 10 Mpc
arm has tag r10Mpc, arm_idx 90 in the scan's seed formula (disjoint from the
scan's 0..15).

Run: $PY tmp/rperp_zoom.py            -> plots/rperp_zoom_8to18.png
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

NSH = 64                                   # 64 x 15k = 960k / arm
scan_kw = {name: kw for name, kw in scan.arms()}
scan_ai = {name: i for i, (name, _) in enumerate(scan.arms())}
# (name, arm_idx for seeds, sampler kwargs, R in Mpc for labels, color)
ARMS = [
    ("m1e14",  scan_ai["m1e14"], scan_kw["m1e14"], 8.441, "#2a78d6"),
    ("r10Mpc", 90, dict(bias_model=1, bias_Rperp=10000.0), 10.0, "#1baf7a"),
    ("m1e15",  scan_ai["m1e15"], scan_kw["m1e15"], 18.2, "#b8860b"),
    ("nobias", scan_ai["nobias"], scan_kw["nobias"], None, "#e34948"),
]


def ensure_shards(jobs=8):
    todo = []
    for zs in scan.ZS_LIST:
        for name, ai, kw, _, _ in ARMS:
            for s in range(NSH):
                p = scan.shard_path(zs, name, s)
                if p.exists():
                    continue
                todo.append([sys.executable,
                             str(REPO / "scripts" / "convergence" / "rperp_pdf_scan.py"),
                             "shard", "--zs", str(zs),
                             "--seed", str(scan.shard_seed(zs, ai, s)),
                             "--n", str(scan.NPERSHARD), "--out", str(p),
                             "--kwargs", json.dumps(kw)])
    if not todo:
        return
    print(f"{len(todo)} shards to generate")
    t0, running, done = time.time(), [], 0
    while todo or running:
        while todo and len(running) < jobs:
            running.append(subprocess.Popen(todo.pop(0)))
        alive = []
        for p in running:
            if p.poll() is None:
                alive.append(p)
            else:
                done += 1
                if done % 32 == 0:
                    print(f"[{done}] ({time.time()-t0:.0f}s)", flush=True)
        running = alive
        time.sleep(0.3)
    print(f"done in {time.time()-t0:.0f}s")


def load(zs, name, half=None):
    ss = {None: range(NSH), "A": range(NSH // 2), "B": range(NSH // 2, NSH)}[half]
    return np.concatenate([np.load(scan.shard_path(zs, name, s))["lnmu"]
                           for s in ss]).astype(np.float64)


def main():
    ensure_shards()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for iz, zs in enumerate(scan.ZS_LIST):
        data = {name: load(zs, name) for name, *_ in ARMS}
        ref = data["m1e14"]
        blo, bhi = np.quantile(ref, 5e-4), np.quantile(ref, 0.995)
        bedges = np.linspace(blo - 0.05, bhi, 221)
        bctr = 0.5 * (bedges[1:] + bedges[:-1])
        tedges = np.linspace(blo - 0.1, np.quantile(ref, 1 - 1e-5), 131)
        tctr = 0.5 * (tedges[1:] + tedges[:-1])
        redges = np.linspace(blo - 0.05, np.quantile(ref, 1 - 2e-3), 41)
        rctr = 0.5 * (redges[1:] + redges[:-1])
        href = np.histogram(ref, bins=redges, density=True)[0]
        hA = np.histogram(load(zs, "m1e14", "A"), bins=redges, density=True)[0]
        hB = np.histogram(load(zs, "m1e14", "B"), bins=redges, density=True)[0]
        band = np.abs(hA - hB) / np.where(href > 0, href, np.inf) / 2.0
        axes[iz, 2].fill_between(rctr, 1.0 - band, 1.0 + band, color="#8a8776",
                                 alpha=0.25, lw=0,
                                 label="8.44 Mpc seed-half spread")

        for name, _, _, Rmpc, col in ARMS:
            x = data[name]
            lab = f"$R_\\perp = {Rmpc:g}$ Mpc" if Rmpc else "no bias"
            sty = dict(color=col, lw=1.1) if Rmpc else dict(color=col, ls=":", lw=1.3)
            axes[iz, 0].plot(bctr, np.histogram(x, bins=bedges, density=True)[0],
                             label=lab, **sty)
            axes[iz, 1].semilogy(tctr, np.histogram(x, bins=tedges, density=True)[0],
                                 **sty)
            h = np.histogram(x, bins=redges, density=True)[0]
            axes[iz, 2].plot(rctr, np.divide(h, href, out=np.full_like(h, np.nan),
                                             where=href > 0), label=lab,
                             **{**sty, "lw": sty["lw"] + 0.2})

        axes[iz, 0].set_title(f"$z_s={zs:g}$: body (linear)")
        axes[iz, 0].set_ylabel(r"$P(\ln\mu)$")
        axes[iz, 1].set_title(f"$z_s={zs:g}$: tail (log)")
        axes[iz, 1].set_ylim(1e-4, None)
        axes[iz, 2].set_title(f"$z_s={zs:g}$: ratio to $R_\\perp=8.44$ Mpc")
        axes[iz, 2].set_ylabel(r"$P/P_{8.44\,{\rm Mpc}}$")
        axes[iz, 2].axhline(1.0, color="k", lw=0.6)
        axes[iz, 2].set_ylim(0.85, 1.15)
        for j in range(3):
            axes[iz, j].set_xlabel(r"$\ln\mu$")
        axes[iz, 0].legend(fontsize=9, frameon=False)
        axes[iz, 2].legend(fontsize=8, frameon=False)
    fig.suptitle("bias_model=1: PDF sensitivity across the candidate window range "
                 "(960k/arm, fixed-$\\langle N\\rangle$, kappa_anchor=1)", y=0.995)
    fig.tight_layout()
    out = REPO / "plots" / "rperp_zoom_8to18.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
