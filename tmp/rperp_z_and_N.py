"""Two follow-ups to the R_perp zoom (2026-07-16):

A. Window zoom at z_s = 0.5 and 10 (arms 8.44/10/18.2 Mpc + nobias, 32 shards
   = 480k/arm)  -> plots/rperp_zoom_z05_z10.png
B. Clustering share vs <N> = Nhalos in {100, 300, 1000} at z_s = 1, arms
   {nobias, field 8.44 Mpc} (16 shards = 240k for the new N; N=100 reuses the
   64-shard extended cache) -> plots/clustering_vs_N.png
   Hypothesis: only EXPLICIT halo counts are field-modulated (kappa_W Gaussian
   is not), so raising <N> moves variance from the unmodulated background into
   the modulated Poisson term => clustering share grows (poor-man's preview of
   the joint framework's weak arm).

Seeds: scan formula SEED0 + 1e6*int(10*zs) + 1e4*arm_idx + s; new arm indices
90 (r10Mpc, existing), 91-94 (N arms). Shards cached in the scan's mc dir.
Run: $PY tmp/rperp_z_and_N.py
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

scan_kw = {name: kw for name, kw in scan.arms()}
scan_ai = {name: i for i, (name, _) in enumerate(scan.arms())}
R844 = scan_kw["m1e14"]["bias_Rperp"]

ZOOM_ZS = [0.5, 10.0]
ZOOM_NSH = 32
ZOOM_ARMS = [
    ("m1e14",  scan_ai["m1e14"], scan_kw["m1e14"], 8.441, "#2a78d6"),
    ("r10Mpc", 90, dict(bias_model=1, bias_Rperp=10000.0), 10.0, "#1baf7a"),
    ("m1e15",  scan_ai["m1e15"], scan_kw["m1e15"], 18.2, "#b8860b"),
    ("nobias", scan_ai["nobias"], scan_kw["nobias"], None, "#e34948"),
]

NLIST = [100, 300, 1000]
N_NSH = 16
N_ARMS = [  # (tag, arm_idx, kwargs); N=100 arms come from the extended cache
    ("nob_N300",  91, dict(bias=False, Nhalos=300)),
    ("f844_N300", 92, dict(bias_model=1, bias_Rperp=R844, Nhalos=300)),
    ("nob_N1000", 93, dict(bias=False, Nhalos=1000)),
    ("f844_N1000", 94, dict(bias_model=1, bias_Rperp=R844, Nhalos=1000)),
]


def run_jobs(todo, jobs=8):
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


def shard_cmd(zs, tag, ai, s, kw):
    return [sys.executable,
            str(REPO / "scripts" / "convergence" / "rperp_pdf_scan.py"),
            "shard", "--zs", str(zs), "--seed", str(scan.shard_seed(zs, ai, s)),
            "--n", str(scan.NPERSHARD), "--out", str(scan.shard_path(zs, tag, s)),
            "--kwargs", json.dumps(kw)]


def load(zs, name, nsh, half=None):
    ss = {None: range(nsh), "A": range(nsh // 2), "B": range(nsh // 2, nsh)}[half]
    return np.concatenate([np.load(scan.shard_path(zs, name, s))["lnmu"]
                           for s in ss]).astype(np.float64)


def ensure_all():
    todo = []
    for zs in ZOOM_ZS:
        for tag, ai, kw, _, _ in ZOOM_ARMS:
            for s in range(ZOOM_NSH):
                if not scan.shard_path(zs, tag, s).exists():
                    todo.append(shard_cmd(zs, tag, ai, s, kw))
    for tag, ai, kw in N_ARMS:
        for s in range(N_NSH):
            if not scan.shard_path(1.0, tag, s).exists():
                todo.append(shard_cmd(1.0, tag, ai, s, kw))
    run_jobs(todo)


def fig_zoom(plt, matplotlib):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for iz, zs in enumerate(ZOOM_ZS):
        data = {tag: load(zs, tag, ZOOM_NSH) for tag, *_ in ZOOM_ARMS}
        ref = data["m1e14"]
        blo, bhi = np.quantile(ref, 5e-4), np.quantile(ref, 0.995)
        bedges = np.linspace(blo - 0.05, bhi, 221)
        bctr = 0.5 * (bedges[1:] + bedges[:-1])
        tedges = np.linspace(blo - 0.1, np.quantile(ref, 1 - 1e-5), 131)
        tctr = 0.5 * (tedges[1:] + tedges[:-1])
        redges = np.linspace(blo - 0.05, np.quantile(ref, 1 - 2e-3), 41)
        rctr = 0.5 * (redges[1:] + redges[:-1])
        href = np.histogram(ref, bins=redges, density=True)[0]
        hA = np.histogram(load(zs, "m1e14", ZOOM_NSH, "A"), bins=redges, density=True)[0]
        hB = np.histogram(load(zs, "m1e14", ZOOM_NSH, "B"), bins=redges, density=True)[0]
        band = np.abs(hA - hB) / np.where(href > 0, href, np.inf) / 2.0
        axes[iz, 2].fill_between(rctr, 1 - band, 1 + band, color="#8a8776",
                                 alpha=0.25, lw=0, label="8.44 Mpc seed-half spread")
        for tag, _, _, Rmpc, col in ZOOM_ARMS:
            x = data[tag]
            lab = f"$R_\\perp = {Rmpc:g}$ Mpc" if Rmpc else "no bias"
            sty = dict(color=col, lw=1.1) if Rmpc else dict(color=col, ls=":", lw=1.3)
            axes[iz, 0].plot(bctr, np.histogram(x, bins=bedges, density=True)[0],
                             label=lab, **sty)
            axes[iz, 1].semilogy(tctr, np.histogram(x, bins=tedges, density=True)[0], **sty)
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
    fig.suptitle("bias_model=1 window sensitivity at $z_s=0.5, 10$ "
                 "(480k/arm, fixed-$\\langle N\\rangle=100$, kappa_anchor=1)", y=0.995)
    fig.tight_layout()
    fig.savefig(REPO / "plots" / "rperp_zoom_z05_z10.png", dpi=150)
    print("wrote plots/rperp_zoom_z05_z10.png")


def fig_N(plt, matplotlib):
    zs = 1.0
    pairs = {100: (load(zs, "nobias", 64), load(zs, "m1e14", 64)),
             300: (load(zs, "nob_N300", N_NSH), load(zs, "f844_N300", N_NSH)),
             1000: (load(zs, "nob_N1000", N_NSH), load(zs, "f844_N1000", N_NSH))}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    cols = {100: "#2a78d6", 300: "#1baf7a", 1000: "#b8860b"}
    share, sigs = [], []
    for N in NLIST:
        nob, fld = pairs[N]
        s2n, s2f = np.var(nob), np.var(fld)
        share.append(100.0 * (s2f - s2n) / s2f)
        sigs.append((np.std(nob), np.std(fld)))
        edges = np.linspace(np.quantile(fld, 5e-4) - 0.05,
                            np.quantile(fld, 1 - 2e-3), 41)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        hn = np.histogram(nob, bins=edges, density=True)[0]
        hf = np.histogram(fld, bins=edges, density=True)[0]
        axes[1].plot(ctr, np.divide(hf, hn, out=np.full_like(hf, np.nan),
                                    where=hn > 0), color=cols[N], lw=1.4,
                     label=f"$\\langle N\\rangle = {N}$")
    axes[0].semilogx(NLIST, share, "o-", color="#2a78d6", lw=1.5)
    for N, sh, (sn, sf) in zip(NLIST, share, sigs):
        axes[0].annotate(f"$\\sigma$: {sn:.4f}$\\to${sf:.4f}", (N, sh),
                         textcoords="offset points", xytext=(8, -12), fontsize=8)
    axes[0].set_xlabel(r"$\langle N\rangle$ (explicit-halo threshold rule)")
    axes[0].set_ylabel(r"clustering share of Var($\ln\mu$) [%]")
    axes[0].set_title(f"$z_s={zs:g}$: field (8.44 Mpc) vs matched no-bias")
    axes[1].axhline(1.0, color="k", lw=0.6)
    axes[1].set_xlabel(r"$\ln\mu$")
    axes[1].set_ylabel(r"$P_{\rm field}/P_{\rm nobias}$ (matched $\langle N\rangle$)")
    axes[1].set_title("clustering signature vs $\\langle N\\rangle$")
    axes[1].set_ylim(0.75, 1.25)
    axes[1].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(REPO / "plots" / "clustering_vs_N.png", dpi=150)
    print("wrote plots/clustering_vs_N.png")
    for N, sh, (sn, sf) in zip(NLIST, share, sigs):
        print(f"<N>={N:5d}: sigma nobias {sn:.4f} field {sf:.4f} "
              f"-> clustering share {sh:.1f}%")


def main():
    ensure_all()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_zoom(plt, matplotlib)
    fig_N(plt, matplotlib)


if __name__ == "__main__":
    main()
