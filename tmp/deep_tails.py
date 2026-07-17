"""Deep-tail extension at z_s = 1 (2026-07-16): window arms {nobias, 8.44, 10,
18.2 Mpc} -> 320 shards = 4.8M realizations each; <N>-study arms
{nob,f844}_N{300,1000} -> 160 shards = 2.4M each. Cache-consistency verified
bitwise against the current module before this run.

Figures (survival functions = binning-free maximum tail resolution; bands =
seed-half spread):
  plots/deep_tail_window.png  S(x)=P(lnmu > x) for the window arms + ratio to nobias
  plots/deep_tail_N.png       field-vs-nobias survival pairs at <N>=100/300/1000

Run: $PY tmp/deep_tails.py
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
scan_kw = {name: kw for name, kw in scan.arms()}
scan_ai = {name: i for i, (name, _) in enumerate(scan.arms())}
R844 = scan_kw["m1e14"]["bias_Rperp"]

# tag -> (arm_idx, kwargs, target shards)
TARGETS = {
    "nobias":     (scan_ai["nobias"], scan_kw["nobias"], 320),
    "m1e14":      (scan_ai["m1e14"], scan_kw["m1e14"], 320),
    "r10Mpc":     (90, dict(bias_model=1, bias_Rperp=10000.0), 320),
    "m1e15":      (scan_ai["m1e15"], scan_kw["m1e15"], 320),
    "nob_N300":   (91, dict(bias=False, Nhalos=300), 160),
    "f844_N300":  (92, dict(bias_model=1, bias_Rperp=R844, Nhalos=300), 160),
    "nob_N1000":  (93, dict(bias=False, Nhalos=1000), 160),
    "f844_N1000": (94, dict(bias_model=1, bias_Rperp=R844, Nhalos=1000), 160),
}


def ensure(jobs=8):
    todo = []
    for tag, (ai, kw, nsh) in TARGETS.items():
        for s in range(nsh):
            if not scan.shard_path(ZS, tag, s).exists():
                todo.append([sys.executable,
                             str(REPO / "scripts" / "convergence" / "rperp_pdf_scan.py"),
                             "shard", "--zs", str(ZS),
                             "--seed", str(scan.shard_seed(ZS, ai, s)),
                             "--n", str(scan.NPERSHARD),
                             "--out", str(scan.shard_path(ZS, tag, s)),
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
                if done % 64 == 0:
                    print(f"[{done}] ({time.time()-t0:.0f}s)", flush=True)
        running = alive
        time.sleep(0.3)
    print(f"done in {time.time()-t0:.0f}s")


def load(tag, half=None):
    nsh = TARGETS[tag][2]
    ss = {None: range(nsh), "A": range(0, nsh, 2), "B": range(1, nsh, 2)}[half]
    return np.sort(np.concatenate([np.load(scan.shard_path(ZS, tag, s))["lnmu"]
                                   for s in ss]).astype(np.float64))


def surv(sorted_x, grid):
    return (len(sorted_x) - np.searchsorted(sorted_x, grid, side="right")) \
        / float(len(sorted_x))


def surv_with_band(tag, grid):
    s = surv(load(tag), grid)
    sA, sB = surv(load(tag, "A"), grid), surv(load(tag, "B"), grid)
    return s, 0.5 * np.abs(sA - sB)


def main():
    ensure()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- window arms
    arms = [("nobias", "no bias", "#e34948", ":"),
            ("m1e15", r"$R_\perp=18.2$ Mpc", "#b8860b", "-"),
            ("r10Mpc", r"$R_\perp=10$ Mpc", "#1baf7a", "-"),
            ("m1e14", r"$R_\perp=8.44$ Mpc", "#2a78d6", "-")]
    grid = np.linspace(0.0, 2.6, 400)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    Sn, Bn = surv_with_band("nobias", grid)
    for tag, lab, col, ls in arms:
        S, B = surv_with_band(tag, grid)
        ax[0].semilogy(grid, S, color=col, ls=ls, lw=1.3, label=lab)
        ax[0].fill_between(grid, np.maximum(S - B, 1e-12), S + B,
                           color=col, alpha=0.18, lw=0)
        if tag != "nobias":
            ok = Sn > 0
            r = np.divide(S, Sn, out=np.full_like(S, np.nan), where=ok)
            re = r * np.sqrt(np.divide(B, S, out=np.zeros_like(S), where=S > 0) ** 2
                             + np.divide(Bn, Sn, out=np.zeros_like(Sn), where=ok) ** 2)
            ax[1].plot(grid, r, color=col, ls=ls, lw=1.3, label=lab)
            ax[1].fill_between(grid, r - re, r + re, color=col, alpha=0.18, lw=0)
    ax[0].set_xlabel(r"$\ln\mu$"); ax[0].set_ylabel(r"$S(\ln\mu) = P(>\ln\mu)$")
    ax[0].set_ylim(2e-7, 1.0); ax[0].legend(fontsize=9, frameon=False)
    ax[0].set_title(f"$z_s={ZS:g}$: survival, 4.8M/arm")
    ax[1].axhline(1.0, color="k", lw=0.7)
    ax[1].set_xlabel(r"$\ln\mu$"); ax[1].set_ylabel(r"$S/S_{\rm nobias}$")
    ax[1].set_ylim(0.5, 2.5); ax[1].legend(fontsize=9, frameon=False)
    ax[1].set_title("tail enhancement vs no bias")
    fig.tight_layout()
    fig.savefig(REPO / "plots" / "deep_tail_window.png", dpi=150)
    print("wrote plots/deep_tail_window.png")

    # ---------------- <N> pairs
    pairs = [(100, "nobias", "m1e14", "#2a78d6"),
             (300, "nob_N300", "f844_N300", "#1baf7a"),
             (1000, "nob_N1000", "f844_N1000", "#b8860b")]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for N, nob_t, fld_t, col in pairs:
        Sf, Bf = surv_with_band(fld_t, grid)
        Sn2, Bn2 = surv_with_band(nob_t, grid)
        ax[0].semilogy(grid, Sf, color=col, lw=1.3,
                       label=f"field, $\\langle N\\rangle={N}$")
        ax[0].semilogy(grid, Sn2, color=col, lw=1.1, ls=":",
                       label=f"no bias, $\\langle N\\rangle={N}$")
        ok = Sn2 > 0
        r = np.divide(Sf, Sn2, out=np.full_like(Sf, np.nan), where=ok)
        re = r * np.sqrt(np.divide(Bf, Sf, out=np.zeros_like(Sf), where=Sf > 0) ** 2
                         + np.divide(Bn2, Sn2, out=np.zeros_like(Sn2), where=ok) ** 2)
        ax[1].plot(grid, r, color=col, lw=1.3, label=f"$\\langle N\\rangle={N}$")
        ax[1].fill_between(grid, r - re, r + re, color=col, alpha=0.18, lw=0)
    ax[0].set_xlabel(r"$\ln\mu$"); ax[0].set_ylabel(r"$S(\ln\mu)$")
    ax[0].set_ylim(2e-7, 1.0); ax[0].legend(fontsize=8.5, frameon=False)
    ax[0].set_title(f"$z_s={ZS:g}$: survival, field (8.44 Mpc) vs matched no-bias")
    ax[1].axhline(1.0, color="k", lw=0.7)
    ax[1].set_xlabel(r"$\ln\mu$"); ax[1].set_ylabel(r"$S_{\rm field}/S_{\rm nobias}$")
    ax[1].set_ylim(0.5, 3.0); ax[1].legend(fontsize=9, frameon=False)
    ax[1].set_title("tail enhancement vs $\\langle N\\rangle$ (matched pairs)")
    fig.tight_layout()
    fig.savefig(REPO / "plots" / "deep_tail_N.png", dpi=150)
    print("wrote plots/deep_tail_N.png")


if __name__ == "__main__":
    main()
