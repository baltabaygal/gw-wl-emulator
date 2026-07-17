"""Frozen-kappa_thr arm of the Var(kappa)-vs-Nz sweep (2026-07-13).

Tests the hypothesis that the unsaturated strong-tail growth with Nz
(`vark_vs_nz.py`) is driven by the fixed-<N>=100 rule's kappa_thr_eff
shifting with the grid: here kappathr_flat is PINNED per z_s to the default
rule's Nz=100 value, so the explicit/Gaussian split cannot move as Nz
refines. If f(kappa>1) still grows, the split is not the cause.

z_s in {5, 10} (the strong-effect cases), Nz in {100, 400, 1600},
4 seeds x 50k, seed namespace 8.1e8 (disjoint from the main arm's 8.0e8).

Run (test env):
  /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/vark_vs_nz_frozen.py
Writes data/results/vark_nz/{shards_frozen/*.npy, vark_vs_nz_frozen.npz,
report_frozen.md}.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "vark_nz"
SHARDS = OUT / "shards_frozen"

H, OM, S8 = 0.674, 0.315, 0.811
# default fixed-<N>=100 kappa_thr_eff at Nz=100 (mmin_convergence summary)
KTHR = {5.0: 0.000903763270798311, 10.0: 0.0013730783279736253}
ZS_LIST = [5.0, 10.0]
NZ_GRID = [100, 400, 1600]
NSEEDS = 4
N = 50_000
SEED0 = 810_000_000
FIXED_CLIPS = [0.5, 1.0]
MAXPROC = 6


def shard_path(z, nz, iseed):
    return SHARDS / f"z{z:g}_nz{nz}_s{iseed}.npy"


def worker(z, nz, iseed):
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    seed = (SEED0 + 1_000_000 * ZS_LIST.index(z)
            + 10_000 * NZ_GRID.index(nz) + iseed)
    raw = gw.sample_lensing_raw_ml(z, H, OM, S8, N, seed=seed, Nz=nz,
                                   kappathr_flat=KTHR[z])
    np.save(shard_path(z, nz, iseed),
            np.asarray(raw["kappa"], dtype=np.float64))


def run_shards():
    SHARDS.mkdir(parents=True, exist_ok=True)
    tasks = [(z, nz, i) for z in ZS_LIST for nz in NZ_GRID
             for i in range(NSEEDS) if not shard_path(z, nz, i).exists()]
    print(f"{len(tasks)} shards to run ({MAXPROC}-way)")
    procs = {}
    while tasks or procs:
        while tasks and len(procs) < MAXPROC:
            z, nz, i = tasks.pop(0)
            p = subprocess.Popen(
                [sys.executable, __file__, "--worker", f"{z:g}", str(nz),
                 str(i)])
            procs[p] = (z, nz, i)
        time.sleep(2)
        for p in list(procs):
            if p.poll() is None:
                continue
            z, nz, i = procs.pop(p)
            if p.returncode != 0:
                raise RuntimeError(f"shard z={z} nz={nz} s{i} exited "
                                   f"{p.returncode}")
            print(f"  done z={z:g} nz={nz} s{i}", flush=True)


def aggregate():
    out = {"nz_grid": np.array(NZ_GRID, float), "zs": np.array(ZS_LIST),
           "nseeds": NSEEDS, "n": N,
           "kthr": np.array([KTHR[z] for z in ZS_LIST])}
    lines = ["# Frozen-kappa_thr arm: Var(kappa) vs Nz\n",
             f"\nkappathr_flat pinned to the default rule's Nz=100 value "
             f"per z_s; N={N} x {NSEEDS} seeds.\n",
             "\n| z_s | Nz | ratio, \\|k\\|<0.5 | ratio, \\|k\\|<1 | "
             "f(k>1) mean +- sd |",
             "|--:|--:|--:|--:|:--|"]
    for z in ZS_LIST:
        fixed = {b: np.empty((len(NZ_GRID), NSEEDS)) for b in FIXED_CLIPS}
        frac1 = np.empty((len(NZ_GRID), NSEEDS))
        for j, nz in enumerate(NZ_GRID):
            for i in range(NSEEDS):
                k = np.load(shard_path(z, nz, i))
                for b in FIXED_CLIPS:
                    x = k[np.abs(k) < b]
                    x = x - x.mean()
                    fixed[b][j, i] = (x ** 2).mean()
                frac1[j, i] = (k > 1).mean()
        for b in FIXED_CLIPS:
            out[f"K2fix{b:g}_z{z:g}"] = fixed[b]
        out[f"frac_kgt1_z{z:g}"] = frac1
        for j, nz in enumerate(NZ_GRID):
            r = "".join(f" {fixed[b][j].mean()/fixed[b][-1].mean():.4f} |"
                        for b in FIXED_CLIPS)
            lines.append(f"| {z:g} | {nz} |{r} {frac1[j].mean():.2e} "
                         f"+- {frac1[j].std():.1e} |")
    np.savez(OUT / "vark_vs_nz_frozen.npz", **out)
    (OUT / "report_frozen.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "vark_vs_nz_frozen.npz", "and report_frozen.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        run_shards()
        aggregate()
