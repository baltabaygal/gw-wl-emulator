"""Total Var(kappa) vs Nz — raw-kappa MC sweep (2026-07-13).

Companion to the Nz JSD study: the JSD arm shows P(lnmu) is converged at
Nz=100; this measures the TOTAL kappa variance response directly (the
analytic arm only covers the background piece sigma_W^2). Full model ON,
default fixed-<N>=100 rule, subhalo off, `sample_lensing_raw_ml(..., Nz=nz)`.

Clip convention mirrors the JSD protocol: common per-z_s kappa support from
the pooled truth run (Nz=400), quantile [1e-4, 1-1e-4]; central moments per
seed after clipping (raw moments are single-ray garbage: one kappa>>1 ray
dominates <k^2> at these N).

Sharding: plain subprocess workers (NOT mp.Pool — it hides worker segfaults
as silent hangs), 6-way, resumable via the shard cache.

Run (test env):
  /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/vark_vs_nz.py
Writes data/results/vark_nz/{shards/*.npy, vark_vs_nz.npz, report.md}.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "vark_nz"
SHARDS = OUT / "shards"

H, OM, S8 = 0.674, 0.315, 0.811
ZS_LIST = [0.2, 1.0, 5.0, 10.0]
NZ_GRID = [25, 50, 100, 200, 400, 800, 1600]   # 400 = the JSD study's truth;
# 800/1600 appended 2026-07-13 (variance staircase not saturated at 400).
# APPEND ONLY: seeds key on the grid index, so reordering breaks the cache.
NSEEDS = 4
N = 50_000
SEED0 = 800_000_000                    # disjoint namespace (mmin 3e8, nz 4e8,
                                       # mmin_pd 5e8, kthr 6e8, ACE 7e8)
QCLIP = 1e-4                           # common-support quantiles from truth
FIXED_CLIPS = [0.5, 1.0]               # fixed |kappa| supports (2026-07-13:
# the quantile clip floats with the deepest grid's own strong-lensing tail,
# which grows with Nz without saturation — fixed supports separate the
# converging body from the unconverged kappa>1 population)
MAXPROC = 6


def shard_path(z, nz, iseed):
    return SHARDS / f"z{z:g}_nz{nz}_s{iseed}.npy"


def worker(z, nz, iseed):
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    seed = (SEED0 + 1_000_000 * ZS_LIST.index(z)
            + 10_000 * NZ_GRID.index(nz) + iseed)
    raw = gw.sample_lensing_raw_ml(z, H, OM, S8, N, seed=seed, Nz=nz)
    k = np.asarray(raw["kappa"], dtype=np.float64)
    np.save(shard_path(z, nz, iseed), k)


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
           "nseeds": NSEEDS, "n": N, "qclip": QCLIP}
    nz_ref = NZ_GRID[-1]
    lines = ["# Total Var(kappa) vs Nz (raw-kappa MC, clipped central "
             "moments)\n",
             f"\nN={N} x {NSEEDS} seeds per config; clip = deepest grid "
             f"(Nz={nz_ref}) quantiles [{QCLIP:g}, {1-QCLIP:g}] per z_s, "
             "common to all Nz.\n",
             f"\n| z_s | Nz | <k^2>_clip mean +- sd | ratio to Nz={nz_ref} |"
             + "".join(f" ratio, \\|k\\|<{b:g} |" for b in FIXED_CLIPS)
             + " f(k>1) |",
             "|--:|--:|:--|--:|" + "--:|" * (len(FIXED_CLIPS) + 1)]
    for z in ZS_LIST:
        tr = np.concatenate([np.load(shard_path(z, nz_ref, i))
                             for i in range(NSEEDS)])
        klo, khi = np.quantile(tr, [QCLIP, 1 - QCLIP])
        out[f"clip_z{z:g}"] = np.array([klo, khi])
        means = np.empty((len(NZ_GRID), NSEEDS))
        fixed = {b: np.empty((len(NZ_GRID), NSEEDS)) for b in FIXED_CLIPS}
        frac1 = np.empty((len(NZ_GRID), NSEEDS))
        for j, nz in enumerate(NZ_GRID):
            for i in range(NSEEDS):
                k = np.load(shard_path(z, nz, i))
                kc = k[(k >= klo) & (k <= khi)]
                kc = kc - kc.mean()
                means[j, i] = (kc ** 2).mean()
                for b in FIXED_CLIPS:
                    x = k[np.abs(k) < b]
                    x = x - x.mean()
                    fixed[b][j, i] = (x ** 2).mean()
                frac1[j, i] = (k > 1).mean()
        out[f"K2clip_z{z:g}"] = means
        for b in FIXED_CLIPS:
            out[f"K2fix{b:g}_z{z:g}"] = fixed[b]
        out[f"frac_kgt1_z{z:g}"] = frac1
        ref = means[-1].mean()
        for j, nz in enumerate(NZ_GRID):
            m, s = means[j].mean(), means[j].std()
            fx = "".join(f" {fixed[b][j].mean()/fixed[b][-1].mean():.4f} |"
                         for b in FIXED_CLIPS)
            lines.append(f"| {z:g} | {nz} | {m:.4e} +- {s:.1e} | "
                         f"{m/ref:.4f} |{fx} {frac1[j].mean():.2e} |")
    np.savez(OUT / "vark_vs_nz.npz", **out)
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "vark_vs_nz.npz", "and report.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        run_shards()
        aggregate()
