"""Bias-off arm of the Var(kappa)-vs-Nz sweep (2026-07-13).

Mechanism test for the unsaturated strong-tail growth with Nz: the BIAS
layer modulates per-(jz,jM)-cell Poisson means by an INDEPENDENT log-normal
whose amplitude sigma_b = sigma(M_b) uses the tube-SEGMENT mass
M_b = 2 pi rmax^2 [dc(zl)-dc(zl-dz)] rhoM0 (lensing.cpp:123) — the segment
length is the shell width, so refining Nz shrinks M_b and GROWS sigma_b,
while the draws stay white in z. Prediction: with bias=0 the f(kappa>1)
growth with Nz disappears.

z_s in {5, 10}, Nz in {100, 400, 1600}, 4 seeds x 50k, namespace 8.2e8.

Run (test env):
  /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/vark_vs_nz_nobias.py
Writes data/results/vark_nz/{shards_nobias/*.npy, vark_vs_nz_nobias.npz,
report_nobias.md}.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "vark_nz"
SHARDS = OUT / "shards_nobias"

H, OM, S8 = 0.674, 0.315, 0.811
ZS_LIST = [5.0, 10.0]
NZ_GRID = [100, 400, 1600]
NSEEDS = 4
N = 50_000
SEED0 = 820_000_000
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
                                   bias=False)
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
           "nseeds": NSEEDS, "n": N}
    lines = ["# Bias-off arm: Var(kappa) vs Nz\n",
             f"\nbias=0 (lambda=1: no per-cell log-normal modulation); "
             f"N={N} x {NSEEDS} seeds.\n",
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
    np.savez(OUT / "vark_vs_nz_nobias.npz", **out)
    (OUT / "report_nobias.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "vark_vs_nz_nobias.npz", "and report_nobias.md")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        run_shards()
        aggregate()
