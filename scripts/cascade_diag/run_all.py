#!/usr/bin/env python
"""Orchestrate the cascade-residual diagnostic dataset.

Launches gen_worker.py as parallel SUBPROCESS shards (avoids the mp.Pool
silent-segfault-hang gotcha noted in CLAUDE.md).  Writes one npz per (config,
arm) into DATADIR and an index.json describing the design.

Env: `test` conda (python 3.12), run from repo root.
"""
import os, sys, json, subprocess, itertools, time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
WORKER = os.path.join("scripts", "cascade_diag", "gen_worker.py")
DATADIR = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
           "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
os.makedirs(DATADIR, exist_ok=True)

FID = dict(Om=0.315, s8=0.811, h=0.674)
ZS_LIST = [0.5, 1.0, 5.0]
N_GRID = 200_000
N_NOISE = 200_000
GRID_SEED = 20260721
NSEED = 8              # seeds for the noise-floor block
NPAR = 8              # parallel workers

import numpy as np
AXES = {
    "Om": np.linspace(0.26, 0.38, 7),
    "s8": np.linspace(0.72, 0.92, 7),
    "h":  np.linspace(0.61, 0.74, 7),
}

# ---- build task list -------------------------------------------------------
# task = (tag, zs, Om, s8, h, N, seed, sub)
tasks = []
index = {"fid": FID, "zs_list": ZS_LIST, "n_grid": N_GRID, "n_noise": N_NOISE,
         "grid_seed": GRID_SEED, "nseed": NSEED, "axes": {k: v.tolist() for k, v in AXES.items()},
         "grid": [], "noise": []}

for axis, vals in AXES.items():
    for i, v in enumerate(vals):
        p = dict(FID)
        p[axis] = float(v)
        for zs in ZS_LIST:
            for sub in (0, 1):
                tag = f"grid_{axis}_{i}_zs{zs}_sub{sub}"
                tasks.append((tag, zs, p["Om"], p["s8"], p["h"], N_GRID, GRID_SEED, sub))
                index["grid"].append(dict(tag=tag, axis=axis, ipt=i, val=float(v),
                                          zs=zs, Om=p["Om"], s8=p["s8"], h=p["h"], sub=sub))

for zs in ZS_LIST:
    for s in range(1, NSEED + 1):
        for sub in (0, 1):
            tag = f"noise_zs{zs}_seed{s}_sub{sub}"
            tasks.append((tag, zs, FID["Om"], FID["s8"], FID["h"], N_NOISE, s, sub))
            index["noise"].append(dict(tag=tag, zs=zs, seed=s, sub=sub,
                                       Om=FID["Om"], s8=FID["s8"], h=FID["h"]))

with open(os.path.join(DATADIR, "index.json"), "w") as f:
    json.dump(index, f, indent=1)

print(f"{len(tasks)} tasks -> {DATADIR} ({NPAR}-way)", flush=True)


def run(t):
    tag, zs, Om, s8, h, N, seed, sub = t
    out = os.path.join(DATADIR, tag + ".npz")
    if os.path.exists(out):
        return tag + " (cached)"
    cmd = [PY, WORKER, out, str(zs), str(Om), str(s8), str(h), str(N), str(seed), str(sub)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return f"FAIL {tag}: {r.stderr[-300:]}"
    return tag + " ok"


t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=NPAR) as ex:
    for msg in ex.map(run, tasks):
        done += 1
        print(f"[{done}/{len(tasks)} {time.time()-t0:6.0f}s] {msg}", flush=True)
print("ALL DONE", time.time() - t0, "s")
