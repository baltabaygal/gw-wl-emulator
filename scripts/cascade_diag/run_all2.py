#!/usr/bin/env python
"""Generate the +bias and +full(sub+bias) arms for #2/#3, reusing the existing
base(sub0)/sub(sub1) shards and the SAME θ grid, z_s and seeds (from index.json).
Parallel subprocess shards. Env: `test`, from repo root."""
import os, sys, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
WORKER = os.path.join("scripts", "cascade_diag", "gen_worker2.py")
DATADIR = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
           "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
NPAR = 8
NEW_ARMS = ["bias", "full"]

idx = json.load(open(os.path.join(DATADIR, "index.json")))

tasks = []  # (out, zs, Om, s8, h, N, seed, arm)
for g in idx["grid"]:
    if g["sub"] != 0:      # one entry per θ-config (sub0/sub1 are the same config)
        continue
    for arm in NEW_ARMS:
        tag = f"grid_{g['axis']}_{g['ipt']}_zs{g['zs']}_{arm}"
        tasks.append((os.path.join(DATADIR, tag + ".npz"), g["zs"], g["Om"], g["s8"],
                      g["h"], idx["n_grid"], idx["grid_seed"], arm))
for nz in idx["noise"]:
    if nz["sub"] != 0:
        continue
    for arm in NEW_ARMS:
        tag = f"noise_zs{nz['zs']}_seed{nz['seed']}_{arm}"
        tasks.append((os.path.join(DATADIR, tag + ".npz"), nz["zs"], nz["Om"], nz["s8"],
                      nz["h"], idx["n_noise"], nz["seed"], arm))

print(f"{len(tasks)} new-arm tasks ({NPAR}-way)", flush=True)


def run(t):
    out = t[0]
    if os.path.exists(out):
        return os.path.basename(out) + " (cached)"
    cmd = [PY, WORKER, out] + [str(a) for a in t[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return (f"FAIL {os.path.basename(out)}: {r.stderr[-300:]}" if r.returncode
            else os.path.basename(out) + " ok")


t0 = time.time(); done = 0
with ThreadPoolExecutor(max_workers=NPAR) as ex:
    for msg in ex.map(run, tasks):
        done += 1
        print(f"[{done}/{len(tasks)} {time.time()-t0:6.0f}s] {msg}", flush=True)
print("ALL DONE", time.time() - t0, "s")
