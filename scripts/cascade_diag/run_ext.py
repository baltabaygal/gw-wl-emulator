"""z_s extension for the mode analysis: base+sub arms at z_s∈{6,8,10} (+ base+bias
at z_s=6 spot check), on the SAME Om/σ8/h grid + a noise block, via gen_worker2.
Scientific-narrative only (z_s≳5 body/shoulder shape; NO tail-amplitude claims)."""
import os, sys, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
PY = sys.executable
W = "scripts/cascade_diag/gen_worker2.py"
D = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
     "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data")
idx = json.load(open(os.path.join(D, "index.json")))
FID, AXES = idx["fid"], idx["axes"]
GSEED, N, NSEED = idx["grid_seed"], idx["n_grid"], 6

tasks = []   # (out, zs, Om, s8, h, N, seed, arm)
def grid_tasks(zs, arms):
    for axis, vals in AXES.items():
        for i, v in enumerate(vals):
            p = dict(FID); p[axis] = float(v)
            for arm in arms:
                tasks.append((os.path.join(D, f"grid_{axis}_{i}_zs{zs}_{arm}.npz"),
                              zs, p["Om"], p["s8"], p["h"], N, GSEED, arm))
def noise_tasks(zs, arms):
    for s in range(1, NSEED + 1):
        for arm in arms:
            tasks.append((os.path.join(D, f"noise_zs{zs}_seed{s}_{arm}.npz"),
                          zs, FID["Om"], FID["s8"], FID["h"], N, s, arm))

for zs in (6.0, 8.0, 10.0):
    grid_tasks(zs, ["base", "sub"]); noise_tasks(zs, ["base", "sub"])
grid_tasks(6.0, ["bias"]); noise_tasks(6.0, ["bias"])   # z=6 clustering spot check

print(f"{len(tasks)} extension tasks", flush=True)
def run(t):
    out = t[0]
    if os.path.exists(out): return "cached"
    r = subprocess.run([PY, W, out] + [str(a) for a in t[1:]], capture_output=True, text=True)
    return f"FAIL {os.path.basename(out)} {r.stderr[-200:]}" if r.returncode else os.path.basename(out)
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, m in enumerate(ex.map(run, tasks)):
        print(f"[{i+1}/{len(tasks)} {time.time()-t0:.0f}s] {m}", flush=True)
print("ALL DONE", time.time() - t0)
