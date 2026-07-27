import os, sys, subprocess, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
PY = sys.executable
W = "scripts/cascade_diag/raw_worker.py"
D = ("/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/"
     "ac612839-196b-4778-8b02-eab1b572a0e1/scratchpad/cascade_data/raw")
os.makedirs(D, exist_ok=True)
ARMS = ["base", "sub", "bias", "full"]; SEEDS = range(11, 21); ZS = [1.0, 5.0]; N = 100000
tasks = [(os.path.join(D, f"z{zs}_{a}_s{s}.npz"), zs, a, s) for zs in ZS for a in ARMS for s in SEEDS]
def run(t):
    out, zs, a, s = t
    if os.path.exists(out): return "cached"
    r = subprocess.run([PY, W, out, str(zs), a, str(s), str(N)], capture_output=True, text=True)
    return f"FAIL {os.path.basename(out)} {r.stderr[-200:]}" if r.returncode else os.path.basename(out)
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, m in enumerate(ex.map(run, tasks)):
        print(f"[{i+1}/{len(tasks)} {time.time()-t0:.0f}s] {m}", flush=True)
print("ALL DONE", time.time() - t0)
