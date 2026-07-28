"""
Acceptance gate for subhalo_model=5 (kappa-thresholded brute) against model 4 (full brute).

Model 5 must reproduce model 4's P(lnmu) to within the threshold's certified budget:
the population-weighted sweep (data/results/subkappathr_population/report.md) predicts a
0.08-0.21% loss in the SUBSTRUCTURE part of sigma_kappa at subhalo_kappathr_factor=0.1,
which is far below the sampling floor here -- so the gate is really "is model 5
statistically indistinguishable from model 4 at the precision we can afford".

Statistics follow the repo's standing rules (CLAUDE.md): raw sd/mean of lnmu are
monster-ray junk, so we compare CLIPPED moments and the JSD of P(lnmu) on a shared
binning, and we calibrate every number against a same-model seed-split FLOOR. A
difference only counts if it exceeds that floor.

Sharded over subprocesses (never mp.Pool -- it hides worker segfaults as silent hangs).

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/subhalo_model5_gate.py --nray 1500 --nshard 8 --zs 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
PY = "/Users/baltabay/miniforge3/envs/test/bin/python"

SHARD_SRC = r"""
import sys, json, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, model, factor = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
virial = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
kw = dict(subhalo=True, subhalo_carve=True, subhalo_model=model, kappa_anchor=1,
          subhalo_virial=virial)
if model == 5:
    kw["subhalo_kappathr_factor"] = factor
x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, nray, seed, **kw))
np.save(sys.argv[6], x)
"""


def run_shards(tag, zs, nray, nshard, model, factor, seed0, outdir, virial=False):
    src = outdir / "_shard.py"
    src.write_text(SHARD_SRC)
    procs, paths = [], []
    for s in range(nshard):
        p = outdir / f"{tag}_s{s}.npy"
        paths.append(p)
        if p.exists():
            continue
        procs.append(subprocess.Popen(
            [PY, str(src), str(zs), str(nray), str(seed0 + 1000 * s), str(model), str(factor),
             str(p), str(int(virial))],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for pr in procs:
        _, err = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"{tag} shard failed: {err.decode()[-400:]}")
    return [np.load(p) for p in paths]


def clipped_moments(x, lo=-1.0, hi=1.0):
    m = (x > lo) & (x < hi)
    return float(x[m].mean()), float(x[m].std()), int(m.sum())


def jsd(a, b, edges):
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    m = 0.5 * (pa + pb)
    def kl(p, q):
        s = p > 0
        return float(np.sum(p[s] * np.log(p[s] / np.maximum(q[s], 1e-300))))
    return 0.5 * kl(pa, m) + 0.5 * kl(pb, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nray", type=int, default=1500)
    ap.add_argument("--nshard", type=int, default=8)
    ap.add_argument("--zs", type=float, default=1.0)
    ap.add_argument("--factor", type=float, default=0.1)
    ap.add_argument("--virial", action="store_true",
                    help="run BOTH arms with subhalo_virial=True (validates the model-5 "
                         "restricted-intensity tables under the virial convention)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    tag_v = "_virial" if args.virial else ""
    outdir = Path(args.outdir) if args.outdir else \
        ROOT / "tmp" / f"m5gate_zs{args.zs}_f{args.factor}{tag_v}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"z_s={args.zs}  {args.nshard} shards x {args.nray} rays  factor={args.factor}")
    print("running model 4 (full brute, ~45 ms/ray) ...", flush=True)
    m4 = run_shards("m4", args.zs, args.nray, args.nshard, 4, 0.0, 12345, outdir, args.virial)
    print("running model 5 (thresholded) ...", flush=True)
    m5 = run_shards("m5", args.zs, args.nray, args.nshard, 5, args.factor, 12345, outdir, args.virial)

    a4, a5 = np.concatenate(m4), np.concatenate(m5)
    edges = np.linspace(-0.35, 0.35, 121)

    r = {"zs": args.zs, "factor": args.factor, "nray_total": int(a4.size),
         "subhalo_virial": bool(args.virial)}
    for tag, a in (("model4", a4), ("model5", a5)):
        mu, sd, n = clipped_moments(a)
        r[tag] = {"clip_mean": mu, "clip_sd": sd, "n_clip": n, "raw_sd": float(a.std())}
        print(f"  {tag}: clipped mean={mu:+.6f} sd={sd:.6f} (n={n}) | raw sd={a.std():.5f}")

    # same-model seed-split floors: the resolution of this comparison
    h = args.nshard // 2
    f4 = jsd(np.concatenate(m4[:h]), np.concatenate(m4[h:]), edges)
    f5 = jsd(np.concatenate(m5[:h]), np.concatenate(m5[h:]), edges)
    cross = jsd(a4, a5, edges)
    r["jsd_cross"] = cross
    r["jsd_floor_m4"] = f4
    r["jsd_floor_m5"] = f5
    r["sd_ratio"] = r["model5"]["clip_sd"] / r["model4"]["clip_sd"]

    print(f"\n  JSD(model4, model5) = {cross:.3e}")
    print(f"  floors: model4 split {f4:.3e}   model5 split {f5:.3e}")
    print(f"  clipped sd ratio m5/m4 = {r['sd_ratio']:.5f} "
          f"(predicted loss at factor {args.factor}: see report)")
    verdict = "AT FLOOR" if cross <= max(f4, f5) else f"{cross/max(f4,f5):.2f}x floor"
    r["verdict"] = verdict
    print(f"  VERDICT: JSD is {verdict}")

    out = (ROOT / "data" / "results" / "subkappathr_population" /
           f"gate_zs{args.zs}_f{args.factor}{tag_v}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
