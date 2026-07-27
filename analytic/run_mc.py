#!/usr/bin/env python
"""Halo-only C++ sGL Monte Carlo -> npz, for the analytic comparison.

Runs the production engine in the analytic chain's model scope:
NFW halos only -- filaments OFF, bias OFF, ellipticity OFF, subhalos OFF --
and stores the RAW per-ray (kappa, gamma1, gamma2) so the mean-kappa anchor
and the image/source-plane weighting can be applied downstream rather than
being baked in.

Sharded over seeds in subprocesses (mp.Pool masks worker segfaults as silent
hangs in this codebase; see CLAUDE.md item 12).

Usage:
  python analytic/run_mc.py --zs 1.0 --nray 200000 --shards 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data"

CFG = dict(filaments=False, bias=False, ell=False, subhalo=False,
           Mmin=1e7, NM=100, Nz=100)


def _shard(zs, nray, seed, h, Om, s8, path, Nhalos=100):
    sys.path.insert(0, str(ROOT / "build"))
    import gwlensing as gw
    raw = gw.sample_lensing_raw_ml(z=zs, h=h, OmegaM=Om, sigma8=s8,
                                   nsamples=nray, seed=seed,
                                   Nhalos=Nhalos, **CFG)
    np.savez_compressed(path,
                        kappa=np.asarray(raw["kappa"], float),
                        gamma1=np.asarray(raw["gamma1"], float),
                        gamma2=np.asarray(raw["gamma2"], float),
                        zs=zs, seed=seed, nray=nray, h=h, Om=Om, s8=s8)


def run(zs, nray, shards, h=0.674, Om=0.315, s8=0.811, seed0=1000, tag="",
        Nhalos=100):
    OUT.mkdir(parents=True, exist_ok=True)
    per = nray // shards
    parts = []
    for i in range(shards):
        p = OUT / f"_shard_zs{zs:g}{tag}_{i}.npz"
        parts.append(p)
    procs = []
    for i, p in enumerate(parts):
        cmd = [sys.executable, __file__, "--worker", "--zs", str(zs),
               "--nray", str(per), "--seed", str(seed0 + i),
               "--h", str(h), "--Om", str(Om), "--s8", str(s8),
               "--Nhalos", str(Nhalos), "--path", str(p)]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    fail = [i for i, q in enumerate(procs) if q.wait() != 0]
    if fail:
        raise RuntimeError(f"MC shards failed: {fail}")

    k = np.concatenate([np.load(p)["kappa"] for p in parts])
    g1 = np.concatenate([np.load(p)["gamma1"] for p in parts])
    g2 = np.concatenate([np.load(p)["gamma2"] for p in parts])
    dest = OUT / f"mc_halo_only_zs{zs:g}{tag}.npz"
    np.savez_compressed(dest, kappa=k, gamma1=g1, gamma2=g2, zs=zs,
                        nray=k.size, h=h, Om=Om, s8=s8, Nhalos=Nhalos,
                        config=str(CFG), seeds=str([seed0 + i
                                                    for i in range(shards)]))
    for p in parts:
        os.remove(p)
    print(f"[mc] zs={zs}  {k.size} rays -> {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, default=1.0)
    ap.add_argument("--nray", type=int, default=200_000)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--h", type=float, default=0.674)
    ap.add_argument("--Om", type=float, default=0.315)
    ap.add_argument("--s8", type=float, default=0.811)
    ap.add_argument("--tag", default="")
    ap.add_argument("--Nhalos", type=int, default=100)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--path", default="")
    a = ap.parse_args()
    if a.worker:
        _shard(a.zs, a.nray, a.seed, a.h, a.Om, a.s8, a.path, a.Nhalos)
    else:
        run(a.zs, a.nray, a.shards, a.h, a.Om, a.s8, a.seed, a.tag, a.Nhalos)


if __name__ == "__main__":
    main()
