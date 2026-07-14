"""Regenerate the 9 anomalous shards found 2026-07-13 and diagnose (Mac only).

v2 (2026-07-13, second pass): the v1 labels were too crude — it called any
bitwise match a "live bug" and mislabeled FP-level reproduction as stale
cache. This version prints full diagnostics and classifies on two separate
questions:

  Q1  Does the current build reproduce the stored shard?
      (bitwise, or to FP noise: same body, same mean)
  Q2  Is the shard's CONTENT actually anomalous in the body, or is its
      flagged mean just heavy-tail seed luck?
      (body = median + KS vs sibling shards; the original 5-sigma flag used
      median shard std, which OVER-FLAGS tail-hit shards, esp. z=0.2)

Outcomes:
  REPRODUCES + body OK        -> false alarm of the flagging heuristic
                                 (tail fluke); keep the shard, unflag it.
  REPRODUCES + body shifted   -> LIVE seed-dependent bug in the sampler;
                                 do not repair, debug the generator.
  REGEN HEALTHY (matches sibs)-> stale-cache artifact; --fix splices the
                                 regenerated shard into <key>.cleanfix.npy.
  OTHER                       -> regen matches neither; inspect by hand.

Run from the repo root in the `test` env:
  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/verify_flagged_shards.py [--fix]
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build"))

spec = importlib.util.spec_from_file_location(
    "cscan", REPO / "scripts" / "convergence" / "convergence_scan.py")
cscan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cscan)

FLAGGED = [
    ("mmin_convergence",    "mmin",    0.2,  "main", "truthA",   0),
    ("mmin_convergence",    "mmin",    10.0, "main", "nm200ctl", 3),
    ("mmin_pd_convergence", "mmin_pd", 1.0,  "main", "m1e5",     1),
    ("mmin_pd_convergence", "mmin_pd", 1.0,  "main", "truthA",   0),
    ("mmin_pd_convergence", "mmin_pd", 1.0,  "main", "truthB",   7),
    ("mmin_pd_convergence", "mmin_pd", 5.0,  "main", "m1e5",     7),
    ("mmin_pd_convergence", "mmin_pd", 5.0,  "main", "truthB",   4),
    ("nz_convergence",      "nz",      0.2,  "main", "truthB",   5),
    ("nz_convergence",      "nz",      1.0,  "main", "nz50",     2),
]


def ks(a, b):
    """Two-sample KS statistic (no scipy dependency)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b]); allv.sort()
    ca = np.searchsorted(a, allv, side="right") / a.size
    cb = np.searchsorted(b, allv, side="right") / b.size
    return float(np.abs(ca - cb).max())


def stat(x):
    x = x[np.isfinite(x)]
    return x.mean(), np.median(x), x.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    import gwlensing as gw

    print(f"{'shard':44s} {'mean':>9s} {'median':>9s} {'std':>8s}")
    for study, axis, z, arm, config, shard in FLAGGED:
        key = f"z{z:g}_{arm}_{config}"
        f = REPO / "data" / "results" / study / f"{key}.npy"
        full = np.load(f)
        block, kwargs = cscan.CONFIGS[(axis, arm)][config]
        per = cscan.n_for(z, config, 1.0) // cscan.NPROC
        seed = cscan.seed_base(axis, z, arm, block) + shard
        stored = full[shard * per:(shard + 1) * per]
        sib = np.concatenate([full[:shard * per], full[(shard + 1) * per:]])

        d = gw.sample_lnmu_ml_with_diagnostics(
            z=z, h=cscan.H, OmegaM=cscan.OM, sigma8=cscan.S8, nsamples=per,
            seed=seed, strict_weak_lensing=False, subhalo=(arm == "sub"),
            subhalo_model=cscan.SUBHALO_MODEL,
            subhalo_factor=cscan.SUBHALO_FACTOR, **kwargs)
        new = np.asarray(d["lnmu"])

        sm, sq, ss = stat(stored); nm_, nq, ns_ = stat(new)
        bm, bq, bs = stat(sib)
        bitwise = np.array_equal(new, stored, equal_nan=True)
        ks_sn = ks(stored[np.isfinite(stored)], new[np.isfinite(new)])
        ks_ns = ks(new[np.isfinite(new)], sib[np.isfinite(sib)])
        ks_ss = ks(stored[np.isfinite(stored)], sib[np.isfinite(sib)])

        name = f"{study}/{key} s{shard} (seed {seed})"
        print(f"\n{name}")
        print(f"  stored   {sm:+9.5f} {sq:+9.5f} {ss:8.5f}")
        print(f"  regen    {nm_:+9.5f} {nq:+9.5f} {ns_:8.5f}   "
              f"bitwise={bitwise}  KS(stored,regen)={ks_sn:.4f}")
        print(f"  siblings {bm:+9.5f} {bq:+9.5f} {bs:8.5f}   "
              f"KS(regen,sib)={ks_ns:.4f}  KS(stored,sib)={ks_ss:.4f}")

        reproduces = bitwise or (ks_sn < 0.02 and abs(nm_ - sm) < 3 * ss /
                                 np.sqrt(per))
        # body anomaly: median shifted vs siblings by >5 sigma_median
        sig_med = 1.2533 * bs / np.sqrt(per)   # ~sqrt(pi/2)*std/sqrt(n)
        body_bad = abs(sq - bq) > 5 * sig_med or ks_ss > 0.03
        regen_healthy = ks_ns < 0.02 and abs(nq - bq) <= 5 * sig_med

        if reproduces and not body_bad:
            v = "VERDICT: reproduces, body consistent with siblings -> " \
                "heavy-tail seed fluke; FLAGGING FALSE ALARM, keep shard"
        elif reproduces and body_bad:
            v = "VERDICT: reproduces AND body-shifted -> LIVE seed-dependent " \
                "BUG; debug generator, do NOT splice"
        elif regen_healthy:
            v = "VERDICT: stale-cache artifact (regen healthy)"
            if args.fix:
                fixed = full.copy()
                fixed[shard * per:(shard + 1) * per] = new
                out = f.with_suffix(".cleanfix.npy")
                np.save(out, fixed)
                v += f"; wrote {out.name}"
        else:
            v = "VERDICT: regen matches neither stored nor siblings -> " \
                "inspect manually"
        print(f"  {v}")


if __name__ == "__main__":
    main()
