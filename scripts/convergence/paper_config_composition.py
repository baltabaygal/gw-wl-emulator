"""Measure the ASSEMBLED paper config against the legacy baseline at depth.

Every physics change of the last week was gated individually against its own
baseline (profile fix, subhalo_virial, fil_bias, bias_window, model 5, halobias).
Nothing had measured the whole stack at once, and composition is NOT guaranteed
additive here -- the cascade diagnostic (CLAUDE.md, `cascade_residual_diagnostic`)
found sigma/edge shifts SUB-additive and growing with z_s (~37% at z_s = 5). This
is also the number the draft's Fig. variance_DL is built from, so it needs a
measurement rather than a sum of parts.

Arms (both via ml.params, so they track the pinned configs):
  legacy = LEGACY_CONFIG      -- pre-2026-07-29 defaults ~ Vaskonen's model
  paper  = PRODUCTION_CONFIG  -- what the draft describes (= the C++ defaults now)

Protocol notes, learned the hard way and enforced here:
  * subprocess shards, never mp.Pool -- Pool hides worker segfaults as silent hangs.
  * clipped sd on a SHARED |lnmu| mask; raw moments are monster-ray junk.
  * the sd SEM is the ACROSS-SHARD spread, never 1/sqrt(2N): the clipped sd is
    heavy-tail dominated so its effective sample size is far below the ray count.
    Reading 1/sqrt(2N) turned a 0.6-sigma result into a fake 2.4 sigma once already.
  * 8-shard SEM is a chi^2 on 7 dof (~27% uncertain) and yields 2-3 sigma FALSE
    POSITIVES; require a depth-doubling or >=5 sigma before calling a ratio real.

Usage:
    $PY scripts/convergence/paper_config_composition.py --zs 1.0 --shards 8 --per 25000
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))

FID = dict(h=0.674, Om=0.315, sigma8=0.811)
CLIP = 1.0          # |lnmu| <= CLIP defines the certified body
OUT = ROOT / "data" / "results" / "paper_config_composition"


def _shard(zs, arm, per, seed, path):
    """One subprocess shard -> .npy of lnmu."""
    import gwlensing as gw
    from ml.params import LEGACY_CONFIG, PRODUCTION_CONFIG
    cfg = LEGACY_CONFIG if arm == "legacy" else PRODUCTION_CONFIG
    r = gw.sample_lnmu_ml_with_diagnostics(
        float(zs), FID["h"], FID["Om"], FID["sigma8"], int(per), int(seed), False, **cfg)
    x = np.asarray(r["lnmu"], dtype=np.float64)
    np.save(path, x[np.isfinite(x)])


def run(zs, shards, per, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in ("legacy", "paper"):
        for s in range(shards):
            p = outdir / f"{arm}_zs{zs}_s{s}.npy"
            if p.exists():
                continue
            jobs.append((arm, s, p))
    procs = []
    for arm, s, p in jobs:
        # seed is arm-INDEPENDENT so the two arms share the base stream where they can
        cmd = [sys.executable, __file__, "--worker", "--zs", str(zs), "--arm", arm,
               "--per", str(per), "--seed", str(7_000_000 + s), "--path", str(p)]
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT)))
        while sum(q.poll() is None for q in procs) >= 8:
            pass
    for q in procs:
        assert q.wait() == 0, "a shard FAILED -- do not use partial results"

    res = {}
    for arm in ("legacy", "paper"):
        res[arm] = [np.load(outdir / f"{arm}_zs{zs}_s{s}.npy") for s in range(shards)]

    # per-shard clipped sd on the SHARED mask definition
    sd = {a: np.array([x[np.abs(x) <= CLIP].std(ddof=1) for x in res[a]]) for a in res}
    flux = {a: np.array([np.mean(np.exp(-x)) for x in res[a]]) for a in res}
    ratio = sd["paper"].mean() / sd["legacy"].mean()
    # SEM of the ratio from the ACROSS-SHARD spread of both arms (not 1/sqrt(2N))
    rel = np.sqrt((sd["paper"].std(ddof=1) / np.sqrt(shards) / sd["paper"].mean()) ** 2 +
                  (sd["legacy"].std(ddof=1) / np.sqrt(shards) / sd["legacy"].mean()) ** 2)
    sig = abs(ratio - 1.0) / rel if rel else float("inf")

    out = dict(
        zs=zs, shards=shards, per_shard=per, rays_per_arm=shards * per, clip=CLIP,
        sd_legacy=float(sd["legacy"].mean()), sd_paper=float(sd["paper"].mean()),
        sd_ratio=float(ratio), sd_ratio_rel_sem=float(rel), sd_ratio_sigma=float(sig),
        pct_change=float(100 * (ratio - 1.0)),
        flux_legacy=float(flux["legacy"].mean()), flux_paper=float(flux["paper"].mean()),
        verdict=("RESOLVED" if sig >= 5 else "NOT RESOLVED at >=5 sigma"),
    )
    print(json.dumps(out, indent=2))
    (outdir / f"composition_zs{zs}.json").write_text(json.dumps(out, indent=2))
    print(f"\nsd(lnmu) clipped |lnmu|<={CLIP}:  legacy {out['sd_legacy']:.6f} "
          f"-> paper {out['sd_paper']:.6f}")
    print(f"change {out['pct_change']:+.3f} % +/- {100*rel:.3f} %  ({sig:.1f} sigma)  "
          f"{out['verdict']}")
    print(f"<1/mu>: legacy {out['flux_legacy']:.6f}  paper {out['flux_paper']:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, default=1.0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--per", type=int, default=25000)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--path")
    a = ap.parse_args()
    if a.worker:
        _shard(a.zs, a.arm, a.per, a.seed, a.path)
        return
    # ⚠ shards are CACHED by path: pass a FRESH --outdir after any physics change,
    # or stale .npy files from a previous config are silently reused.
    run(a.zs, a.shards, a.per, Path(a.outdir) if a.outdir else OUT)


if __name__ == "__main__":
    main()
