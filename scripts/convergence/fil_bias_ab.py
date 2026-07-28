"""
A/B acceptance for the filament clustering bias (fil_bias) at the production config.

fil_bias switches the filament count modulation from the halo bias b(M,z) to the
PBS bias of the code's own filament barrier, filbias (p,q) = (0, 0.7), which is
10-20% lower. Filaments are subdominant to halos, so the expected P(lnmu) shift is
SMALL -- the design note (docs/filament_bias_note.md sec. 4) measured it below MC
noise at 70k samples. This script is what turns "below noise" into a number with a
calibrated floor, and it also logs the two quantities the emulator retrain cares
about: the low-mu EDGE and the FLUX normalization.

Statistics follow the standing rules in CLAUDE.md:
  - raw sd / mean / <1/mu> of lnmu are monster-ray junk -> clipped + trimmed only;
  - every difference is quoted against a same-model seed-split FLOOR, and only
    counts if it exceeds that floor;
  - sharded over subprocesses, never mp.Pool (which hides worker segfaults as
    silent hangs).

Both arms run the full production config (ml.params.PRODUCTION_CONFIG): subhalo
model 5 + carve, correlated field with the spherical top-hat at R_s = 20 Mpc, the
conditional weak arm, and the robust kappa anchor. The ONLY thing that differs
between arms is fil_bias.

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/fil_bias_ab.py --nray 30000 --nshard 8 --zs 0.5 1.0 5.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
PY = "/Users/baltabay/miniforge3/envs/test/bin/python"

# Kept as a literal (not an import of ml.params) so the shard source stays a
# standalone string; asserted equal to PRODUCTION_CONFIG in main().
SHARD_SRC = r"""
import sys, json, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, fil = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
kw = dict(subhalo=True, subhalo_model=5, subhalo_carve=True, m_floor=1e7,
          subhalo_kappathr_factor=0.1,
          bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True,
          fil_bias=bool(fil), kappa_anchor=1, kappa_anchor_cut=1.0)
# NOTE sample_lnmu positional order is (z, OmegaM, sigma8, h) -- h LAST.
x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, nray, seed, **kw))
np.save(sys.argv[5], x)
"""


def run_shards(tag, zs, nray, nshard, fil, seed0, outdir):
    src = outdir / "_shard.py"
    src.write_text(SHARD_SRC)
    procs, paths = [], []
    for s in range(nshard):
        p = outdir / f"{tag}_s{s}.npy"
        paths.append(p)
        if p.exists():
            continue
        procs.append(subprocess.Popen(
            [PY, str(src), str(zs), str(nray), str(seed0 + 1000 * s), str(int(fil)), str(p)],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for pr in procs:
        _, err = pr.communicate()
        if pr.returncode != 0:
            raise RuntimeError(f"{tag} shard failed: {err.decode()[-500:]}")
    return [np.load(p) for p in paths]


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


def clipped_sd(x, lo=-1.0, hi=1.0):
    x = x[np.isfinite(x)]
    m = (x > lo) & (x < hi)
    return float(x[m].std())


def sd_ratio_with_error(off_shards, on_shards):
    """Clipped-sd ratio with a shard-to-shard uncertainty, plus a within-arm null.

    The two arms cannot be paired ray-by-ray: switching fil_bias changes the
    filament counts and so reshuffles the shared RNG stream, which destroys
    common random numbers (the same effect that sank the cascade-residual
    experiment). So the ratio is a difference of two independent means and its
    error is the shard SEMs propagated. The within-arm 4-vs-4 null says how big
    a ratio this estimator produces from pure sampling noise.
    """
    from itertools import combinations

    so = np.array([clipped_sd(a) for a in off_shards])
    sn = np.array([clipped_sd(a) for a in on_shards])
    n = len(so)
    mo, mn = so.mean(), sn.mean()
    eo = so.std(ddof=1) / np.sqrt(n)
    en = sn.std(ddof=1) / np.sqrt(n)
    ratio = mn / mo
    err = ratio * np.sqrt((eo / mo) ** 2 + (en / mn) ** 2)

    nulls = []
    for arr in (so, sn):
        half = n // 2
        for c in combinations(range(n), half):
            if 0 not in c:          # each balanced partition once
                continue
            rest = np.array(sorted(set(range(n)) - set(c)))
            nulls.append(arr[rest].mean() / arr[np.array(c)].mean())
    null_sd = float(np.std(nulls, ddof=1))

    return dict(sd_off=float(mo), sd_on=float(mn), ratio=float(ratio),
                ratio_err=float(err), pct=float(100 * (ratio - 1)),
                pct_err=float(100 * err), null_pct=float(100 * null_sd),
                nsigma=float((ratio - 1) / err) if err > 0 else 0.0)


def summarize(x, lo=-1.0, hi=1.0):
    """Clipped body moments + the edge and flux diagnostics the emulator refits."""
    x = x[np.isfinite(x)]
    m = (x > lo) & (x < hi)
    body = x[m]
    # flux: ln<1/mu> on the SAME clipped support. The raw mean is corrupted by
    # kappa > 1 rays (see docs/edge_tail_flux_note.md) -- never target raw <1/mu>.
    ln_inv_mu = float(np.log(np.mean(np.exp(-body))))
    return dict(
        clip_mean=float(body.mean()), clip_sd=float(body.std()), n_clip=int(m.sum()),
        raw_sd=float(x.std()),
        q001=float(np.quantile(x, 0.001)), q01=float(np.quantile(x, 0.01)),
        q10=float(np.quantile(x, 0.10)),
        q99=float(np.quantile(x, 0.99)), q999=float(np.quantile(x, 0.999)),
        ln_inv_mu_trim=ln_inv_mu,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nray", type=int, default=30000, help="rays per shard")
    ap.add_argument("--nshard", type=int, default=8)
    ap.add_argument("--zs", type=float, nargs="+", default=[0.5, 1.0, 5.0])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # guard: the arms must be the pinned production config
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from ml.params import PRODUCTION_CONFIG
    shard_kw = dict(subhalo=True, subhalo_model=5, subhalo_carve=True, m_floor=1e7,
                    subhalo_kappathr_factor=0.1, bias_model=1, bias_window=1,
                    bias_Rperp=20000.0, bias_weak=True, kappa_anchor=1,
                    kappa_anchor_cut=1.0)
    assert shard_kw == {k: v for k, v in PRODUCTION_CONFIG.items() if k != "fil_bias"}, \
        "shard config drifted from ml.params.PRODUCTION_CONFIG"

    results = {}
    for zs in args.zs:
        # nray is part of the cache key: shards are reused when they already
        # exist, so a rerun at a different depth must not silently pick up the
        # shallow ones.
        outdir = ROOT / "tmp" / f"filbias_ab_zs{zs}_n{args.nray}"
        outdir.mkdir(parents=True, exist_ok=True)
        ntot = args.nray * args.nshard
        print(f"\n=== z_s = {zs}  ({args.nshard} shards x {args.nray} = {ntot} rays/arm) ===",
              flush=True)

        print("  arm OFF (filaments ride halobias) ...", flush=True)
        off = run_shards("off", zs, args.nray, args.nshard, False, 12345, outdir)
        print("  arm ON  (filaments ride filbias)  ...", flush=True)
        on = run_shards("on", zs, args.nray, args.nshard, True, 12345, outdir)

        a_off, a_on = np.concatenate(off), np.concatenate(on)
        edges = np.linspace(-0.6, 0.6, 161)

        r = {"zs": zs, "nray_total": int(a_off.size),
             "off": summarize(a_off), "on": summarize(a_on)}

        h = args.nshard // 2
        f_off = jsd(np.concatenate(off[:h]), np.concatenate(off[h:]), edges)
        f_on = jsd(np.concatenate(on[:h]), np.concatenate(on[h:]), edges)
        cross = jsd(a_off, a_on, edges)
        floor = max(f_off, f_on)
        sdr = sd_ratio_with_error(off, on)
        r.update(jsd_cross=cross, jsd_floor_off=f_off, jsd_floor_on=f_on,
                 jsd_ratio=cross / floor if floor > 0 else float("inf"),
                 sd_ratio=sdr["ratio"], sd_ratio_detail=sdr,
                 d_q01=r["on"]["q01"] - r["off"]["q01"],
                 d_ln_inv_mu=r["on"]["ln_inv_mu_trim"] - r["off"]["ln_inv_mu_trim"])
        r["verdict"] = "AT FLOOR" if cross <= floor else f"{cross/floor:.2f}x floor"
        r["sd_verdict"] = ("consistent with zero" if abs(sdr["nsigma"]) < 3
                           else f"{sdr['nsigma']:+.1f} sigma")

        print(f"  clipped sd : off {sdr['sd_off']:.6f}  on {sdr['sd_on']:.6f}")
        print(f"    ratio = {sdr['ratio']:.5f} +/- {sdr['ratio_err']:.5f}"
              f"  ({sdr['pct']:+.3f} +/- {sdr['pct_err']:.3f} %)"
              f"  -> {sdr['nsigma']:+.2f} sigma, {r['sd_verdict']}")
        print(f"    within-arm null ratio spread = +/- {sdr['null_pct']:.3f} %")
        print(f"  JSD(off,on) = {cross:.3e}   floors {f_off:.3e} / {f_on:.3e}"
              f"   -> {r['verdict']}")
        print(f"  edge q01 shift    = {r['d_q01']:+.2e}   "
              f"(bias_weak arm moved it -1.4e-2 for scale)")
        print(f"  flux dln<1/mu>    = {r['d_ln_inv_mu']:+.2e}")
        results[str(zs)] = r

    # keyed by depth so a deeper rerun does not clobber a shallower sweep
    out = ROOT / "data" / "results" / "fil_bias" / f"ab_n{args.nray}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
