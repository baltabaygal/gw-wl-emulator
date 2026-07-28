"""
A/B for the JvdB14 virial convention (subhalo_virial) at the production config.

Legacy applies a virial-calibrated substructure population over an r_200 aperture:
JvdB14 defines f_s and psi = m/M inside R_vir, and Green+21 normalizes the radial
bias at r_vir, but the engine's M is M_200c and the profile was sampled only to
r_200. Only the bias SCALE x0 was converted (etaVirTo200). The analytic estimate of
the resulting excess substructure inside r_200 is

    net = (M_200/M_vir) / [N(<r_200)/N(<r_vir)]
        = 1.49 / 1.28 / 1.18 / 1.11 / 1.09   at z = 0.1 / 0.5 / 1 / 2 / 5

so switching the flag on should REDUCE the scatter, by more at low z_s. This script
measures that on P(lnmu) and reports the two quantities the emulator refits (edge,
flux), with shard-SEM error bars and same-arm seed-split floors, per the standing
rules in CLAUDE.md (clipped/trimmed statistics only; subprocess shards, never
mp.Pool).

Both arms use ml.params.PRODUCTION_CONFIG; only subhalo_virial differs.

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python \
      scripts/convergence/subhalo_virial_ab.py --nray 60000 --nshard 8 --zs 0.5 1.0 5.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
PY = "/Users/baltabay/miniforge3/envs/test/bin/python"

SHARD_SRC = r"""
import sys, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, vir = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
kw = dict(subhalo=True, subhalo_model=5, subhalo_carve=True, m_floor=1e7,
          subhalo_kappathr_factor=0.1,
          bias_model=1, bias_window=1, bias_Rperp=20000.0, bias_weak=True,
          fil_bias=True, kappa_anchor=1, kappa_anchor_cut=1.0,
          subhalo_virial=bool(vir))
# NOTE sample_lnmu positional order is (z, OmegaM, sigma8, h) -- h LAST.
x = np.asarray(gw.sample_lnmu(zs, 0.315, 0.811, 0.674, nray, seed, **kw))
np.save(sys.argv[5], x)
"""


def run_shards(tag, zs, nray, nshard, vir, seed0, outdir):
    src = outdir / "_shard.py"
    src.write_text(SHARD_SRC)
    procs, paths = [], []
    for s in range(nshard):
        p = outdir / f"{tag}_s{s}.npy"
        paths.append(p)
        if p.exists():
            continue
        procs.append(subprocess.Popen(
            [PY, str(src), str(zs), str(nray), str(seed0 + 1000 * s), str(int(vir)), str(p)],
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
    """Ratio of clipped sds with shard-SEM error, plus a within-arm null.

    The arms cannot be paired ray-by-ray: changing the population reshuffles the
    shared RNG stream, so common random numbers do not apply.
    """
    so = np.array([clipped_sd(a) for a in off_shards])
    sn = np.array([clipped_sd(a) for a in on_shards])
    n = len(so)
    mo, mn = so.mean(), sn.mean()
    eo, en = so.std(ddof=1) / np.sqrt(n), sn.std(ddof=1) / np.sqrt(n)
    ratio = mn / mo
    err = ratio * np.sqrt((eo / mo) ** 2 + (en / mn) ** 2)
    nulls = []
    for arr in (so, sn):
        for c in combinations(range(n), n // 2):
            if 0 not in c:
                continue
            rest = np.array(sorted(set(range(n)) - set(c)))
            nulls.append(arr[rest].mean() / arr[np.array(c)].mean())
    return dict(sd_off=float(mo), sd_on=float(mn), ratio=float(ratio),
                ratio_err=float(err), pct=float(100 * (ratio - 1)),
                pct_err=float(100 * err), null_pct=float(np.std(nulls, ddof=1) * 100),
                nsigma=float((ratio - 1) / err) if err > 0 else 0.0)


def summarize(x, lo=-1.0, hi=1.0):
    x = x[np.isfinite(x)]
    body = x[(x > lo) & (x < hi)]
    return dict(clip_mean=float(body.mean()), clip_sd=float(body.std()),
                n_clip=int(body.size), raw_sd=float(x.std()),
                q01=float(np.quantile(x, 0.01)), q10=float(np.quantile(x, 0.10)),
                q99=float(np.quantile(x, 0.99)), q999=float(np.quantile(x, 0.999)),
                ln_inv_mu_trim=float(np.log(np.mean(np.exp(-body)))))


# analytic prediction from the aperture mismatch (see report.md sec. 2)
PREDICTED_EXCESS = {0.5: 1.282, 1.0: 1.178, 5.0: 1.090}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nray", type=int, default=60000)
    ap.add_argument("--nshard", type=int, default=8)
    ap.add_argument("--zs", type=float, nargs="+", default=[0.5, 1.0, 5.0])
    args = ap.parse_args()

    results = {}
    for zs in args.zs:
        outdir = ROOT / "tmp" / f"subvirial_ab_zs{zs}_n{args.nray}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== z_s = {zs}  ({args.nshard} x {args.nray} = "
              f"{args.nshard*args.nray} rays/arm) ===", flush=True)
        print("  arm OFF (legacy: virial population in an r200 aperture) ...", flush=True)
        off = run_shards("off", zs, args.nray, args.nshard, False, 12345, outdir)
        print("  arm ON  (JvdB14 virial convention) ...", flush=True)
        on = run_shards("on", zs, args.nray, args.nshard, True, 12345, outdir)

        a_off, a_on = np.concatenate(off), np.concatenate(on)
        edges = np.linspace(-0.6, 0.6, 161)
        h = args.nshard // 2
        f_off = jsd(np.concatenate(off[:h]), np.concatenate(off[h:]), edges)
        f_on = jsd(np.concatenate(on[:h]), np.concatenate(on[h:]), edges)
        cross = jsd(a_off, a_on, edges)
        floor = max(f_off, f_on)
        sdr = sd_ratio_with_error(off, on)

        s_off, s_on = summarize(a_off), summarize(a_on)
        r = dict(zs=zs, nray_total=int(a_off.size), off=s_off, on=s_on,
                 jsd_cross=cross, jsd_floor_off=f_off, jsd_floor_on=f_on,
                 jsd_ratio=cross / floor if floor > 0 else float("inf"),
                 sd_ratio_detail=sdr,
                 predicted_excess=PREDICTED_EXCESS.get(zs),
                 d_q01=float(s_on["q01"] - s_off["q01"]),
                 d_ln_inv_mu=float(s_on["ln_inv_mu_trim"] - s_off["ln_inv_mu_trim"]))
        r["verdict"] = "AT FLOOR" if cross <= floor else f"{cross/floor:.1f}x floor"

        print(f"  clipped sd : off {sdr['sd_off']:.6f}  on {sdr['sd_on']:.6f}")
        print(f"    ratio = {sdr['ratio']:.5f} +/- {sdr['ratio_err']:.5f}"
              f"  ({sdr['pct']:+.3f} +/- {sdr['pct_err']:.3f} %) -> {sdr['nsigma']:+.1f} sigma")
        print(f"    within-arm null spread = +/- {sdr['null_pct']:.3f} %"
              f"   | predicted substructure excess {PREDICTED_EXCESS.get(zs)}")
        print(f"  JSD(off,on) = {cross:.3e}  floors {f_off:.3e}/{f_on:.3e} -> {r['verdict']}")
        print(f"  edge q01 shift = {r['d_q01']:+.3e}   flux dln<1/mu> = {r['d_ln_inv_mu']:+.3e}")
        results[str(zs)] = r

    out = ROOT / "data" / "results" / "subhalo_virial" / f"ab_n{args.nray}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
