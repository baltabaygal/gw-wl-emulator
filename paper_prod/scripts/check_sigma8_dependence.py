#!/usr/bin/env python
"""Does sigma_8 really move only the LOW-mu side of dP_I/dmu?

Written 2026-07-29 to test a sentence in Sec. II.B before it goes in the paper:

    "at fixed source redshift its shape depends mainly on sigma_8, whose imprint
     sits almost entirely on the low-magnification side while the region mu > 1
     is left nearly unchanged [Premadi:2001]"

Premadi et al. 2001 sec. 5.1.1 does say that (paper_writer.md sec 6 verified it),
but they say it about THEIR ray-tracing model, and they vary lambda_0 as well.
The claim as written is about OUR distributions, so it has to be measured here.

Varies sigma_8 ALONE at the fiducial Omega_M and h, at the full PRODUCTION_CONFIG,
and reports where the distribution actually moves:

  - the 0.1% quantile of ln mu          -> the low-mu edge
  - the mode and the median             -> the body
  - clipped sd(ln mu) on mu <= 3        -> the width (the certified estimator;
                                           raw sd is monster-ray junk, see
                                           paper_writer.md sec 4)
  - P(mu > 1.2 / 1.5 / 2)               -> the high-mu side
  - the ratio dP_I/dmu (s8) / (s8_fid)  -> per-bin, so "nearly unchanged above
                                           mu = 1" is a measurement, not a hope

Run (needs the `test` conda env for the C++ module):
    PY=/Users/baltabay/miniforge3/envs/test/bin/python
    $PY paper_prod/scripts/check_sigma8_dependence.py
    $PY paper_prod/scripts/check_sigma8_dependence.py --nreal 400000 --zs 1 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "build"))

from ml.params import PRODUCTION_CONFIG, PRODUCTION_CONFIG_HASH  # noqa: E402

# Same non-physics knobs the figure script uses, so this is the same model.
COMMON = dict(filaments=True, ell=True, Nhalos=100, Mmin=1e7, NM=100, Nz=100,
              bias=True)
FID = dict(OmegaM=0.315, h=0.674)
S8_FID = 0.811

EDGES = np.geomspace(0.05, 200.0, 4000)
CEN = np.sqrt(EDGES[:-1] * EDGES[1:])


def run(zs, s8, nreal, seed):
    import gwlensing as gw
    lnmu = np.asarray(gw.sample_lnmu(z=float(zs), sigma8=float(s8),
                                     Nreal=int(nreal), seed=int(seed),
                                     **FID, **COMMON, **PRODUCTION_CONFIG),
                      dtype=float)
    return lnmu[np.isfinite(lnmu)]


def summarize(lnmu):
    mu = np.exp(lnmu)
    clip = np.abs(mu) <= 3.0          # certified support, not the raw sample
    counts, _ = np.histogram(mu, bins=EDGES)
    pdf = counts / (mu.size * np.diff(EDGES))
    return dict(
        n=int(mu.size),
        q001=float(np.quantile(lnmu, 0.001)),
        median_mu=float(np.median(mu)),
        mode_mu=float(CEN[int(np.argmax(counts))]),
        sd_lnmu_clip3=float(np.std(lnmu[clip])),
        p_gt_1p2=float(np.mean(mu > 1.2)),
        p_gt_1p5=float(np.mean(mu > 1.5)),
        p_gt_2=float(np.mean(mu > 2.0)),
        pdf=pdf,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0, 5.0])
    ap.add_argument("--s8", type=float, nargs="+", default=[0.65, 0.811, 1.05])
    ap.add_argument("--nreal", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str,
                    default=str(REPO / "data" / "results" / "sigma8_shape"
                               / "sigma8_shape.json"))
    args = ap.parse_args()

    print(f"PRODUCTION_CONFIG hash {PRODUCTION_CONFIG_HASH}", flush=True)
    out = {"config_hash": PRODUCTION_CONFIG_HASH, "nreal": args.nreal,
           "seed": args.seed, "fid": FID, "s8_fid": S8_FID, "runs": {}}

    for zs in args.zs:
        ref = None
        for s8 in args.s8:
            lnmu = run(zs, s8, args.nreal, args.seed)
            s = summarize(lnmu)
            pdf = s.pop("pdf")
            if abs(s8 - S8_FID) < 1e-9:
                ref = pdf
            out["runs"][f"z{zs:g}_s8{s8:g}"] = s
            print(f"[z_s={zs:g} s8={s8:g}] n={s['n']:,} "
                  f"q0.1%(lnmu)={s['q001']:+.4f} mode={s['mode_mu']:.4f} "
                  f"median={s['median_mu']:.4f} sd_clip={s['sd_lnmu_clip3']:.5f} "
                  f"P(>1.2)={s['p_gt_1p2']:.4e} P(>1.5)={s['p_gt_1p5']:.4e} "
                  f"P(>2)={s['p_gt_2']:.4e}", flush=True)
            out["runs"][f"z{zs:g}_s8{s8:g}"]["_pdf"] = pdf.tolist()

        # per-bin ratio against the fiducial sigma_8, in three mu windows
        if ref is not None:
            for s8 in args.s8:
                if abs(s8 - S8_FID) < 1e-9:
                    continue
                pdf = np.asarray(out["runs"][f"z{zs:g}_s8{s8:g}"]["_pdf"])
                for lo, hi, name in ((0.7, 1.0, "low  0.7-1.0"),
                                     (1.0, 1.5, "body 1.0-1.5"),
                                     (1.5, 3.0, "high 1.5-3.0")):
                    m = (CEN >= lo) & (CEN < hi) & (ref > 0) & (pdf > 0)
                    # count-weighted, so sparse bins do not dominate
                    w = ref[m]
                    r = np.average(pdf[m] / ref[m], weights=w)
                    print(f"    z_s={zs:g} s8 {S8_FID}->{s8:g}  {name}: "
                          f"mean dP ratio = {r:.4f}", flush=True)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    for k in out["runs"]:
        out["runs"][k].pop("_pdf", None)
    p.write_text(json.dumps(out, indent=2))
    print(f"[out] {p}", flush=True)


if __name__ == "__main__":
    main()
