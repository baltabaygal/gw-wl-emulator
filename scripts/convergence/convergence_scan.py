#!/usr/bin/env python3
"""Grid-convergence JSD acceptance test for Mmin and Nz (axis-generic).

Generalizes scripts/subhalo_gate/kappathr_subhalo_jsd.py to arbitrary
numerical-knob axes. Protocol identical to the kappathr study (2026-07-10):
  - truth = the most-refined grid value, run as TWO independent seed halves ->
    finite-sample JSD floor from half-vs-half, rescaled as c*(1/N1+1/N2);
  - candidates share NOTHING with truth (disjoint seed blocks, no CRN);
  - metrics: JSD(ln mu) to combined truth, q99/q99.9/q99.99 of mu with 95%
    order-statistic CIs, sigma(ln mu), <1/mu>, wall-clock;
  - default threshold rule (fixed-<N>=100, kappathr_flat=-1) everywhere.

Axes:
  mmin  truth Mmin=1e4; candidates 1e5..1e9 (1e7 = production default) plus an
        NM=200 control at Mmin=1e4 that isolates M-grid coarsening (NM is fixed
        at 100 while the integration range widens) from missing-halo physics.
  nz    truth Nz=TRUTH_NZ (set from the quadrature arm,
        scripts/convergence/analytic_arms.py); candidates 25..200 (+400 when
        truth is 800). Under the fixed-<N> rule kappa_thr moves with Nz — that
        coupling is part of the production error and is deliberately included.

Arms:
  main  subhalo off, z_s in {0.2, 1, 5, 10}.
  sub   subhalo on (model 3, factor 1e-5), z_s=1 spot-check only. For mmin the
        sub arm's truth is {Mmin=1e6, m_floor=1e6}; candidates separate the two
        knobs: production default (Mmin=1e7) and Mmin=1e6 with m_floor left at
        its 1e7 default.

Sampling is cached per config under data/results/{axis}_convergence{tag}/
(finished shards under shards/ -> interrupted runs resume shard-granular);
re-running skips finished configs and just re-analyzes.

Usage:
  PY scripts/convergence/convergence_scan.py --axis mmin --tag _pilot --scale 0.05
  PY scripts/convergence/convergence_scan.py --axis mmin          # full run
  PY scripts/convergence/convergence_scan.py --axis nz --arms main sub
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build"))

H, OM, S8 = 0.674, 0.315, 0.811
SUBHALO_MODEL = 3
SUBHALO_FACTOR = 1.0e-5
NPROC = 8          # process-level parallelism sweet spot on the 10-core box
NBINS = 120
NHALOS_DEFAULT = 100

# Truth refinement for the nz axis; set from the quadrature arm before the
# full run (400 unless analytic_arms.py shows drift past 400, then 800).
TRUTH_NZ = 400

# config -> (seed_block, sampler-kwargs). seed_block is a PERMANENT per-config
# RNG offset within (axis, arm, z): cached samples are keyed by config NAME,
# so never renumber an existing config (stale caches + reused seeds would
# break the disjoint-block assumption). truthA/truthB are the two halves.
CONFIGS = {
    ("mmin", "main"): {
        "truthA": (0, {"Mmin": 1e4}),
        "truthB": (1, {"Mmin": 1e4}),
        "m1e5":   (2, {"Mmin": 1e5}),
        "m1e6":   (3, {"Mmin": 1e6}),
        "m1e7":   (4, {"Mmin": 1e7}),      # production default
        "m1e8":   (5, {"Mmin": 1e8}),
        "m1e9":   (6, {"Mmin": 1e9}),
        "nm200ctl": (7, {"Mmin": 1e4, "NM": 200}),
    },
    ("mmin", "sub"): {
        "truthA": (0, {"Mmin": 1e6, "m_floor": 1e6}),
        "truthB": (1, {"Mmin": 1e6, "m_floor": 1e6}),
        "m1e7":   (2, {"Mmin": 1e7}),      # production default
        # separates the two floors: Mmin lowered, m_floor at default.
        # Segfaulted pre-2026-07-12 (grid-edge OOB in interpolateNFWMass,
        # fixed by the jm clamp); needs the rebuilt module.
        "m1e6_mfloor1e7": (3, {"Mmin": 1e6}),
    },
    # per-decade-matched family: NM = 10 pts/decade of [Mmin, 1e17] for every
    # config (the plain mmin axis holds NM=100 fixed, so its truth at Mmin=1e4
    # has 7.7 pts/decade — the 2026-07-11 full run showed that M-grid
    # coarsening alone contributes few-e-4 JSD at z>=5, same order as the Mmin
    # physics; this family isolates the physics at fixed resolution).
    ("mmin_pd", "main"): {
        "truthA": (0, {"Mmin": 1e4, "NM": 130}),
        "truthB": (1, {"Mmin": 1e4, "NM": 130}),
        "m1e5":   (2, {"Mmin": 1e5, "NM": 120}),
        "m1e6":   (3, {"Mmin": 1e6, "NM": 110}),
        "m1e7":   (4, {"Mmin": 1e7, "NM": 100}),   # = production default
        "m1e8":   (5, {"Mmin": 1e8, "NM": 90}),
        "m1e9":   (6, {"Mmin": 1e9, "NM": 80}),
    },
    ("nz", "main"): {
        "truthA": (0, {"Nz": TRUTH_NZ}),
        "truthB": (1, {"Nz": TRUTH_NZ}),
        "nz25":   (2, {"Nz": 25}),
        "nz50":   (3, {"Nz": 50}),
        "nz100":  (4, {"Nz": 100}),        # production default
        "nz200":  (5, {"Nz": 200}),
    },
    ("nz", "sub"): {
        "truthA": (0, {"Nz": TRUTH_NZ}),
        "truthB": (1, {"Nz": TRUTH_NZ}),
        "nz100":  (2, {"Nz": 100}),
    },
}
if TRUTH_NZ >= 800:
    CONFIGS[("nz", "main")]["nz400"] = (6, {"Nz": 400})

AXIS_IDX = {"mmin": 3, "nz": 4, "mmin_pd": 5}   # keeps seeds disjoint from the
ARM_IDX = {"main": 0, "sub": 1}    # kappathr study (which used blocks < 5.2e6)

Z_TABLE = {
    0.2: dict(zi=1, n_truth_half=120_000, n_cand=240_000),
    1.0: dict(zi=2, n_truth_half=120_000, n_cand=240_000),
    5.0: dict(zi=3, n_truth_half=120_000, n_cand=240_000),
    10.0: dict(zi=4, n_truth_half=120_000, n_cand=240_000),
}
SUB_ZS = [1.0]     # the sub arm is a z_s=1 spot-check only


def seed_base(axis: str, z: float, arm: str, block: int) -> int:
    return (100_000_000 * AXIS_IDX[axis] + 1_000_000 * Z_TABLE[z]["zi"]
            + 100_000 * ARM_IDX[arm] + 10_000 * block)


def n_for(z: float, name: str, scale: float) -> int:
    zt = Z_TABLE[z]
    n = zt["n_truth_half"] if name.startswith("truth") else zt["n_cand"]
    return max(NPROC * 250, int(round(n * scale / NPROC)) * NPROC)


def worker_main(spec_json: str) -> None:
    """Subprocess entry: run ONE shard, save its .npy, exit.

    Plain subprocesses replaced mp.Pool on 2026-07-12: after an OS memory
    kill, mp spawn handshakes wedged reproducibly (workers parked in the
    bootstrap os.read, 0% CPU) — independent processes have no handshake to
    wedge and match the proven process-parallelism pattern (tmp/bench_mp.py).
    """
    spec = json.loads(spec_json)
    import gwlensing as gw
    d = gw.sample_lnmu_ml_with_diagnostics(
        z=spec["z"], h=H, OmegaM=OM, sigma8=S8, nsamples=spec["n"],
        seed=spec["seed"], strict_weak_lensing=False, subhalo=spec["sub"],
        subhalo_model=SUBHALO_MODEL, subhalo_factor=SUBHALO_FACTOR,
        **spec["kwargs"])
    tmp = Path(spec["outfile"] + ".tmp.npy")
    np.save(tmp, np.asarray(d["lnmu"]))
    tmp.rename(spec["outfile"])          # atomic: no half-written shards


def run_pool_subproc(tasks, shard_dir: Path):
    """Yield (key, shard, lnmu, wall_s) from up to NPROC shard subprocesses."""
    pending = list(tasks)
    running = {}
    while pending or running:
        while pending and len(running) < NPROC:
            key, shard, z, kwargs, sub_on, n, seed = pending.pop(0)
            outfile = shard_dir / f"{key}_{shard}.npy"
            spec = json.dumps(dict(key=key, shard=shard, z=z, kwargs=kwargs,
                                   sub=sub_on, n=n, seed=seed,
                                   outfile=str(outfile)))
            log = open(shard_dir / f"{key}_{shard}.log", "w")
            p = subprocess.Popen([sys.executable, "-u", __file__,
                                  "--worker-task", spec],
                                 stdout=log, stderr=subprocess.STDOUT)
            running[p] = (key, shard, outfile, log, time.perf_counter())
        finished = [p for p in running if p.poll() is not None]
        if not finished:
            time.sleep(0.5)
            continue
        for p in finished:
            key, shard, outfile, log, t0 = running.pop(p)
            log.close()
            if p.returncode != 0 or not outfile.exists():
                raise RuntimeError(
                    f"shard {key}_{shard} failed (rc={p.returncode}); see "
                    f"{shard_dir / f'{key}_{shard}.log'}")
            (shard_dir / f"{key}_{shard}.log").unlink(missing_ok=True)
            yield key, shard, np.load(outfile), time.perf_counter() - t0


# ---------------------------------------------------------------- sampling
def run_sampling(axis: str, outdir: Path, zs, arms, scale: float) -> dict:
    import gwlensing as gw
    shard_dir = outdir / "shards"
    shard_dir.mkdir(exist_ok=True)
    tasks, meta, parts = [], {}, {}

    def finish(key):
        shards = parts.pop(key)
        arr = np.concatenate([shards[i] for i in range(NPROC)])
        np.save(outdir / f"{key}.npy", arr)
        for i in range(NPROC):
            (shard_dir / f"{key}_{i}.npy").unlink(missing_ok=True)
        return arr

    for arm in arms:
        for z in (zs if arm == "main" else [z for z in SUB_ZS if z in zs]):
            for name, (block, kwargs) in CONFIGS[(axis, arm)].items():
                key = f"z{z:g}_{arm}_{name}"
                f = outdir / f"{key}.npy"
                n = n_for(z, name, scale)
                grid_kw = {k: v for k, v in kwargs.items()
                           if k in ("Mmin", "NM", "Nz")}
                # default rule: threshold the sampler actually solves for
                if "NM" in grid_kw or "Nz" in grid_kw or "Mmin" in grid_kw:
                    kthr_eff = gw.get_kappa_threshold(
                        z, H, OM, S8, NHALOS_DEFAULT, **grid_kw)
                else:
                    kthr_eff = gw.get_kappa_threshold(z, H, OM, S8, NHALOS_DEFAULT)
                meta[key] = dict(z=z, arm=arm, config=name, kwargs=kwargs,
                                 kthr_eff=float(kthr_eff), n=n)
                if f.exists():
                    continue
                per = n // NPROC
                base = seed_base(axis, z, arm, block)
                for s in range(NPROC):
                    sf = shard_dir / f"{key}_{s}.npy"
                    if sf.exists():
                        parts.setdefault(key, {})[s] = np.load(sf)
                        continue
                    cost = per * (1.0 + 25.0 * (arm == "sub")) * (1.0 + z)
                    tasks.append((cost, (key, s, z, kwargs, arm == "sub",
                                         per, base + s)))
                if len(parts.get(key, ())) == NPROC:
                    finish(key)

    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    if not tasks:
        print("[sample] all configs cached", flush=True)
        return meta

    tasks.sort(key=lambda t: -t[0])
    print(f"[sample] {len(tasks)} shards over {NPROC} procs", flush=True)
    times = {}
    t0 = time.perf_counter()
    for key, shard, lnmu, dt in run_pool_subproc([t[1] for t in tasks], shard_dir):
        parts.setdefault(key, {})[shard] = lnmu
        times[key] = times.get(key, 0.0) + dt
        if len(parts[key]) == NPROC:
            arr = finish(key)
            print(f"[done] {key:26s} n={arr.size:>7,}  cpu={times[key]:8.1f}s "
                  f"({times[key]/arr.size*1e4:7.2f} s/1e4)  "
                  f"wall so far {time.perf_counter()-t0:7.1f}s", flush=True)
    print(f"[sample] total wall {time.perf_counter()-t0:.1f}s", flush=True)
    return meta


# ---------------------------------------------------------------- analysis
# density()/jsd() MUST stay bit-identical to the kappathr studies
# (kappathr_convergence_decision.py, kappathr_subhalo_jsd.py) — cross-study
# JSD comparisons depend on an identical estimator. Change all or none.
def density(x, edges):
    c, _ = np.histogram(x, bins=edges)
    w = np.diff(edges)
    t = c.sum()
    return c / (t * w) if t else np.zeros_like(w, float)


def jsd(p, q, w):
    eps = 1e-300
    p = np.clip(p, eps, None); p = p / (p * w).sum()
    q = np.clip(q, eps, None); q = q / (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * float((p * np.log(p / m) * w).sum() + (q * np.log(q / m) * w).sum())


def quantile_ci(x_sorted, q, alpha=0.05):
    from scipy.stats import binom
    n = x_sorted.size
    lo = int(binom.ppf(alpha / 2, n, q))
    hi = int(min(binom.ppf(1 - alpha / 2, n, q), n - 1))
    return (float(x_sorted[min(max(lo, 0), n - 1)]),
            float(np.quantile(x_sorted, q)),
            float(x_sorted[hi]))


def analyze(axis: str, outdir: Path, meta: dict, zs, arms) -> dict:
    out = {}
    for arm in arms:
        for z in (zs if arm == "main" else [z for z in SUB_ZS if z in zs]):
            names = list(CONFIGS[(axis, arm)])
            keys = {c: f"z{z:g}_{arm}_{c}" for c in names}
            if not all((outdir / f"{k}.npy").exists() for k in keys.values()):
                continue
            smp = {c: np.load(outdir / f"{k}.npy") for c, k in keys.items()}
            smp = {c: v[np.isfinite(v)] for c, v in smp.items()}
            truth = np.concatenate([smp["truthA"], smp["truthB"]])

            allx = np.concatenate(list(smp.values()))
            lo, hi = np.quantile(allx, 1e-4), np.quantile(allx, 1 - 1e-4)
            pad = 0.08 * (hi - lo)
            edges = np.linspace(lo - pad, hi + pad, NBINS + 1)
            w = np.diff(edges)

            na, nb = smp["truthA"].size, smp["truthB"].size
            floor_half = jsd(density(smp["truthA"], edges),
                             density(smp["truthB"], edges), w)
            c_floor = floor_half / (1.0 / na + 1.0 / nb)

            p_truth = density(truth, edges)
            grp = dict(floor_half=floor_half, c_floor=c_floor,
                       n_truth=int(truth.size), edges=edges.tolist(), configs={})
            for c in [c for c in names if c != "truthB"] :
                x, lab = (truth, "truth") if c == "truthA" else (smp[c], c)
                j = jsd(p_truth, density(x, edges), w) if lab != "truth" else 0.0
                fl = (c_floor * (1.0 / truth.size + 1.0 / x.size)
                      if lab != "truth" else 0.0)
                mu = np.exp(np.sort(x))
                qs = {f"q{q}".replace("0.", ""): quantile_ci(mu, q)
                      for q in (0.99, 0.999, 0.9999)}
                grp["configs"][lab] = dict(
                    n=int(x.size), kwargs=meta[keys[c]]["kwargs"],
                    kthr_eff=meta[keys[c]]["kthr_eff"],
                    jsd=j, floor_pred=fl, jsd_excess=max(j - fl, 0.0),
                    sigma_lnmu=float(np.std(x)),
                    inv_mu=float(np.mean(np.exp(-x))),
                    tails=qs)
            out[f"z{z:g}_{arm}"] = grp
    return out


def report(axis: str, res: dict) -> str:
    truth_desc = {"mmin": "Mmin=1e4 (sub arm: Mmin=1e6, m_floor=1e6)",
                  "mmin_pd": "Mmin=1e4, NM=130 (10 pts/decade for all configs)",
                  "nz": f"Nz={TRUTH_NZ}"}[axis]
    lines = [f"# {axis} grid-convergence JSD test (default fixed-<N>=100 rule)",
             "", f"Truth = {truth_desc}, two independent seed halves, combined. "
             "JSD in nats, 120 shared bins; floor_pred = half-split floor rescaled "
             "by (1/N1+1/N2); excess = max(JSD - floor, 0).", ""]
    for gk, g in res.items():
        lines += [f"## {gk}   (floor_half={g['floor_half']:.2e}, n_truth={g['n_truth']:,})", "",
                  "| config | kwargs | kthr_eff | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |",
                  "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for c, d in g["configs"].items():
            t = d["tails"]
            kw = ",".join(f"{k}={v:g}" for k, v in d["kwargs"].items()) or "-"
            lines.append(
                f"| {c} | {kw} | {d['kthr_eff']:.2e} | {d['jsd']:.2e} | "
                f"{d['floor_pred']:.2e} | {d['jsd_excess']:.2e} | {d['sigma_lnmu']:.4f} | "
                f"{d['inv_mu']:.4f} | {t['q99'][1]:.3f} | {t['q999'][1]:.3f} | {t['q9999'][1]:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    if "--worker-task" in sys.argv:          # subprocess shard entry point
        worker_main(sys.argv[sys.argv.index("--worker-task") + 1])
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", required=True, choices=["mmin", "nz", "mmin_pd"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--zs", type=float, nargs="+", default=[0.2, 1.0, 5.0, 10.0])
    ap.add_argument("--arms", nargs="+", choices=["main", "sub"], default=["main"])
    args = ap.parse_args()
    for z in args.zs:
        if z not in Z_TABLE:
            ap.error(f"--zs {z:g} unsupported; supported: {sorted(Z_TABLE)}")

    outdir = REPO / "data" / "results" / f"{args.axis}_convergence{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    meta = run_sampling(args.axis, outdir, args.zs, args.arms, args.scale)
    res = analyze(args.axis, outdir, meta, args.zs, args.arms)
    (outdir / "summary.json").write_text(json.dumps(res, indent=1))
    txt = report(args.axis, res)
    (outdir / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
