#!/usr/bin/env python3
"""PDF-level acceptance test for the production subhalo_factor (model 3 vs brute).

Closes the open item from the Wsub work (CLAUDE.md #10, 2026-07-09): model 3 was
validated against brute at the clipped-VARIANCE level over factor 1e-5..1, but the
production factor was never picked at the PDF level. The risk at large factor is
tail-specific: clumps near the resolution boundary are replaced by the Gaussian
Wsub term, which is exact in mean/variance at ANY factor (restriction theorem)
but drops higher cumulants (analytic dropped-c3 share: 0.2%/0.7%/3% at
1e-3/1e-2/1e-1) — invisible to Var, visible in P(lnmu) tails.

Protocol identical to the kappathr/mmin/nz JSD studies (2026-07-10/12):
  - truth = BRUTE subhalo sampling (every clump above m_floor explicit; model 1
    forced on the brute arm since model 3 + brute throws), run as TWO
    independent seed halves -> finite-sample JSD floor from half-vs-half,
    rescaled as c*(1/N1+1/N2);
  - candidates: model 3 at factor 1e-5 (production default), 1e-3, 1e-2 (the
    performance target under test), 1e-1; plus model 1 at 1e-2 (contrast arm —
    shows what the Wsub compensation buys at the PDF level);
  - all configs share NOTHING (disjoint seed blocks, no CRN); default
    fixed-<N>=100 threshold rule everywhere; full model (filaments/bias/ell on),
    subhalo=True, m_floor=1e7;
  - metrics: JSD(ln mu) to combined truth, q99/q99.9/q99.99 of mu with 95%
    order-statistic CIs, sigma(ln mu), <1/mu>, wall-clock.

Sampling is cached per config under data/results/subhalo_factor_jsd{tag}/
(finished shards under shards/ -> interrupted runs resume shard-granular);
re-running skips finished configs and just re-analyzes. Plain subprocess
shards, not mp.Pool (worker segfaults must surface as exit codes).

Usage:
  PY scripts/convergence/subhalo_factor_jsd.py --tag _pilot --scale 0.02
  PY scripts/convergence/subhalo_factor_jsd.py                # full run
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
NPROC = 8          # process-level parallelism sweet spot on the 10-core box
NBINS = 120
NHALOS_DEFAULT = 100
AXIS_IDX = 6       # seed namespace 6e8 — disjoint from mmin(3)/nz(4)/mmin_pd(5)
                   # blocks and the ACE sweep (7e8) and kappathr (<5.2e6)

# config -> (seed_block, subhalo-kwargs for the sampler). seed_block is a
# PERMANENT per-config RNG offset within z: cached samples are keyed by config
# NAME, so never renumber an existing config (stale caches + reused seeds would
# break the disjoint-block assumption). truthA/truthB are the two halves.
CONFIGS = {
    "truthA":   (0, dict(subhalo_brute=True, subhalo_model=1)),
    "truthB":   (1, dict(subhalo_brute=True, subhalo_model=1)),
    "f1em5":    (2, dict(subhalo_model=3, subhalo_factor=1e-5)),  # prod default
    "f1em3":    (3, dict(subhalo_model=3, subhalo_factor=1e-3)),
    "f1em2":    (4, dict(subhalo_model=3, subhalo_factor=1e-2)),  # perf target
    "f1em1":    (5, dict(subhalo_model=3, subhalo_factor=1e-1)),
    "m1_f1em2": (6, dict(subhalo_model=1, subhalo_factor=1e-2)),  # no-Wsub contrast
}

Z_TABLE = {
    0.2: dict(zi=1, n_truth_half=120_000, n_cand=240_000),
    1.0: dict(zi=2, n_truth_half=120_000, n_cand=240_000),
    5.0: dict(zi=3, n_truth_half=120_000, n_cand=240_000),
    10.0: dict(zi=4, n_truth_half=60_000, n_cand=120_000),
}


def seed_base(z: float, block: int) -> int:
    return 100_000_000 * AXIS_IDX + 1_000_000 * Z_TABLE[z]["zi"] + 10_000 * block


def n_for(z: float, name: str, scale: float) -> int:
    zt = Z_TABLE[z]
    n = zt["n_truth_half"] if name.startswith("truth") else zt["n_cand"]
    return max(NPROC * 250, int(round(n * scale / NPROC)) * NPROC)


def worker_main(spec_json: str) -> None:
    """Subprocess entry: run ONE shard, save its .npy, exit."""
    spec = json.loads(spec_json)
    import gwlensing as gw
    d = gw.sample_lnmu_ml_with_diagnostics(
        z=spec["z"], h=H, OmegaM=OM, sigma8=S8, nsamples=spec["n"],
        seed=spec["seed"], strict_weak_lensing=False, subhalo=True,
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
            key, shard, z, kwargs, n, seed = pending.pop(0)
            outfile = shard_dir / f"{key}_{shard}.npy"
            spec = json.dumps(dict(key=key, shard=shard, z=z, kwargs=kwargs,
                                   n=n, seed=seed, outfile=str(outfile)))
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
def run_sampling(outdir: Path, zs, scale: float) -> dict:
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

    for z in zs:
        kthr_eff = gw.get_kappa_threshold(z, H, OM, S8, NHALOS_DEFAULT)
        for name, (block, kwargs) in CONFIGS.items():
            key = f"z{z:g}_{name}"
            f = outdir / f"{key}.npy"
            n = n_for(z, name, scale)
            fac = kwargs.get("subhalo_factor")
            meta[key] = dict(
                z=z, config=name, kwargs=kwargs, kthr_eff=float(kthr_eff),
                kthr_clump=(float(fac * kthr_eff) if fac else None), n=n)
            if f.exists():
                continue
            per = n // NPROC
            base = seed_base(z, block)
            for s in range(NPROC):
                sf = shard_dir / f"{key}_{s}.npy"
                if sf.exists():
                    parts.setdefault(key, {})[s] = np.load(sf)
                    continue
                # brute shards dominate; smaller factor -> more explicit clumps
                cost = per * (1.0 + z) * (
                    100.0 if kwargs.get("subhalo_brute")
                    else 25.0 / (1.0 + 1e3 * kwargs.get("subhalo_factor", 1e-5)))
                tasks.append((cost, (key, s, z, kwargs, per, base + s)))
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
            print(f"[done] {key:24s} n={arr.size:>7,}  cpu={times[key]:8.1f}s "
                  f"({times[key]/arr.size*1e4:7.2f} s/1e4)  "
                  f"wall so far {time.perf_counter()-t0:7.1f}s", flush=True)
    print(f"[sample] total wall {time.perf_counter()-t0:.1f}s", flush=True)
    return meta


# ---------------------------------------------------------------- analysis
# density()/jsd() MUST stay bit-identical to the kappathr/mmin/nz studies
# (kappathr_convergence_decision.py, kappathr_subhalo_jsd.py,
# convergence_scan.py) — cross-study JSD comparisons depend on an identical
# estimator. Change all or none.
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


def analyze(outdir: Path, meta: dict, zs) -> dict:
    out = {}
    for z in zs:
        keys = {c: f"z{z:g}_{c}" for c in CONFIGS}
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
        for c in [c for c in CONFIGS if c != "truthB"]:
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
                kthr_clump=meta[keys[c]]["kthr_clump"],
                jsd=j, floor_pred=fl, jsd_excess=max(j - fl, 0.0),
                sigma_lnmu=float(np.std(x)),
                inv_mu=float(np.mean(np.exp(-x))),
                tails=qs)
        out[f"z{z:g}"] = grp
    return out


def report(res: dict) -> str:
    lines = ["# subhalo_factor PDF-level JSD acceptance (model 3 vs brute truth)",
             "", "Truth = brute subhalo sampling (model 1 arm, every clump above "
             "m_floor=1e7 explicit), two independent seed halves, combined. "
             "Default fixed-<N>=100 threshold rule, full model, subhalo on. "
             "JSD in nats, 120 shared bins; floor_pred = half-split floor rescaled "
             "by (1/N1+1/N2); excess = max(JSD - floor, 0).", ""]
    for gk, g in res.items():
        lines += [f"## {gk}   (floor_half={g['floor_half']:.2e}, n_truth={g['n_truth']:,})", "",
                  "| config | kwargs | kthr_clump | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |",
                  "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for c, d in g["configs"].items():
            t = d["tails"]
            kw = ",".join(f"{k}={v:g}" for k, v in d["kwargs"].items()) or "-"
            kc = f"{d['kthr_clump']:.2e}" if d["kthr_clump"] else "-"
            lines.append(
                f"| {c} | {kw} | {kc} | {d['jsd']:.2e} | "
                f"{d['floor_pred']:.2e} | {d['jsd_excess']:.2e} | {d['sigma_lnmu']:.4f} | "
                f"{d['inv_mu']:.4f} | {t['q99'][1]:.3f} | {t['q999'][1]:.3f} | {t['q9999'][1]:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    if "--worker-task" in sys.argv:          # subprocess shard entry point
        worker_main(sys.argv[sys.argv.index("--worker-task") + 1])
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0, 5.0])
    args = ap.parse_args()
    for z in args.zs:
        if z not in Z_TABLE:
            ap.error(f"--zs {z:g} unsupported; supported: {sorted(Z_TABLE)}")

    outdir = REPO / "data" / "results" / f"subhalo_factor_jsd{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    meta = run_sampling(outdir, args.zs, args.scale)
    res = analyze(outdir, meta, args.zs)
    (outdir / "summary.json").write_text(json.dumps(res, indent=1))
    txt = report(res)
    (outdir / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
