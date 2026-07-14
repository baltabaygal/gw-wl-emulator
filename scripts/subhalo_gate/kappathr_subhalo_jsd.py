#!/usr/bin/env python3
"""Threshold-rule JSD acceptance test WITH subhalos on (model 3).

Extends the halo-only study (scripts/figures/kappathr_convergence_decision.py):
the fixed-<N>=100 default was chosen on subhalo-off evidence; subhalos are only
generated inside EXPLICIT hosts, so a higher threshold discards substructure
from every host it demotes to the Gaussian field. This script measures whether
that changes the JSD picture.

Protocol (agreed 2026-07-10):
  - subhalo=True, subhalo_model=3, subhalo_factor=1e-5 — identical in every run;
  - truth = flat kappa_thr = 3e-5, run as TWO independent seed halves ->
    finite-sample JSD floor from half-vs-half, rescaled as c*(1/N1+1/N2);
  - candidates: flat 1e-4, flat 1e-3, legacy fixed-<N>=100 (kappathr_flat=-1);
  - z_s in {1, 10};
  - a subhalo-OFF arm with the identical protocol (the halo-only study's z=1
    JSDs sat AT the 80k-sample floor, so its numbers can't be reused directly);
  - metrics: JSD(ln mu) to combined truth, q99/q99.9/q99.99 of mu with 95%
    order-statistic CIs, sigma(ln mu), <1/mu>.

All configs use disjoint seed blocks (no common random numbers), so the
half-vs-half floor applies directly to every comparison.

Sampling is cached per config under data/results/kappathr_subhalo_jsd<tag>/
(finished shards under shards/, so an interrupted run resumes shard-granular);
re-running skips finished configs and just re-analyzes.

Usage:
  PY scripts/subhalo_gate/kappathr_subhalo_jsd.py --tag _pilot --scale 0.05
  PY scripts/subhalo_gate/kappathr_subhalo_jsd.py            # full run
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build"))

H, OM, S8 = 0.674, 0.315, 0.811
SUBHALO_MODEL = 3
SUBHALO_FACTOR = 1.0e-5
TRUTH_KTHR = 3.0e-5
NPROC = 8          # process-level parallelism sweet spot on the 10-core box
NBINS = 120
NHALOS_DEFAULT = 100   # mirrors py::arg("Nhalos") in cpp/python_bindings.cpp

# rule -> (seed_block, kappathr_flat); -1 = legacy fixed-<N>=100.
# seed_block is a PERMANENT per-rule RNG offset: cached samples are keyed by
# rule NAME, so never renumber an existing rule — reordering blocks would mix
# seed streams with stale caches and break the disjoint-seed-block assumption.
RULES = {
    "truthA": (0, TRUTH_KTHR),
    "truthB": (1, TRUTH_KTHR),
    "flat_1em4": (2, 1.0e-4),
    "flat_1em3": (3, 1.0e-3),
    "fixedN": (4, -1.0),
}

# per-z seed offsets and sample sizes (truth halves n_truth_half each);
# extend this table to support more z_s values (zi must stay unique).
Z_TABLE = {
    1.0: dict(zi=1, n_truth_half=120_000, n_cand=240_000),
    10.0: dict(zi=2, n_truth_half=60_000, n_cand=120_000),
}
ARMS = {"on": True, "off": False}


def seed_base(z: float, arm: str, rule: str) -> int:
    zi = Z_TABLE[z]["zi"]
    ai = {"on": 0, "off": 1}[arm]
    return 1_000_000 * zi + 100_000 * ai + 10_000 * RULES[rule][0]


def n_for(z: float, rule: str, scale: float) -> int:
    zt = Z_TABLE[z]
    n = zt["n_truth_half"] if rule.startswith("truth") else zt["n_cand"]
    return max(NPROC * 250, int(round(n * scale / NPROC)) * NPROC)


def worker(task):
    """One shard: returns (key, shard_idx, lnmu, wall_seconds)."""
    key, shard, z, kthr, sub_on, n, seed = task
    import gwlensing as gw  # per-process import
    t0 = time.perf_counter()
    d = gw.sample_lnmu_ml_with_diagnostics(
        z=z, h=H, OmegaM=OM, sigma8=S8, nsamples=n, seed=seed,
        strict_weak_lensing=False, subhalo=sub_on,
        subhalo_model=SUBHALO_MODEL, subhalo_factor=SUBHALO_FACTOR,
        kappathr_flat=kthr)
    dt = time.perf_counter() - t0
    return key, shard, np.asarray(d["lnmu"]), dt


# ---------------------------------------------------------------- sampling
def run_sampling(outdir: Path, zs, arms, scale: float) -> dict:
    import gwlensing as gw
    shard_dir = outdir / "shards"   # per-shard cache -> crash/resume granularity
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
        for arm in arms:
            for rule, (_, kthr) in RULES.items():
                key = f"z{z:g}_{arm}_{rule}"
                f = outdir / f"{key}.npy"
                n = n_for(z, rule, scale)
                kthr_eff = (kthr if kthr > 0
                            else gw.get_kappa_threshold(z, H, OM, S8, NHALOS_DEFAULT))
                nexp = gw.get_expected_halo_count(z, H, OM, S8, kthr_eff)
                meta[key] = dict(z=z, arm=arm, rule=rule, kthr=kthr,
                                 kthr_eff=float(kthr_eff), Nexp=float(nexp), n=n)
                if f.exists():
                    continue
                per = n // NPROC
                base = seed_base(z, arm, rule)
                for s in range(NPROC):
                    sf = shard_dir / f"{key}_{s}.npy"
                    if sf.exists():   # shard survived an interrupted run
                        parts.setdefault(key, {})[s] = np.load(sf)
                        continue
                    # crude cost estimate for longest-first scheduling
                    cost = per * (nexp if arm == "on" else 0.05 * nexp + 5)
                    tasks.append((cost, (key, s, z, kthr, ARMS[arm], per, base + s)))
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
    with mp.Pool(NPROC) as pool:
        for key, shard, lnmu, dt in pool.imap_unordered(worker, [t[1] for t in tasks]):
            np.save(shard_dir / f"{key}_{shard}.npy", lnmu)
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
# density()/jsd() MUST stay bit-identical to the halo-only study
# (scripts/figures/kappathr_convergence_decision.py) — the cross-study JSD
# comparison depends on an identical estimator. Change both or neither.
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


def analyze(outdir: Path, meta: dict, zs, arms) -> dict:
    out = {}
    for z in zs:
        for arm in arms:
            keys = {r: f"z{z:g}_{arm}_{r}" for r in RULES}
            if not all((outdir / f"{k}.npy").exists() for k in keys.values()):
                continue
            smp = {r: np.load(outdir / f"{k}.npy") for r, k in keys.items()}
            smp = {r: v[np.isfinite(v)] for r, v in smp.items()}
            truth = np.concatenate([smp["truthA"], smp["truthB"]])

            allx = np.concatenate(list(smp.values()))
            lo, hi = np.quantile(allx, 1e-4), np.quantile(allx, 1 - 1e-4)
            pad = 0.08 * (hi - lo)
            edges = np.linspace(lo - pad, hi + pad, NBINS + 1)
            w = np.diff(edges)

            # finite-sample floor: JSD(truthA, truthB) = c*(1/Na+1/Nb)
            na, nb = smp["truthA"].size, smp["truthB"].size
            floor_half = jsd(density(smp["truthA"], edges),
                             density(smp["truthB"], edges), w)
            c_floor = floor_half / (1.0 / na + 1.0 / nb)

            p_truth = density(truth, edges)
            grp = dict(floor_half=floor_half, c_floor=c_floor,
                       n_truth=int(truth.size), edges=edges.tolist(), rules={})
            for r in ("flat_1em4", "flat_1em3", "fixedN", "truthA"):
                x = smp[r] if r != "truthA" else None
                lab = r
                if r == "truthA":   # truth self-metrics from the combined sample
                    x, lab = truth, "truth"
                j = jsd(p_truth, density(x, edges), w) if lab != "truth" else 0.0
                fl = c_floor * (1.0 / truth.size + 1.0 / x.size) if lab != "truth" else 0.0
                mu = np.exp(np.sort(x))
                qs = {f"q{q}".replace("0.", ""): quantile_ci(mu, q)
                      for q in (0.99, 0.999, 0.9999)}
                grp["rules"][lab] = dict(
                    n=int(x.size),
                    kthr_eff=meta[keys[r]]["kthr_eff"], Nexp=meta[keys[r]]["Nexp"],
                    jsd=j, floor_pred=fl, jsd_excess=max(j - fl, 0.0),
                    sigma_lnmu=float(np.std(x)),
                    inv_mu=float(np.mean(np.exp(-x))),
                    tails=qs)
            out[f"z{z:g}_{arm}"] = grp
    return out


def report(res: dict) -> str:
    lines = ["# kappathr rule JSD test — subhalo on (model 3, factor 1e-5) vs off",
             "", "Truth = flat kappa_thr=3e-5 (two independent halves, combined). "
             "JSD in nats, 120 shared bins; floor_pred = half-split floor rescaled "
             "by (1/N1+1/N2); excess = max(JSD - floor, 0).", ""]
    for gk, g in res.items():
        lines += [f"## {gk}   (floor_half={g['floor_half']:.2e}, n_truth={g['n_truth']:,})", "",
                  "| rule | kthr_eff | <N> | JSD | floor | JSD excess | sigma(lnmu) | <1/mu> | q99 | q99.9 | q99.99 |",
                  "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r, d in g["rules"].items():
            t = d["tails"]
            lines.append(
                f"| {r} | {d['kthr_eff']:.2e} | {d['Nexp']:.0f} | {d['jsd']:.2e} | "
                f"{d['floor_pred']:.2e} | {d['jsd_excess']:.2e} | {d['sigma_lnmu']:.4f} | "
                f"{d['inv_mu']:.4f} | {t['q99'][1]:.3f} | {t['q999'][1]:.3f} | {t['q9999'][1]:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--zs", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument("--arms", nargs="+", choices=["on", "off"], default=["on", "off"])
    args = ap.parse_args()
    for z in args.zs:
        if z not in Z_TABLE:
            ap.error(f"--zs {z:g} unsupported (no seed/size entry); "
                     f"supported: {sorted(Z_TABLE)} — extend Z_TABLE to add it")

    outdir = REPO / "data" / "results" / f"kappathr_subhalo_jsd{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    meta = run_sampling(outdir, args.zs, args.arms, args.scale)
    res = analyze(outdir, meta, args.zs, args.arms)
    (outdir / "summary.json").write_text(json.dumps(res, indent=1))
    txt = report(res)
    (outdir / "report.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
