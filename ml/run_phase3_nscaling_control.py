"""Phase 3D control: is the 'JSD rises and saturates with N' curve reproduced
sim-vs-sim?

Phase 3D concluded that JSD/TV increasing toward ln 2 as catalog size N grows
proves a systematic NSF bias. But that saturation is the generic behaviour of
comparing two over-concentrated posteriors whose peaks differ by ANY nonzero
amount -- including two independent Monte-Carlo simulator references.

This script computes, on the same 12x12 grid and same N sweep:
  - JSD(sim_seedA, sim_seedB)  -- the reference against a second copy of itself
  - JSD(nsf, sim_seedA)        -- the NSF against the reference
If the sim-vs-sim curve also rises and saturates at ln 2, the N-scaling result
is a metric artifact, not evidence of NSF bias.

Trick: the simulator PMF at each (z, theta) is independent of catalog size N.
We compute two PMF grids (two seeds) once, then re-weight by catalog counts for
each N. Catalogs are nested subsamples of one truth draw so the sweep is clean.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from ml.phase3_common import (
    build_grid,
    ensure_dir,
    import_gwlensing,
    load_nsf_model,
    normalize_log_grid,
)
from ml.nsf_likelihood import nsf_mixed_catalog_log_likelihood
from ml.run_phase3_posterior_grid import load_bin_edges

_BIN = _NSIM = _SEED = _ZS = None


def _init(zs, bin_edges, nsim, seed):
    global _BIN, _NSIM, _SEED, _ZS
    _ZS, _BIN, _NSIM, _SEED = zs, bin_edges, nsim, seed


def _pmf_stack(theta):
    gw = import_gwlensing()
    h, om, s8 = [float(x) for x in theta]
    stack = []
    for z in _ZS:
        res = gw.sample_lnmu_ml_with_diagnostics(float(z), h, om, s8, int(_NSIM), int(_SEED), False)
        lnmu = np.asarray(res["lnmu"], dtype=np.float64)
        lnmu = lnmu[np.isfinite(lnmu)]
        clamped = np.clip(lnmu, _BIN[0] + 1e-9, _BIN[-1] - 1e-9)
        counts, _ = np.histogram(clamped, bins=_BIN)
        p = counts / max(float(counts.sum()), 1.0)
        stack.append(p)
    return np.array(stack)  # (n_z, n_bins)


def sim_pmf_grid(theta_grid, zs, bin_edges, nsim, seed, workers=10):
    with mp.Pool(workers, initializer=_init, initargs=(zs, bin_edges, nsim, seed)) as pool:
        stacks = pool.map(_pmf_stack, list(theta_grid))
    return np.array(stacks)  # (n_theta, n_z, n_bins)


def loglik_from_pmf(pmf_grid, counts_by_z):
    # pmf_grid: (n_theta, n_z, n_bins); counts_by_z: (n_z, n_bins)
    logp = np.log(pmf_grid + 1e-12)
    return np.einsum("tzb,zb->t", logp, counts_by_z)


def jsd(p, q):
    eps = 1e-15
    return float(jensenshannon(p.ravel() + eps, q.ravel() + eps, base=np.e) ** 2)


def tv(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


def make_truth_catalog(zs, n_per_z_max, truth, bin_edges, seed=610001):
    """Draw at least n_per_z_max VALID events per z at truth (the gammaj>0 filter
    drops a fraction of raw samples, so we accumulate until we have enough)."""
    gw = import_gwlensing()
    h, om, s8 = truth
    per_z = {}
    draw = max(4 * n_per_z_max, 20000)
    for i, z in enumerate(zs):
        acc = []
        attempt = 0
        while sum(len(a) for a in acc) < n_per_z_max:
            res = gw.sample_lnmu_ml_with_diagnostics(float(z), h, om, s8, int(draw), int(seed + i + 1000 * attempt), False)
            lnmu = np.asarray(res["lnmu"], dtype=np.float64)
            acc.append(lnmu[np.isfinite(lnmu)])
            attempt += 1
        per_z[z] = np.concatenate(acc)[:n_per_z_max]
    return per_z


def subsamples_for_N(per_z, zs, N):
    """Return per-z lnmu arrays of length N//len(zs) (consistent for counts and NSF)."""
    per = N // len(zs)
    return {z: per_z[z][:per] for z in zs}


def counts_for_N(per_z, zs, N, bin_edges):
    subs = subsamples_for_N(per_z, zs, N)
    counts = []
    for z in zs:
        clamped = np.clip(subs[z], bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
        c, _ = np.histogram(clamped, bins=bin_edges)
        counts.append(c)
    return np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--Ns", type=int, nargs="+", default=[250, 1000, 5000])
    ap.add_argument("--nsim", type=int, default=10000)
    ap.add_argument("--seedA", type=int, default=100)
    ap.add_argument("--seedB", type=int, default=777)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--zs", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    ap.add_argument("--model_path", default="data/models/conditional_nsf_backend_current.pt")
    ap.add_argument("--out_json", default="data/results/phase3_nscaling_control.json")
    ap.add_argument("--out_md", default="docs/phase3_nscaling_control.md")
    args = ap.parse_args()

    truth = (0.67, 0.30, 0.85)
    zs = list(args.zs)
    bin_edges = load_bin_edges()
    grid = build_grid("2d", args.resolution, fixed_h=truth[0])
    theta_grid = grid["theta"]
    shape = tuple(len(v) for v in grid["axes"].values())

    print("Computing simulator PMF grid (seed A)...", flush=True)
    pmf_A = sim_pmf_grid(theta_grid, zs, bin_edges, args.nsim, args.seedA, args.workers)
    print("Computing simulator PMF grid (seed B)...", flush=True)
    pmf_B = sim_pmf_grid(theta_grid, zs, bin_edges, args.nsim, args.seedB, args.workers)

    print("Drawing nested truth catalog...", flush=True)
    n_max = max(args.Ns) // len(zs) + 1
    per_z = make_truth_catalog(zs, n_max, truth, bin_edges)

    model = load_nsf_model(args.model_path)

    rows = []
    for N in args.Ns:
        counts = counts_for_N(per_z, zs, N, bin_edges)
        post_A = normalize_log_grid(loglik_from_pmf(pmf_A, counts).reshape(shape))
        post_B = normalize_log_grid(loglik_from_pmf(pmf_B, counts).reshape(shape))
        # NSF posterior at this N: build flat lnmu+z arrays from the same subsamples
        subs = subsamples_for_N(per_z, zs, N)
        z_arr, lnmu_arr = [], []
        for z in zs:
            z_arr += [z] * len(subs[z])
            lnmu_arr += list(subs[z])
        nsf_ll = nsf_mixed_catalog_log_likelihood(model, np.array(z_arr), np.array(lnmu_arr), theta_grid,
                                                  likelihood_mode="simulator_compatible", bin_edges=bin_edges).reshape(shape)
        post_nsf = normalize_log_grid(nsf_ll)
        r = {
            "N": N,
            "jsd_sim_vs_sim": round(jsd(post_A, post_B), 6),
            "tv_sim_vs_sim": round(tv(post_A, post_B), 6),
            "jsd_nsf_vs_sim": round(jsd(post_nsf, post_A), 6),
            "tv_nsf_vs_sim": round(tv(post_nsf, post_A), 6),
        }
        rows.append(r)
        print(f"N={N:5d}  sim-vs-sim JSD={r['jsd_sim_vs_sim']:.4f}  nsf-vs-sim JSD={r['jsd_nsf_vs_sim']:.4f}", flush=True)

    ln2 = float(np.log(2))
    ensure_dir(Path(args.out_json).parent)
    Path(args.out_json).write_text(json.dumps({"ln2": ln2, "nsim": args.nsim, "seeds": [args.seedA, args.seedB], "rows": rows}, indent=2))

    L = ["# Phase 3D Control: sim-vs-sim N-scaling", "",
         f"Max JSD (ln 2) = {ln2:.6f}. Two simulator references at seeds {args.seedA}/{args.seedB}, nsim={args.nsim}.",
         "Catalogs are nested subsamples of one truth draw.", "",
         "| N | sim-vs-sim JSD | sim-vs-sim TV | nsf-vs-sim JSD | nsf-vs-sim TV |",
         "|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['N']} | {r['jsd_sim_vs_sim']} | {r['tv_sim_vs_sim']} | {r['jsd_nsf_vs_sim']} | {r['tv_nsf_vs_sim']} |")
    L += ["", "If the sim-vs-sim column also rises toward ln 2 with N, the Phase 3D",
          "scaling curve is a metric artifact of comparing over-concentrated posteriors,",
          "not evidence of systematic NSF bias.", ""]
    ensure_dir(Path(args.out_md).parent)
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print(f"\nWrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
