"""Phase 3 reference-control experiments.

Goal: determine whether the Phase 3 smoke-gate failure (JSD ~ ln 2 between the
simulator-reference posterior and the NSF posterior) reflects a real emulator
defect, or a broken validation gate (over-concentrated, MC-noisy reference that
cannot reproduce itself or recover truth).

Controls:
  1. Sim-vs-sim reproducibility at fixed nsim, different RNG seeds.
  2. Sim MC-convergence: low vs high nsim_per_z.
  3. Truth recovery: where does each posterior's MAP/mean land vs injected truth?
  4. NSF-vs-sim for context, using the same 2D grid.

This is a diagnostic; it writes JSON + a markdown summary and does not touch
production caches.
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
    load_npz_with_metadata,
    load_nsf_model,
    mixed_catalog_simulator_log_likelihood,
    normalize_log_grid,
)
from ml.nsf_likelihood import nsf_mixed_catalog_log_likelihood
from ml.run_phase3_posterior_grid import load_bin_edges

_Z = _LNMU = _BIN = None
_NSIM = _SEED = None


def _init(z, lnmu, bin_edges, nsim, seed):
    global _Z, _LNMU, _BIN, _NSIM, _SEED
    _Z, _LNMU, _BIN, _NSIM, _SEED = z, lnmu, bin_edges, nsim, seed


def _work(theta):
    return mixed_catalog_simulator_log_likelihood(
        _Z, _LNMU, theta, _BIN, nsim_per_z=int(_NSIM), seed=int(_SEED)
    )


def sim_grid(z, lnmu, theta, shape, bin_edges, nsim, seed, workers=10):
    with mp.Pool(workers, initializer=_init, initargs=(z, lnmu, bin_edges, nsim, seed)) as pool:
        vals = np.array(pool.map(_work, list(theta)), dtype=np.float64)
    return vals.reshape(shape)


def summarize(axes, log_like):
    post = normalize_log_grid(log_like)
    names = list(axes)
    mesh = np.meshgrid(*[axes[n] for n in names], indexing="ij")
    coords = {n: m.ravel() for n, m in zip(names, mesh)}
    flat = post.ravel()
    imap = int(np.argmax(flat))
    out = {"post": post}
    for n in names:
        v = coords[n]
        mean = float(np.sum(v * flat))
        std = float(np.sqrt(np.sum((v - mean) ** 2 * flat)))
        out[n] = {"MAP": float(v[imap]), "mean": mean, "std": std}
    return out


def jsd_tv(p, q):
    eps = 1e-15
    jsd = float(jensenshannon(p.ravel() + eps, q.ravel() + eps, base=np.e) ** 2)
    tv = float(0.5 * np.sum(np.abs(p - q)))
    return jsd, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/mock_catalogs/phase3/mixed_uniform_central_N1000_seed610001_57cb44ad1ef0f9d4.npz")
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 777, 2024])
    ap.add_argument("--nsim_lo", type=int, default=10000)
    ap.add_argument("--nsim_hi", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out_json", default="data/results/phase3_reference_controls.json")
    ap.add_argument("--out_md", default="docs/phase3/phase3_reference_controls.md")
    args = ap.parse_args()

    arrays, md = load_npz_with_metadata(args.catalog)
    z = arrays["z"].astype(np.float64)
    lnmu = arrays["lnmu"].astype(np.float64)
    truth = {"h": float(md["true_h"]), "OmegaM": float(md["true_OmegaM"]), "sigma8": float(md["true_sigma8"])}
    grid = build_grid("2d", args.resolution, fixed_h=truth["h"])
    axes = grid["axes"]
    shape = tuple(len(v) for v in axes.values())
    bin_edges = load_bin_edges()

    print(f"Truth: {truth}")
    print(f"Grid 2d res {args.resolution}, shape {shape}")

    sims = {}
    # reproducibility cloud at nsim_lo
    for s in args.seeds:
        print(f"[sim] nsim={args.nsim_lo} seed={s} ...", flush=True)
        sims[(args.nsim_lo, s)] = sim_grid(z, lnmu, grid["theta"], shape, bin_edges, args.nsim_lo, s, args.workers)
    # high-stat convergence at first seed
    print(f"[sim] nsim={args.nsim_hi} seed={args.seeds[0]} ...", flush=True)
    sims[(args.nsim_hi, args.seeds[0])] = sim_grid(z, lnmu, grid["theta"], shape, bin_edges, args.nsim_hi, args.seeds[0], args.workers)

    # NSF posterior (simulator_compatible) on same grid
    print("[nsf] simulator_compatible ...", flush=True)
    model = load_nsf_model()
    nsf_ll = nsf_mixed_catalog_log_likelihood(model, z, lnmu, grid["theta"], likelihood_mode="simulator_compatible", bin_edges=bin_edges).reshape(shape)

    # summaries
    summ = {f"sim_nsim{n}_seed{s}": summarize(axes, ll) for (n, s), ll in sims.items()}
    summ["nsf_compatible"] = summarize(axes, nsf_ll)

    def loc(key):
        d = summ[key]
        return {p: {"MAP": d[p]["MAP"], "mean": round(d[p]["mean"], 4), "std": round(d[p]["std"], 4)} for p in axes}

    # pairwise JSD among the seed cloud
    seed_keys = [f"sim_nsim{args.nsim_lo}_seed{s}" for s in args.seeds]
    repro = {}
    for i in range(len(seed_keys)):
        for j in range(i + 1, len(seed_keys)):
            jsd, tv = jsd_tv(summ[seed_keys[i]]["post"], summ[seed_keys[j]]["post"])
            repro[f"{seed_keys[i]} vs {seed_keys[j]}"] = {"jsd": round(jsd, 6), "tv": round(tv, 6)}

    # convergence: lo vs hi nsim, same seed
    lo_key = f"sim_nsim{args.nsim_lo}_seed{args.seeds[0]}"
    hi_key = f"sim_nsim{args.nsim_hi}_seed{args.seeds[0]}"
    jsd_conv, tv_conv = jsd_tv(summ[lo_key]["post"], summ[hi_key]["post"])

    # nsf vs sim (hi-stat reference)
    jsd_ns, tv_ns = jsd_tv(summ["nsf_compatible"]["post"], summ[hi_key]["post"])

    result = {
        "truth": truth,
        "grid_shape": list(shape),
        "nsim_lo": args.nsim_lo,
        "nsim_hi": args.nsim_hi,
        "seeds": args.seeds,
        "locations": {k: loc(k) for k in summ},
        "sim_vs_sim_reproducibility_same_nsim": repro,
        "sim_mc_convergence_lo_vs_hi": {"jsd": round(jsd_conv, 6), "tv": round(tv_conv, 6)},
        "nsf_vs_sim_histat": {"jsd": round(jsd_ns, 6), "tv": round(tv_ns, 6)},
    }
    ensure_dir(Path(args.out_json).parent)
    Path(args.out_json).write_text(json.dumps(result, indent=2))

    # markdown
    L = ["# Phase 3 Reference Controls", "",
         f"Catalog: `{Path(args.catalog).name}`  N={z.size}  truth OmegaM={truth['OmegaM']} sigma8={truth['sigma8']}", "",
         "## 1. Sim-vs-sim reproducibility (same nsim, different seed)",
         "If JSD here is large, the reference cannot reproduce itself and the gate is broken.", "",
         "| Pair | JSD | TV |", "|---|---:|---:|"]
    for k, v in repro.items():
        L.append(f"| {k} | {v['jsd']} | {v['tv']} |")
    L += ["", "## 2. MC convergence (lo vs hi nsim, same seed)",
          f"- nsim {args.nsim_lo} vs {args.nsim_hi}: JSD={jsd_conv:.6f}, TV={tv_conv:.6f}", "",
          "## 3. Truth recovery (MAP / mean per posterior)",
          f"Injected truth: OmegaM={truth['OmegaM']}, sigma8={truth['sigma8']}", "",
          "| Posterior | OmegaM MAP | OmegaM mean | sigma8 MAP | sigma8 mean |",
          "|---|---:|---:|---:|---:|"]
    for k in summ:
        d = summ[k]
        L.append(f"| {k} | {d['OmegaM']['MAP']:.4f} | {d['OmegaM']['mean']:.4f} | {d['sigma8']['MAP']:.4f} | {d['sigma8']['mean']:.4f} |")
    L += ["", "## 4. NSF vs hi-stat simulator",
          f"- JSD={jsd_ns:.6f}, TV={tv_ns:.6f}", ""]
    ensure_dir(Path(args.out_md).parent)
    Path(args.out_md).write_text("\n".join(L) + "\n")

    print("\n=== SUMMARY ===")
    print("sim-vs-sim (same nsim, diff seed):", json.dumps(repro, indent=2))
    print(f"MC convergence lo-vs-hi: JSD={jsd_conv:.6f} TV={tv_conv:.6f}")
    print(f"NSF vs hi-stat sim: JSD={jsd_ns:.6f} TV={tv_ns:.6f}")
    for k in summ:
        d = summ[k]
        print(f"  {k:28s} OmegaM(MAP={d['OmegaM']['MAP']:.3f} mean={d['OmegaM']['mean']:.3f})  sigma8(MAP={d['sigma8']['MAP']:.3f} mean={d['sigma8']['mean']:.3f})")
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
