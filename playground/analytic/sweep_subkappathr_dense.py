"""
Dense population-weighted kappa_thr,sub sweep on an ABSOLUTE threshold grid.

Companion to `sweep_subkappathr_population.py` (2026-07-27), which evaluated only
5 multiples of the host counting threshold {0, 0.01, 0.1, 1, 10}. That grid is
adequate for a table but too coarse to draw a curve, and it stops at
10*kappa_thr = 1.3e-3 while the Monte Carlo sweep
(`playground/sigma_on_off_vs_subkappathr_model5_crn.json`) runs out to
kappa_thr,sub = 10, where the retained population is empty.

This script evaluates the same Campbell quadrature on a dense ABSOLUTE grid so the
analytic prediction can be overlaid point-for-point on the MC. Everything else --
host weights from the production engine, the aperture convention, the interpolation
onto the 99x99 engine grid -- is identical to the coarse version; the only changes
are the threshold grid and that the output carries the absolute kappa_thr,sub axis.

Outputs, per z_s:
    var_sub(kappa_thr,sub)      population-weighted substructure Var(kappa)
    clumps_per_ray(kappa_thr,sub)   expected rendered clumps on a sightline
Both are linear in the per-sightline host weight dNh, so they aggregate by a plain
weighted sum (hosts are Poisson-independent).

Run (repo root, needs the cached host-weight dumps in tmp/):
    python playground/analytic/sweep_subkappathr_dense.py
Writes playground/analytic/subkappathr_dense.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground.analytic.sigma_vs_subkappathr import run_kthr  # noqa: E402
from playground.analytic.sweep_subkappathr_population import (  # noqa: E402
    M_NODES,
    ZL_FRACS,
    interp_to_grid,
    load_weights,
)

# Absolute kappa_thr,sub grid. Matches the span of the MC sweep (1e-7 .. 10) with
# 0.0 prepended as the "render everything" reference that normalizes sigma/sigma0.
KGRID = np.concatenate([[0.0], np.logspace(-7.0, 1.0, 49)])

Z_SOURCES = (0.5, 1.0, 5.0)

# wall-clock budget per invocation (the node cache makes the sweep resumable, which
# matters when the runner caps a single call at well under the ~70 s a z_s needs)
BUDGET_S = float(__import__("os").environ.get("DENSE_BUDGET_S", "1e9"))


def _one_node(args):
    """Worker: Campbell quadrature at a single (M, z_l) node, all thresholds."""
    M, zl, zs, kappa_thr, kthr_subs = args
    _, rows = run_kthr(float(M), float(zl), zs, kappa_thr, list(kthr_subs))
    return ([r["n_retained"] for r in rows],
            [r["sigma_total"] ** 2 for r in rows])


CACHE = Path(__file__).resolve().parents[2] / "tmp" / "subkappathr_dense_nodes"


def node_table(zs, kappa_thr, kthr_subs, verbose=True, budget_s=1e9):
    """Campbell quadrature at every (M, z_l) node -> N_ret and Var_tot per threshold.

    Serial, but cached per M node under tmp/, so the sweep can be run in slices
    under a wall-clock `budget_s` and resumed. (mp.Pool is deliberately avoided:
    see CLAUDE.md item 12 -- worker faults surface as silent hangs.)
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    zl_nodes = ZL_FRACS * zs
    nk = len(kthr_subs)
    nret = np.full((len(M_NODES), len(zl_nodes), nk), np.nan)
    var = np.full_like(nret, np.nan)

    t0 = time.time()
    missing = 0
    for i, M in enumerate(M_NODES):
        # nk in the key: the dense (absolute grid) and z_s (mult grid) sweeps use
        # different threshold counts and must not clobber each other's cache
        f = CACHE / f"zs{zs}_nk{nk}_M{i:02d}.npz"
        if f.exists():
            d = np.load(f)
            if d["nret"].shape[-1] == nk:
                nret[i], var[i] = d["nret"], d["var"]
                continue
        if time.time() - t0 > budget_s:
            missing += 1
            continue
        for j, zl in enumerate(zl_nodes):
            n_j, v_j = _one_node((M, zl, zs, kappa_thr, kthr_subs))
            nret[i, j], var[i, j] = n_j, v_j
        np.savez(f, nret=nret[i], var=var[i])
        if verbose:
            print(f"  M={M:.2e} done ({time.time() - t0:.0f}s)", flush=True)

    if missing:
        raise SystemExit(f"INCOMPLETE: {missing}/{len(M_NODES)} M nodes left for "
                         f"z_s={zs}; rerun to resume (cache in {CACHE})")
    return zl_nodes, nret, var


def main() -> None:
    out_path = ROOT / "playground" / "analytic" / "subkappathr_dense.json"
    # resumable: each z_s is appended to the json so the sweep can be run in slices
    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text()).get("results", {})
    todo = [z for z in Z_SOURCES if str(z) not in results]
    if len(sys.argv) > 1:                       # optional: restrict to one z_s
        todo = [float(sys.argv[1])]
    for zs in todo:
        wpath = ROOT / "tmp" / f"host_weights_zs{zs}.txt"
        head, zl, M, w, _rmax_engine = load_weights(wpath)
        kappa_thr = head["kappathr"]
        print(f"\n=== z_s={zs}  kappa_thr={kappa_thr:.4e}  <N_host>={head['NhfNFW']:.2f} ===")

        zl_nodes, nret, var = node_table(zs, kappa_thr, KGRID, budget_s=BUDGET_S)

        clumps, var_sub = [], []
        for k in range(len(KGRID)):
            n_g = interp_to_grid(M_NODES, zl_nodes, nret[:, :, k], M, zl)
            v_g = interp_to_grid(M_NODES, zl_nodes, var[:, :, k], M, zl)
            clumps.append(float(np.sum(w * n_g)))
            var_sub.append(float(np.sum(w * v_g)))

        sig_ratio = np.sqrt(np.array(var_sub) / var_sub[0])
        results[str(zs)] = {
            "kappa_thr": kappa_thr,
            "N_host": head["NhfNFW"],
            "kthr_sub": KGRID.tolist(),
            "clumps_per_ray": clumps,
            "var_sub": var_sub,
            "sigma_sub_ratio": sig_ratio.tolist(),
        }

        # terse echo at the multiples the coarse sweep reported, as a regression check
        for mult in (0.01, 0.1, 1.0, 10.0):
            kk = mult * kappa_thr
            sr = float(np.interp(np.log(kk), np.log(KGRID[1:]), sig_ratio[1:]))
            cc = float(np.exp(np.interp(np.log(kk), np.log(KGRID[1:]),
                                        np.log(np.array(clumps[1:]) + 1e-300))))
            print(f"  mult={mult:<6g} kthr_sub={kk:.3e}  clumps/ray={cc:10.4g}  "
                  f"sigma/sigma0={sr:.5f}  loss={100 * (1 - sr):.3f}%")

        out_path.write_text(json.dumps({"M_NODES": M_NODES.tolist(),
                                        "ZL_FRACS": ZL_FRACS.tolist(),
                                        "KGRID": KGRID.tolist(),
                                        "results": results}, indent=2))
        print(f"  saved {out_path} ({sorted(results)})", flush=True)


if __name__ == "__main__":
    main()
