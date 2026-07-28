"""
Does the model-5 per-clump threshold still work across the whole source-redshift range?

The 2026-07-27 sweeps established the threshold at z_s = 0.5/1/5 only. This extends the
population-weighted Campbell calculation to z_s in [0.2, 10] on a mult grid, where
mult = kappa_thr,sub / kappa_thr(z_s) -- i.e. the production knob
`subhalo_kappathr_factor`, whose default is 0.1. Two curves per z_s:

    sigma_sub(mult)/sigma_sub(0)   what fraction of the substructure scatter survives
    clumps_per_ray(mult)           what it costs to render

SCOPE -- read this before quoting anything. This script predicts the THRESHOLD
DEPENDENCE (the shape) and the COST, both of which are pure substructure quantities the
Campbell quadrature computes exactly. It does NOT predict the amplitude of the
substructure boost sigma_ON/sigma_OFF, because that needs Var of the host + smooth field,
and the MC measures that on a kappa<=1 clipped estimator while the Campbell integral is
unclipped -- CLAUDE.md item 13 warns the law-level and clipped variances differ by tens
of percent, and the gap grows with z_s exactly where clipping bites hardest. So the boost
amplitude per z_s comes from MC (scripts/convergence/subhalo_kappathr_zs_mc.py), not from
here. Mixing the two would manufacture a z_s trend out of an estimator mismatch.

Host weights: the cached production-engine dumps (tmp/host_weights_zs{0.5,1.0,5.0}.txt)
where they exist, otherwise the validated Python replica in host_weights_py.py -- which
reproduces the engine's kappa_thr to 0.5%, NhfNFW to 0.02% and the per-cell weights to a
median 0.02%. The provenance of each z_s is recorded in the output json.

Run (resumable in slices; node cache in tmp/subkappathr_dense_nodes/):
    DENSE_BUDGET_S=30 python playground/analytic/sweep_subkappathr_zs.py [zs]
Writes playground/analytic/subkappathr_zs.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground.analytic.host_weights_py import (  # noqa: E402
    kappa_thr_fixed_N, load_engine_dump, weight_grid,
)
from playground.analytic.sweep_subkappathr_population import (  # noqa: E402
    M_NODES, ZL_FRACS, interp_to_grid,
)
from playground.analytic.sweep_subkappathr_dense import node_table  # noqa: E402

# kappa_thr,sub as a multiple of the host counting threshold. 0.0 = render everything,
# which normalizes sigma_sub; the default is mult = 0.1.
MULTS = np.concatenate([[0.0], np.logspace(-4.0, 4.0, 33)])

Z_SOURCES = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0)
BUDGET_S = float(os.environ.get("DENSE_BUDGET_S", "1e9"))
OUT = ROOT / "playground" / "analytic" / "subkappathr_zs.json"


def weights_for(zs: float):
    """(zl, M, dNh, kappa_thr, source) -- engine dump if cached, else Python replica."""
    dump = ROOT / "tmp" / f"host_weights_zs{zs}.txt"
    if dump.exists():
        head, zl, M, w, _rm = load_engine_dump(dump)
        return zl, M, w, head["kappathr"], "engine_probe"
    kthr = kappa_thr_fixed_N(zs)
    zl, M, w, _rm = weight_grid(zs, kthr)
    return zl, M, w, kthr, "python_replica"


def main() -> None:
    results = json.loads(OUT.read_text())["results"] if OUT.exists() else {}
    todo = [z for z in Z_SOURCES if str(z) not in results]
    if len(sys.argv) > 1:
        todo = [float(sys.argv[1])]

    for zs in todo:
        zl, M, w, kappa_thr, src = weights_for(zs)
        print(f"\n=== z_s={zs}  kappa_thr={kappa_thr:.4e}  <N_host>={w.sum():.2f} "
              f"  weights: {src} ===", flush=True)

        kthr_subs = MULTS * kappa_thr
        try:
            zl_nodes, nret, var = node_table(zs, kappa_thr, kthr_subs, budget_s=BUDGET_S)
        except SystemExit as e:
            # out of wall-clock budget: the finished nodes are cached, so just move on
            # and let the next invocation resume this z_s
            print(f"  {e}", flush=True)
            continue

        clumps, var_sub = [], []
        for k in range(len(MULTS)):
            n_g = interp_to_grid(M_NODES, zl_nodes, nret[:, :, k], M, zl)
            v_g = interp_to_grid(M_NODES, zl_nodes, var[:, :, k], M, zl)
            clumps.append(float(np.sum(w * n_g)))
            var_sub.append(float(np.sum(w * v_g)))
        sig_ratio = np.sqrt(np.array(var_sub) / var_sub[0])

        results[str(zs)] = {
            "kappa_thr": kappa_thr, "N_host": float(w.sum()), "weight_source": src,
            "mults": MULTS.tolist(), "clumps_per_ray": clumps,
            "var_sub": var_sub, "sigma_sub_ratio": sig_ratio.tolist(),
        }

        i_def = int(np.argmin(np.abs(MULTS - 0.1)))
        print(f"  at the production default mult=0.1: clumps/ray={clumps[i_def]:.4g} "
              f"({clumps[0] / clumps[i_def]:.3g}x fewer than all "
              f"{clumps[0]:.3g}), sigma_sub loss={100 * (1 - sig_ratio[i_def]):.3f}%")

        OUT.write_text(json.dumps({"MULTS": MULTS.tolist(), "results": results}, indent=2))
        print(f"  saved {OUT.name} ({len(results)} z_s)", flush=True)

    # ---- summary table across everything computed so far -------------------------
    zz = sorted(float(k) for k in results)
    print("\n  z_s   kappa_thr    <N_h>   clumps/ray(all)  at mult=0.1: clumps   "
          "reduction  sigma_sub loss   weights")
    for z in zz:
        r = results[str(z)]
        m = np.array(r["mults"])
        i = int(np.argmin(np.abs(m - 0.1)))
        c0, ci = r["clumps_per_ray"][0], r["clumps_per_ray"][i]
        loss = 100 * (1 - r["sigma_sub_ratio"][i])
        print(f"{z:6.2f}  {r['kappa_thr']:.3e}  {r['N_host']:6.1f}  {c0:15.4g}  "
              f"{ci:15.4g}  {c0 / ci:9.3g}  {loss:12.3f}%   {r['weight_source']}")


if __name__ == "__main__":
    main()
