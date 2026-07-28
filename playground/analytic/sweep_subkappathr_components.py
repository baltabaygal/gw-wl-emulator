"""
Population-weighted variance DECOMPOSITION of the substructure term vs kappa_thr,sub.

The earlier sweeps kept only sigma_total. This one keeps all three pieces of the exact
Campbell identity (sigma_vs_subkappathr.py, ~line 165)

    Var_sub,tot = Var_host + Var_clumps + 2 Cov(host, clumps)

where, per host and per ray,
    Var_clumps = the scatter of the rendered clump sum  sum_i kappa_i
    Var_host   = the response of the SMOOTH host to being carved by the realized
                 substructure mass, (dkappa_host/dM)^2 * Var(sum_i m_i)
    Cov        = -(dkappa_host/dM) * <sum_i m_i kappa_i>, NEGATIVE by construction:
                 a realization with more clump mass near the ray has a lighter host
Every term is linear in the per-sightline host weight dNh, so all three
population-weight by the same plain sum and the identity survives aggregation exactly.
2Cov is recovered as Var_tot - Var_host - Var_clumps rather than accumulated separately,
which makes the identity exact by construction at the population level too.

This is the population version of what Fig. 4 shows for a single host against radius. The
useful readings are (a) the clump sum dominates the host response by roughly an order of
magnitude in sigma, and (b) the negative covariance is not negligible -- treating the
carve as a deterministic mass reduction would misstate the substructure variance.

z_s = 1 only (the redshift the MC sweep exists at); the absolute kappa_thr,sub grid
matches sweep_subkappathr_dense.py so the two outputs can be plotted on one axis.

Run (resumable in slices; own node cache):
    DENSE_BUDGET_S=36 python playground/analytic/sweep_subkappathr_components.py
Writes playground/analytic/subkappathr_components.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playground.analytic.sigma_vs_subkappathr import run_kthr  # noqa: E402
from playground.analytic.sweep_subkappathr_dense import KGRID  # noqa: E402
from playground.analytic.sweep_subkappathr_population import (  # noqa: E402
    M_NODES, ZL_FRACS, interp_to_grid, load_weights,
)

Z_SOURCE = 1.0
BUDGET_S = float(os.environ.get("DENSE_BUDGET_S", "1e9"))
CACHE = ROOT / "tmp" / "subkappathr_component_nodes_v2"
OUT = ROOT / "playground" / "analytic" / "subkappathr_components.json"

# Three variance channels, the retained count, and the MEANS. The means were added
# 2026-07-27 after the single-host spot check showed the aperture-mean convergence is NOT
# conserved by the threshold (+1 to +3% per host at 1e13..1e15) even though the variance
# is preserved to 0.1%. Means are linear in the host weight dNh exactly as variances are
# (compound-Poisson), so they aggregate by the same weighted sum.
#   kappa_total  = <kappa_carved_host + kappa_clumps>, the perturbed halo
#   kappa_sub    = <kappa_clumps> alone
#   kappa_nosub  = <kappa> of the SAME host with no substructure at all (meta k_ns_mean,
#                  threshold independent; broadcast so every channel has one shape)
KEYS = ("var_host", "var_clumps", "var_tot", "n_retained",
        "kappa_total", "kappa_sub", "kappa_nosub")


def node_table(zs, kappa_thr, kthr_subs, budget_s=1e9):
    """Per-(M, z_l) node: the three variance channels and the retained clump count."""
    CACHE.mkdir(parents=True, exist_ok=True)
    zl_nodes = ZL_FRACS * zs
    nk = len(kthr_subs)
    out = {k: np.full((len(M_NODES), len(zl_nodes), nk), np.nan) for k in KEYS}

    t0, missing = time.time(), 0
    for i, M in enumerate(M_NODES):
        f = CACHE / f"zs{zs}_nk{nk}_M{i:02d}.npz"
        if f.exists():
            d = np.load(f)
            for k in KEYS:
                out[k][i] = d[k]
            continue
        if time.time() - t0 > budget_s:
            missing += 1
            continue
        for j, zl in enumerate(zl_nodes):
            meta, rows = run_kthr(float(M), float(zl), zs, kappa_thr, list(kthr_subs))
            out["var_host"][i, j] = [r["sigma_host"] ** 2 for r in rows]
            out["var_clumps"][i, j] = [r["sigma_sub"] ** 2 for r in rows]
            out["var_tot"][i, j] = [r["sigma_total"] ** 2 for r in rows]
            out["n_retained"][i, j] = [r["n_retained"] for r in rows]
            out["kappa_total"][i, j] = [r["kappa_total"] for r in rows]
            out["kappa_sub"][i, j] = [r["kappa_sub"] for r in rows]
            out["kappa_nosub"][i, j] = meta["k_ns_mean"]      # threshold independent
        np.savez(f, **{k: out[k][i] for k in KEYS})
        print(f"  M={M:.2e} done ({time.time() - t0:.0f}s)", flush=True)

    if missing:
        raise SystemExit(f"INCOMPLETE: {missing}/{len(M_NODES)} M nodes left; rerun")
    return zl_nodes, out


def main() -> None:
    head, zl, M, w, _rm = load_weights(ROOT / "tmp" / f"host_weights_zs{Z_SOURCE}.txt")
    kappa_thr = head["kappathr"]
    print(f"=== z_s={Z_SOURCE}  kappa_thr={kappa_thr:.4e}  "
          f"<N_host>={head['NhfNFW']:.2f} ===", flush=True)

    zl_nodes, tab = node_table(Z_SOURCE, kappa_thr, KGRID, budget_s=BUDGET_S)

    agg = {k: [] for k in KEYS}
    for idx in range(len(KGRID)):
        for k in KEYS:
            g = interp_to_grid(M_NODES, zl_nodes, tab[k][:, :, idx], M, zl)
            agg[k].append(float(np.sum(w * g)))

    var_host = np.array(agg["var_host"])
    var_clumps = np.array(agg["var_clumps"])
    var_tot = np.array(agg["var_tot"])
    twocov = var_tot - var_host - var_clumps          # exact by construction

    OUT.write_text(json.dumps({
        "z_source": Z_SOURCE, "kappa_thr": kappa_thr, "N_host": head["NhfNFW"],
        "kthr_sub": KGRID.tolist(),
        "var_host": var_host.tolist(), "var_clumps": var_clumps.tolist(),
        "var_tot": var_tot.tolist(), "twocov": twocov.tolist(),
        "clumps_per_ray": agg["n_retained"],
        "kappa_total": agg["kappa_total"], "kappa_sub": agg["kappa_sub"],
        "kappa_nosub": agg["kappa_nosub"],
    }, indent=2))
    print(f"saved {OUT}")

    print(f"\n{'kthr_sub':>11} {'clumps':>10} {'sig_tot':>10} {'sig_clumps':>11} "
          f"{'sig_host':>10} {'2Cov/Var':>9} {'sub/host':>9}")
    for i in (0, 1, 12, 20, 25, 30, 35, 40, 45, 49):
        if i >= len(KGRID):
            continue
        k = KGRID[i]
        st, sc, sh = np.sqrt(var_tot[i]), np.sqrt(var_clumps[i]), np.sqrt(var_host[i])
        print(f"{k:11.3e} {agg['n_retained'][i]:10.4g} {st:10.4e} {sc:11.4e} "
              f"{sh:10.4e} {twocov[i] / var_tot[i]:9.4f} {sc / sh:9.3f}")

    kd = 0.1 * kappa_thr
    j = int(np.argmin(np.abs(np.log(KGRID[1:] / kd)))) + 1
    print(f"\nat the production default kappa_thr,sub = 0.1 kappa_thr = {kd:.3e}:")
    print(f"  sigma_clumps/sigma_host = {np.sqrt(var_clumps[j] / var_host[j]):.3f}, "
          f"2Cov/Var_tot = {twocov[j] / var_tot[j]:.4f}, "
          f"Var_clumps/Var_tot = {var_clumps[j] / var_tot[j]:.4f}")
    print(f"  identity check: (Var_host+Var_clumps+2Cov)/Var_tot - 1 = "
          f"{(var_host[j] + var_clumps[j] + twocov[j]) / var_tot[j] - 1:.2e}")

    # ---- the MEAN, which the threshold does NOT preserve as well as the variance -----
    kt = np.array(agg["kappa_total"])
    ksb = np.array(agg["kappa_sub"])
    kns = np.array(agg["kappa_nosub"])
    print(f"\nMEAN convergence per sightline (population weighted):")
    print(f"  no substructure at all          <kappa> = {kns[0]:.6e}")
    print(f"  fully populated (kthr_sub=0)    <kappa> = {kt[0]:.6e}  "
          f"({100 * (kt[0] / kns[0] - 1):+.3f}% vs no-sub)")
    print(f"  at the default 0.1 kappa_thr    <kappa> = {kt[j]:.6e}  "
          f"({100 * (kt[j] / kns[0] - 1):+.3f}% vs no-sub)")
    print(f"  -> the threshold moves <kappa> by {100 * (kt[j] / kt[0] - 1):+.3f}% of the "
          f"total, i.e. {100 * (kt[j] - kt[0]) / (kt[0] - kns[0]):+.1f}% of the whole "
          f"substructure mean excess")
    print(f"  <kappa_clumps> alone: {ksb[0]:.4e} -> {ksb[j]:.4e} "
          f"({100 * (ksb[j] / ksb[0] - 1):+.1f}%, i.e. the dropped clumps carried "
          f"{100 * (1 - ksb[j] / ksb[0]):.0f}% of the mean clump convergence)")
    print(f"\n{'kthr_sub':>11} {'<k_tot>':>12} {'vs kthr=0':>10} {'<k_sub>':>12} "
          f"{'sigma_tot':>11} {'vs kthr=0':>10}")
    for i in (0, 1, 12, 20, 25, 30, 35, 40, 45):
        if i >= len(KGRID):
            continue
        print(f"{KGRID[i]:11.3e} {kt[i]:12.5e} {100 * (kt[i] / kt[0] - 1):+9.3f}% "
              f"{ksb[i]:12.4e} {np.sqrt(var_tot[i]):11.4e} "
              f"{100 * (np.sqrt(var_tot[i] / var_tot[0]) - 1):+9.3f}%")
    print("\nCAVEAT: a rising <kappa> with threshold is the expected signature of the "
          "untruncated-NFW clump approximation -- the carve removes the BOUND mass m_i "
          "while the clump is rendered as an untruncated profile that leaks convergence "
          "outside the aperture, so moving that mass into the fully-enclosed host profile "
          "puts MORE kappa inside. It is a model caveat, not a bug in the threshold.")


if __name__ == "__main__":
    main()
