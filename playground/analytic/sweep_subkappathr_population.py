"""
Population-weighted kappa_thr,sub sweep: what does a per-subhalo convergence cut buy
across the WHOLE host population on a sightline, not just the fiducial 1e14 host?

Motivation (2026-07-27). `sigma_vs_subkappathr.py` showed that for M=1e14, z_l=0.5,
z_s=1 a cut at the host's own kappa_thr retains 0.15 clumps instead of 3.4e4 (2.3e5x
fewer) for a 0.2% loss in sigma_kappa. That is the lever that makes subhalo_model=4
(brute, every subhalo explicit) affordable. But cost and variance are both dominated
by whichever hosts a ray actually meets, so the single-host number cannot be quoted.

Weighting. NhfNFW (cpp/lensing.cpp:138) integrates the expected host count per
sightline as
    dNh(z_l, M) = c*pi*((1+z_l)*rmax)^2 / H(z_l) * dndlnM * dlnM * dz,
so dNh IS the expected number of hosts of that (z_l, M) class per ray. Those hosts are
Poisson-independent, so both quantities of interest are linear in dNh:
    clumps per ray            = sum dNh * <N_ret>(M, z_l, kthr_sub)
    substructure Var(kappa)   = sum dNh * Var_tot(M, z_l, kthr_sub)
<N_ret> and Var_tot come from the exact Campbell quadrature in run_kthr(), already
area-weighted over the same aperture q(y)=2y/rmax^2 that dNh's pi*rmax^2 defines, so
the product is consistent.

The weights are dumped from the PRODUCTION engine by playground/host_weight_probe.cpp
(not re-derived here) so the HMF, rmaxfNFW and H(z) are the ones the simulator uses.

Run (repo root, after building + running the probe for each zs):
  ./playground/host_weight_probe 1.0 > tmp/host_weights_zs1.0.txt
  /Users/baltabay/miniforge3/envs/test/bin/python \
      playground/analytic/sweep_subkappathr_population.py
Writes playground/analytic/subkappathr_population.{json,png}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from playground.analytic.sigma_vs_subkappathr import run_kthr  # noqa: E402
from playground.dgate.subhalo_factor_dgate_area_scan import host_rmax  # noqa: E402

# Host grid nodes for the (expensive) Campbell quadrature. The engine grid is
# 99x99; we evaluate on a coarse log grid and interpolate log(N_ret), log(Var)
# onto the full engine grid before weighting.
M_NODES = np.logspace(7.0, 16.5, 20)   # spans the engine grid (1e7..1e17); the top
                                       # two decades carry ~1e-80 weight but cost nothing
ZL_FRACS = np.array([0.08, 0.2, 0.35, 0.5, 0.65, 0.8, 0.93])   # of z_s


def load_weights(path: Path):
    """Return (zl, M, dNh, rmax) arrays + header dict from host_weight_probe output."""
    head, zl, M, w, rm = {}, [], [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            p = line[1:].split()
            if len(p) >= 2 and p[0] in ("zs", "kappathr", "NhfNFW", "total"):
                head[p[0]] = float(p[1])
            continue
        if line.startswith("W "):
            _, a, b, c, d = line.split()
            zl.append(float(a)); M.append(float(b)); w.append(float(c)); rm.append(float(d))
    return head, np.array(zl), np.array(M), np.array(w), np.array(rm)


def node_table(zs, kappa_thr, kthr_subs, verbose=True):
    """Campbell quadrature at every (M, z_l) node -> N_ret and Var_tot per threshold."""
    zl_nodes = ZL_FRACS * zs
    nret = np.zeros((len(M_NODES), len(zl_nodes), len(kthr_subs)))
    var = np.zeros_like(nret)
    t0 = time.time()
    for i, M in enumerate(M_NODES):
        for j, zl in enumerate(zl_nodes):
            _, rows = run_kthr(float(M), float(zl), zs, kappa_thr, kthr_subs)
            for k, r in enumerate(rows):
                nret[i, j, k] = r["n_retained"]
                var[i, j, k] = r["sigma_total"] ** 2
        if verbose:
            print(f"  M={M:.2e} done ({time.time()-t0:.0f}s)", flush=True)
    return zl_nodes, nret, var


def interp_to_grid(M_nodes, zl_nodes, table, M_grid, zl_grid):
    """Bilinear in (log M, z_l) on log(table); table >= 0, zeros handled by a floor."""
    floor = 1e-300
    lt = np.log(np.maximum(table, floor))
    lM_n, lM_g = np.log(M_nodes), np.log(M_grid)
    # interpolate in log M first (axis 0), then in z_l (axis 1)
    out_m = np.empty((len(lM_g), lt.shape[1]))
    for j in range(lt.shape[1]):
        out_m[:, j] = np.interp(lM_g, lM_n, lt[:, j])
    out = np.empty(len(lM_g))
    for i in range(len(lM_g)):
        out[i] = np.interp(zl_grid[i], zl_nodes, out_m[i])
    return np.exp(out)


def main() -> None:
    results = {}
    for zs in (0.5, 1.0, 5.0):
        wpath = ROOT / "tmp" / f"host_weights_zs{zs}.txt"
        head, zl, M, w, rmax_engine = load_weights(wpath)
        kappa_thr = head["kappathr"]
        print(f"\n=== z_s={zs}  kappa_thr={kappa_thr:.4e}  <N_host>={head['NhfNFW']:.2f} ===")

        # aperture cross-check: engine rmaxfNFW vs the Python host_rmax the
        # Campbell quadrature uses. A mismatch would make dNh and <N_ret>
        # refer to different apertures and silently bias the aggregate.
        sel = w > w.max() * 1e-6
        idx = np.argsort(w[sel])[-200:]
        rp = np.array([host_rmax(float(m), float(z), zs, kappa_thr)
                       for m, z in zip(M[sel][idx], zl[sel][idx])])
        rat = rp / rmax_engine[sel][idx]
        print(f"  aperture check (200 heaviest cells): rmax_py/rmax_engine "
              f"median={np.median(rat):.5f} min={rat.min():.5f} max={rat.max():.5f}")

        # thresholds expressed as multiples of the host counting threshold, so the
        # "self-consistent" choice kappa_thr,sub = kappa_thr is the 1.0 entry.
        mults = [0.0, 0.01, 0.1, 1.0, 10.0]
        kthr_subs = [m * kappa_thr for m in mults]
        zl_nodes, nret, var = node_table(zs, kappa_thr, kthr_subs)

        row = {"kappa_thr": kappa_thr, "N_host": head["NhfNFW"],
               "aperture_ratio_median": float(np.median(rat)), "mults": mults,
               "clumps_per_ray": [], "var_sub": []}
        for k in range(len(mults)):
            n_g = interp_to_grid(M_NODES, zl_nodes, nret[:, :, k], M, zl)
            v_g = interp_to_grid(M_NODES, zl_nodes, var[:, :, k], M, zl)
            row["clumps_per_ray"].append(float(np.sum(w * n_g)))
            row["var_sub"].append(float(np.sum(w * v_g)))
        row["clumps_per_ray_full"] = row["clumps_per_ray"][0]
        row["clumps_per_ray_cut"] = row["clumps_per_ray"][mults.index(1.0)]
        row["var_sub_full"] = row["var_sub"][0]
        row["var_sub_cut"] = row["var_sub"][mults.index(1.0)]
        row["clump_reduction"] = row["clumps_per_ray_full"] / row["clumps_per_ray_cut"]
        row["sigma_ratio"] = float(np.sqrt(row["var_sub_cut"] / row["var_sub_full"]))

        print("   mult  kappa_thr,sub   clumps/ray   reduction   sigma/sigma0   loss%")
        for k, m in enumerate(mults):
            red = row["clumps_per_ray"][0] / row["clumps_per_ray"][k]
            sr = np.sqrt(row["var_sub"][k] / row["var_sub"][0])
            print("  %5.2f  %13.4e  %11.4g  %10.4g  %13.5f  %6.3f"
                  % (m, m * kappa_thr, row["clumps_per_ray"][k], red, sr, 100 * (1 - sr)))

        # where does the cost sit? cumulative clump budget vs host mass
        n_full = interp_to_grid(M_NODES, zl_nodes, nret[:, :, 0], M, zl)
        contrib = w * n_full
        order = np.argsort(M)
        cM, cC = M[order], np.cumsum(contrib[order]) / contrib.sum()
        row["M_median_cost"] = float(np.interp(0.5, cC, cM))
        row["M_90pct_cost"] = float(np.interp(0.9, cC, cM))

        print(f"  clumps/ray: {row['clumps_per_ray_full']:.4g} -> "
              f"{row['clumps_per_ray_cut']:.4g}  ({row['clump_reduction']:.4g}x fewer)")
        print(f"  sigma_sub(pop) ratio: {row['sigma_ratio']:.5f} "
              f"(loss {100*(1-row['sigma_ratio']):.3f}%)")
        print(f"  clump cost median host mass {row['M_median_cost']:.3g}, "
              f"90% below {row['M_90pct_cost']:.3g}")
        results[str(zs)] = row

    out = ROOT / "playground" / "analytic" / "subkappathr_population.json"
    out.write_text(json.dumps({"M_NODES": M_NODES.tolist(),
                               "ZL_FRACS": ZL_FRACS.tolist(),
                               "results": results}, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
