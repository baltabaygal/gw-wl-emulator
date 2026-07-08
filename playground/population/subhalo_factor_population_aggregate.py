"""
Step 3: population-level subhalo_factor deficit.

For a full line of sight the substructure variance is additive over independent
host encounters (compound Poisson, 2-halo dropped), so the production paired
excess is

    E_prod(gate) = sum_{M,z_l} Nbar(M,z_l) * e_host_nonpooled(M,z_l; gate)

with
  Nbar(M,z_l) = mean resolved host encounters / sightline  (C++ dNH[jz][jM][0]),
  e_host_nonpooled = <C + (mu+delta)^2>_y + 2 <kappa_full (mu+delta)>_y
                   (compound-Poisson per-encounter 2nd moment; NO pooled-mean
                    subtraction - that is the production form, cf.
                    subhalo_factor_analytic_deficit.run(pooled=False)).

The per-host piece uses the analytic model validated against the stratified MC
and the C++ physics (2026-07-07).  Nbar comes straight from the C++ so the host
weighting is exact.

The population retained fraction for the production proxy-r gate is

    ratio_pop(f) = E_prod(proxy,f) / E_prod(brute)
                 = sum_w w * r_host(f) / sum_w w,   w = Nbar * e0,  r = e_f/e0,

i.e. the brute-excess-weighted average of the per-host retained fraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT))

from playground.subhalo_factor_analytic_deficit import run  # noqa: E402


def load_barN(path: Path):
    header = path.read_text().splitlines()[0]
    kthr = float(header.split("kappathrH=")[1])
    zs = float(header.split("zs=")[1].split()[0])
    d = np.loadtxt(path)
    return zs, kthr, d[:, 0], d[:, 1], d[:, 2]   # zs, kthr, zl, M, barN


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--barN", type=Path, default=ROOT / "playground/step3_data/barN_zs1.0.txt")
    p.add_argument("--m-floor", type=float, default=1.0e7)
    p.add_argument("--factors", type=float, nargs="+",
                   default=[1e-5, 3.162e-5, 1e-4, 3.162e-4, 1e-3, 3.162e-3, 1e-2, 1e-1])
    p.add_argument("--nM", type=int, default=13)     # coarse host-eval grid in log M
    p.add_argument("--nz", type=int, default=8)      # coarse host-eval grid in z_l
    p.add_argument("--logM-lo", type=float, default=8.5)
    p.add_argument("--logM-hi", type=float, default=15.3)
    p.add_argument("--n-mass", type=int, default=120)
    p.add_argument("--n-y", type=int, default=220)
    p.add_argument("--n-d", type=int, default=280)
    p.add_argument("--n-theta", type=int, default=384)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    zs, kthr, zl_f, M_f, barN_f = load_barN(args.barN)
    out = args.out or ROOT / f"playground/subhalo_factor_population_zs{zs:g}.json"
    print(f"zs={zs:g}  kappathrH={kthr:.4e}  total barN={barN_f.sum():.1f}  bins={len(barN_f)}")

    logM_c = np.linspace(args.logM_lo, args.logM_hi, args.nM)
    zl_c = np.linspace(0.04, zs * 0.97, args.nz)

    e0 = np.full((args.nM, args.nz), np.nan)          # brute per-host excess
    ratio = {f: np.full((args.nM, args.nz), np.nan) for f in args.factors}

    t0 = time.time()
    for i, lM in enumerate(logM_c):
        Mh = 10.0**lM
        for j, zl in enumerate(zl_c):
            if args.m_floor / Mh >= 1.0:
                e0[i, j] = 0.0
                for f in args.factors:
                    ratio[f][i, j] = 1.0
                continue
            _, rows = run(Mh, zl, zs, kthr, args.m_floor, args.factors,
                          n_mass=args.n_mass, n_y=args.n_y, n_d=args.n_d,
                          n_theta=args.n_theta, pooled=False)
            eb = rows[0]["E_brute"]
            e0[i, j] = eb
            for r in rows:
                ratio[r["subhalo_factor"]][i, j] = r["E_proxy"] / eb if eb != 0 else 1.0
        print(f"  logM={lM:.2f} done  [{time.time()-t0:.0f}s]", flush=True)

    # interpolate log(e0) and ratio onto the fine barN grid (regular coarse grid)
    e0_safe = np.clip(e0, 1e-30, None)
    interp_e0 = RegularGridInterpolator((logM_c, zl_c), np.log(e0_safe),
                                        bounds_error=False, fill_value=None)
    logM_q = np.clip(np.log10(M_f), logM_c[0], logM_c[-1])
    zl_q = np.clip(zl_f, zl_c[0], zl_c[-1])
    pts = np.column_stack([logM_q, zl_q])
    e0_fine = np.exp(interp_e0(pts))
    # zero out hosts that cannot host clumps
    e0_fine = np.where(args.m_floor / M_f >= 1.0, 0.0, e0_fine)

    w = barN_f * e0_fine
    E_brute = float(w.sum())
    print(f"\nE_prod(brute) = {E_brute:.4e}")
    print(f"{'factor':>10} {'kappa_thr_clump':>16} {'ratio_pop':>10} {'bias %':>8}")
    results = {"zs": zs, "kappa_thr_host": kthr, "E_prod_brute": E_brute,
               "total_barN": float(barN_f.sum()), "rows": []}
    for f in args.factors:
        ri = RegularGridInterpolator((logM_c, zl_c), ratio[f],
                                     bounds_error=False, fill_value=None)
        r_fine = ri(pts)
        r_fine = np.where(args.m_floor / M_f >= 1.0, 1.0, r_fine)
        rp = float(np.sum(w * r_fine) / E_brute)
        print(f"{f:10.3e} {f*kthr:16.3e} {rp:10.4f} {100*(rp-1):8.2f}")
        results["rows"].append({"subhalo_factor": f, "kappa_thr_clump": f*kthr,
                                "ratio_pop": rp, "bias_pct": 100*(rp-1)})

    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
