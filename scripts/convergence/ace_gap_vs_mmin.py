"""ACE - Vaskonen convergence-moment gap as a function of Mmin.

Extends scripts/subhalo_gate/ace_gap.py (single point at the default
Mmin=1e7) to the Mmin convergence-study grid. ACE-Lensing is built from
N-body sims with NO halo mass floor (all particles projected; particle mass
4.1*Om*1e9 Msun/h, Table 1 + Sec 2.2 of Turker+2025, verified against the
PDF 2026-07-11), so its moments are Mmin-independent by construction and the
sweep asks: does the halo model's missing sub-Mmin variance move the
Vaskonen moments toward ACE, and by how much of the 50-86% gap?

Vaskonen arm: full model ON (filaments/bias/ell), default fixed-<N>=100
rule, raw kappa central moments; 4 disjoint seeds per config for error bars
(<k^3> is tail-noisy: +-20% per 5e4 draws, so NEVER single-seed).

Run with the test env:
  /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/ace_gap_vs_mmin.py
"""
import sys, json, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build"))
sys.path.insert(0, str(REPO / "ace_lensing"))

H, OM, S8 = 0.674, 0.315, 0.811
ZS_LIST = [1.0, 5.0]
MMIN_GRID = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
NSEEDS = 4
N = 400_000
SEED0 = 700_000_000          # disjoint from the convergence_scan namespaces


def worker(task):
    z, mmin, iseed, klo, khi = task
    import gwlensing as gw
    seed = SEED0 + 1_000_000 * ZS_LIST.index(z) + 10_000 * MMIN_GRID.index(mmin) + iseed
    t0 = time.perf_counter()
    raw = gw.sample_lensing_raw_ml(z, H, OM, S8, N, seed=seed, Mmin=mmin)
    k = np.asarray(raw["kappa"])
    k = k - k.mean()
    # clipped variant: restrict to ACE's own kappa support before central
    # moments — raw moments are dominated by single kappa>>1 rays at low Mmin
    # (one kappa~500 ray shifts <k^2> by 0.15 at N=4e5), while the ACE PDF has
    # finite support, so the raw comparison is apples-to-oranges in the tail.
    kc = k[(k >= klo) & (k <= khi)]
    kc = kc - kc.mean()
    return (z, mmin, iseed, float((k**2).mean()), float((k**3).mean()),
            float((kc**2).mean()), float((kc**3).mean()),
            time.perf_counter() - t0)


def ace_moments(z):
    from ace_lensing import predict_pdf
    mu, pdf = predict_pdf(Om=OM, h=H, w=-1.0, s8=S8, z=z, verbose=False)
    mu = np.asarray(mu, float)
    pdf = np.clip(np.asarray(pdf, float), 0, None)
    ok = np.isfinite(mu) & np.isfinite(pdf)
    mu, pdf = mu[ok], pdf[ok]
    kap = (mu - 1.0) / 2.0            # weak-regime kappa = (mu-1)/2
    w = pdf / np.trapezoid(pdf, kap)
    m = np.trapezoid(w * kap, kap)
    K2 = np.trapezoid(w * (kap - m) ** 2, kap)
    K3 = np.trapezoid(w * (kap - m) ** 3, kap)
    return K2, K3, float(kap.min()), float(kap.max())


def main():
    ace = {z: ace_moments(z) for z in ZS_LIST}    # (K2, K3, kap_lo, kap_hi)
    tasks = [(z, m, i, ace[z][2], ace[z][3])
             for z in ZS_LIST for m in MMIN_GRID for i in range(NSEEDS)]
    print(f"[ace_gap_vs_mmin] {len(tasks)} tasks (N={N:,} each) over 8 procs",
          flush=True)
    for z in ZS_LIST:
        print(f"  ACE z={z:g}: kappa support [{ace[z][2]:.3f}, {ace[z][3]:.3f}]",
              flush=True)
    K2 = {}; K3 = {}; K2c = {}; K3c = {}
    t0 = time.perf_counter()
    with mp.Pool(8) as pool:
        for z, m, i, k2, k3, k2c, k3c, dt in pool.imap_unordered(worker, tasks):
            K2.setdefault((z, m), [None] * NSEEDS)[i] = k2
            K3.setdefault((z, m), [None] * NSEEDS)[i] = k3
            K2c.setdefault((z, m), [None] * NSEEDS)[i] = k2c
            K3c.setdefault((z, m), [None] * NSEEDS)[i] = k3c
            print(f"  z={z:g} Mmin={m:.0e} seed{i}: <k2>c={k2c:.4e} "
                  f"<k3>c={k3c:.4e} ({dt:.0f}s, wall {time.perf_counter()-t0:.0f}s)",
                  flush=True)

    out = dict(mmin_grid=MMIN_GRID, zs=ZS_LIST, nseeds=NSEEDS, n=N)
    for z in ZS_LIST:
        k2a, k3a, klo, khi = ace[z]
        out[f"ace_K2_z{z:g}"] = k2a
        out[f"ace_K3_z{z:g}"] = k3a
        out[f"ace_ksupport_z{z:g}"] = [klo, khi]
        out[f"K2_z{z:g}"] = [K2[(z, m)] for m in MMIN_GRID]
        out[f"K3_z{z:g}"] = [K3[(z, m)] for m in MMIN_GRID]
        out[f"K2clip_z{z:g}"] = [K2c[(z, m)] for m in MMIN_GRID]
        out[f"K3clip_z{z:g}"] = [K3c[(z, m)] for m in MMIN_GRID]
        print(f"\nz_s={z:g}:  ACE <k2>={k2a:.4e}  <k3>={k3a:.4e}  "
              f"support [{klo:.3f},{khi:.3f}]")
        print(f"{'Mmin':>8s} {'<k2>clip mean+-sd':>24s} {'gap':>7s} "
              f"{'<k3>clip mean+-sd':>24s} {'gap':>7s}")
        for m in MMIN_GRID:
            a2 = np.array(K2c[(z, m)]); a3 = np.array(K3c[(z, m)])
            g2 = abs(k2a - a2.mean()) / a2.mean()
            g3 = abs(k3a - a3.mean()) / a3.mean()
            print(f"{m:8.0e} {a2.mean():.4e} +- {a2.std():.1e} {g2:6.0%} "
                  f"{a3.mean():.4e} +- {a3.std():.1e} {g3:6.0%}")

    outdir = REPO / "data" / "results" / "mmin_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "ace_gap_vs_mmin.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n-> {outdir}/ace_gap_vs_mmin.json  "
          f"(total {time.perf_counter()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
