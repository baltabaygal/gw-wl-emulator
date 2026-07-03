"""
Convergence of Var(kappa) with subhalo_factor (the clump-resolution gate).

subhalo_factor rescales the clump threshold: kappa_thr,clump = factor * kappa_thr,host.
Lowering it lowers the resolved-mass floor; at factor -> 0 the floor bottoms out at
C.Mmin = m_floor and the dynamic split becomes exactly brute force. The variance excess
must therefore plateau at the brute value; we find the largest factor already on the
plateau and fix the default there.

Noise control (same trick as plot_variance_vs_factor.py): within one run, kappa and
kappa_nosub share all RNG draws, and kappa_nosub tracks the full unperturbed host, so
  excess(f) = var(kappa) - var(kappa_nosub)
is the substructure effect with host/weak sampling noise cancelled. Errors: bootstrap
over realizations, resampling (kappa, kappa_nosub) jointly.

Run AFTER the 2026-07-02 fixes (corrected w_f, double-angle shear, conservative floor).
Writes data/subhalo_factor_convergence_z{zs}.npz + plots/figures/subhalo_factor_convergence.png
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool

ROOT = "/Users/baltabay/Desktop/gw-wl-emulator"
sys.path.insert(0, os.path.join(ROOT, "build"))
import gwlensing

ZS = 1.0
N = 40_000
SEED = 42
NBOOT = 200
FACTORS = np.logspace(-4, 2, 13)          # 1e-4 ... 1e2, includes 1.0

COMMON = dict(z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
              nsamples=N, seed=SEED, filaments=False, bias=False, ell=False,
              m_floor=1e7, subhalo_model=1)

def excess_and_err(k, k_ns, rng):
    ex = k.var() - k_ns.var()
    idx = rng.integers(0, len(k), size=(NBOOT, len(k)))
    boot = k[idx].var(axis=1) - k_ns[idx].var(axis=1)
    return ex, boot.std()

def run_one(args):
    label, kw = args
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(**COMMON, **kw)
    dt = time.time() - t0
    k = np.asarray(res["kappa"], dtype=float)
    k_ns = np.asarray(res["kappa_nosub"], dtype=float)
    ex, err = excess_and_err(k, k_ns, np.random.default_rng(0))
    return label, ex, err, dt

if __name__ == "__main__":
    # expensive jobs first (low factor ~ brute cost), chunksize=1 so no worker gets
    # two expensive jobs serialized; imap_unordered prints progress as results land
    jobs = [("brute", dict(subhalo=True, subhalo_brute=True))]
    jobs += [(f"{f:.4g}", dict(subhalo=True, subhalo_factor=f)) for f in np.sort(FACTORS)]

    results = []
    with Pool(processes=6) as pool:
        for out in pool.imap_unordered(run_one, jobs, chunksize=1):
            results.append(out)
            print(f"  done: factor={out[0]:>6}  excess={out[1]:.4e}  [{out[3]:.0f}s]",
                  flush=True)

    res = {label: (ex, err, dt) for label, ex, err, dt in results}
    ex_b, err_b, dt_b = res["brute"]
    print(f"\nzs={ZS}, N={N}; substructure variance excess (var(k) - var(k_nosub)):")
    print(f"{'factor':>10}  {'excess':>12}  {'err':>10}  {'vs brute':>10}  {'time':>7}")
    print(f"{'brute':>10}  {ex_b:12.4e}  {err_b:10.2e}  {'--':>10}  {dt_b:6.1f}s")
    rows = []
    for f in FACTORS:
        ex, err, dt = res[f"{f:.4g}"]
        dev = ex - ex_b
        rows.append((f, ex, err, dt))
        print(f"{f:10.4g}  {ex:12.4e}  {err:10.2e}  {100*dev/ex_b:+9.2f}%  {dt:6.1f}s")

    rows = np.array(rows)
    # convergence: largest factor whose deviation from brute is within
    # max(2 sigma of the difference, 2% of the brute excess)
    dev = np.abs(rows[:, 1] - ex_b)
    tol = np.maximum(2*np.hypot(rows[:, 2], err_b), 0.02*abs(ex_b))
    conv = rows[:, 0][dev <= tol]
    f_rec = conv.max() if len(conv) else rows[0, 0]
    print(f"\nconverged (within max(2sigma, 2% of brute excess)) up to factor = {f_rec:.4g}")

    out = os.path.join(ROOT, f"data/subhalo_factor_convergence_z{ZS:g}.npz")
    np.savez(out, factors=rows[:, 0], excess=rows[:, 1], err=rows[:, 2],
             runtime=rows[:, 3], excess_brute=ex_b, err_brute=err_b,
             runtime_brute=dt_b, f_recommended=f_rec, zs=ZS, N=N)
    print(f"saved {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(ex_b - 2*err_b, ex_b + 2*err_b, color="0.85", zorder=0,
               label=f"brute force (m_floor=1e7) ± 2σ  [{dt_b:.0f} s]")
    ax.axhline(ex_b, color="0.4", ls="--", lw=1.5, zorder=1)
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], marker="o", ms=6,
                lw=2, capsize=3, color="#4f46e5", zorder=3, label="dynamic split")
    ax.axvline(f_rec, color="#dc2626", ls=":", lw=2,
               label=f"recommended factor = {f_rec:.3g}")
    ax.set_xscale("log")
    ax.set_xlabel("subhalo_factor  (clump κ_thr / host κ_thr)")
    ax.set_ylabel("substructure variance excess  var(κ) − var(κ_nosub)")
    ax.set_title(f"Convergence of the substructure κ-variance, z_s={ZS}, N={N}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    png = os.path.join(ROOT, "plots/figures/subhalo_factor_convergence.png")
    fig.savefig(png, dpi=200)
    print(f"saved {png}")
