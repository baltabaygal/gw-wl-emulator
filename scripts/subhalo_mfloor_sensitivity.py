"""
Test whether subhalo variance convergence is controlled by the low-mass floor.

For each floor, set both:
  - C.Mmin, the cosmology/NFW grid minimum
  - m_floor, the brute-force subhalo minimum

Then measure the paired substructure variance excess:
  Var(kappa) - Var(kappa_nosub)

If the excess grows as the floor is lowered, the apparent low-factor plateau at
1e7 Msun is resolution-floor dependent rather than physically converged.
"""
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT / "build"))
import gwlensing

ZS = 1.0
N = 1_000
SEED = 42
NBOOT = 80
FLOORS = np.array([1.0e7, 1.0e6, 1.0e5])
SUBHALO_FACTOR = 1.0e-8
OUT_DATA = ROOT / "data" / "subhalo_mfloor_sensitivity_z1.npz"
OUT_PLOT = ROOT / "plots" / "figures" / "subhalo_mfloor_sensitivity.png"


def excess_and_err(kappa: np.ndarray, kappa_nosub: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    excess = float(kappa.var() - kappa_nosub.var())
    idx = rng.integers(0, len(kappa), size=(NBOOT, len(kappa)))
    boot = kappa[idx].var(axis=1) - kappa_nosub[idx].var(axis=1)
    return excess, float(boot.std())


def run_floor(floor: float) -> tuple[float, float, float, float]:
    t0 = time.time()
    res = gwlensing.sample_lensing_raw_ml(
        z=ZS,
        h=0.674,
        OmegaM=0.315,
        sigma8=0.811,
        nsamples=N,
        seed=SEED,
        filaments=False,
        bias=False,
        ell=False,
        Nhalos=100,
        subhalo=True,
        m_floor=float(floor),
        subhalo_threads=4,
        subhalo_parallel_threshold=1000,
        subhalo_model=1,
        subhalo_brute=False,
        subhalo_factor=SUBHALO_FACTOR,
        custom_kappathr=-1.0,
        Mmin=float(floor),
    )
    kappa = np.asarray(res["kappa"], dtype=float)
    kappa_nosub = np.asarray(res["kappa_nosub"], dtype=float)
    if not np.isfinite(kappa).all() or not np.isfinite(kappa_nosub).all():
        raise RuntimeError(f"non-finite raw sample for floor={floor:g}")
    excess, err = excess_and_err(kappa, kappa_nosub, np.random.default_rng(123))
    return floor, excess, err, time.time() - t0


def main() -> None:
    rows = []
    for floor in FLOORS:
        row = run_floor(float(floor))
        rows.append(row)
        print(
            f"floor={row[0]:.3g}  excess={row[1]:.4e}  err={row[2]:.2e}  time={row[3]:.1f}s",
            flush=True,
        )

    rows = np.array(rows)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DATA,
        floors=rows[:, 0],
        excess=rows[:, 1],
        err=rows[:, 2],
        runtime=rows[:, 3],
        zs=ZS,
        N=N,
        seed=SEED,
        subhalo_brute=False,
        subhalo_factor=SUBHALO_FACTOR,
    )

    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=rows[:, 2], marker="o", lw=2.3, capsize=3)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("minimum subhalo mass floor [Msun]")
    ax.set_ylabel("Var(kappa) - Var(kappa_nosub)")
    ax.set_title(
        f"Subhalo variance sensitivity to low-mass floor, z_s={ZS:g}, "
        f"N={N:,}, factor={SUBHALO_FACTOR:g}"
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=220, facecolor="white")
    print(f"saved {OUT_DATA}")
    print(f"saved {OUT_PLOT}")


if __name__ == "__main__":
    main()
