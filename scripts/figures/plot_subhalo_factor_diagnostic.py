"""
Two-panel diagnostic for the subhalo_factor resolution split.

Panel 1 shows the intuitive total variance story:
large subhalo_factor -> subhalo=off, small subhalo_factor -> brute-force plateau.

Panel 2 shows the lower-noise paired excess used for the actual convergence decision:
Var(kappa) - Var(kappa_nosub), compared with brute force.

Inputs are cached by the heavier sweep scripts:
  data/variance_sweep_data_z1.npz
  data/subhalo_factor_convergence_z1.npz
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/baltabay/Desktop/gw-wl-emulator")
sys.path.insert(0, str(ROOT / "build"))
import gwlensing

TOTAL_DATA = ROOT / "data" / "variance_sweep_data_z1.npz"
EXCESS_DATA = ROOT / "data" / "subhalo_factor_convergence_z1.npz"
EXTENDED_TOTAL_DATA = ROOT / "data" / "variance_sweep_data_z1_extended_high_factor.npz"
OUT = ROOT / "plots" / "figures" / "subhalo_factor_variance_diagnostic.png"

HIGH_FACTORS = np.array([1.0e3, 1.0e4, 1.0e5, 1.0e6])
N_EXTENDED = 50_000
SEED = 42


def run_total_variance(factor: float) -> float:
    res = gwlensing.sample_lensing_raw_ml(
        z=1.0,
        h=0.674,
        OmegaM=0.315,
        sigma8=0.811,
        nsamples=N_EXTENDED,
        seed=SEED,
        filaments=False,
        bias=False,
        ell=False,
        subhalo=True,
        subhalo_factor=float(factor),
        m_floor=1.0e7,
        subhalo_model=1,
    )
    k = np.asarray(res["kappa"], dtype=float)
    k_ns = np.asarray(res["kappa_nosub"], dtype=float)
    return float(k.var() - k_ns.var())


def load_or_build_extended_total(var_off: float) -> tuple[np.ndarray, np.ndarray]:
    if EXTENDED_TOTAL_DATA.exists():
        data = np.load(EXTENDED_TOTAL_DATA)
        return data["factors"], data["var_on_paired"]

    excess = []
    for factor in HIGH_FACTORS:
        ex = run_total_variance(factor)
        excess.append(ex)
        print(f"computed high-factor point {factor:.4g}: excess={ex:.4e}", flush=True)

    factors = HIGH_FACTORS.copy()
    var_on_paired = var_off + np.array(excess)
    np.savez(EXTENDED_TOTAL_DATA, factors=factors, var_on_paired=var_on_paired, var_off=var_off)
    return factors, var_on_paired


def main() -> None:
    if not TOTAL_DATA.exists():
        raise FileNotFoundError(f"missing {TOTAL_DATA}; run scripts/plot_variance_vs_factor.py first")
    if not EXCESS_DATA.exists():
        raise FileNotFoundError(f"missing {EXCESS_DATA}; run scripts/subhalo_factor_convergence.py first")

    total = np.load(TOTAL_DATA)
    conv = np.load(EXCESS_DATA)

    factors_total = total["factors"]
    var_total = total["var_on_paired"]
    var_off = float(total["var_off"])
    factors_high, var_high = load_or_build_extended_total(var_off)
    factors_total = np.concatenate([factors_total, factors_high])
    var_total = np.concatenate([var_total, var_high])
    order = np.argsort(factors_total)
    factors_total = factors_total[order]
    var_total = var_total[order]

    factors_excess = conv["factors"]
    excess = conv["excess"]
    err = conv["err"]
    excess_brute = float(conv["excess_brute"])
    err_brute = float(conv["err_brute"])
    f_recommended = float(conv["f_recommended"])
    zs = float(conv["zs"])
    n_conv = int(conv["N"])
    brute_total = var_off + excess_brute

    OUT.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f"Subhalo-factor variance convergence, z_s={zs:g}", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(
        factors_total,
        var_total,
        marker="o",
        ms=6,
        lw=2.4,
        color="#4f46e5",
        label="subhalo on, dynamic split",
    )
    ax.axhline(var_off, color="#334155", ls="--", lw=2.0, label="subhalo off")
    ax.axhline(brute_total, color="#6b7280", ls="-.", lw=1.8, label="brute-force total estimate")
    ax.axvline(f_recommended, color="#dc2626", ls=":", lw=2.0)
    ax.fill_between(
        factors_total,
        var_total,
        var_off,
        where=var_total >= var_off,
        color="#4f46e5",
        alpha=0.10,
        interpolate=True,
    )
    ax.set_ylabel("Var(kappa)")
    ax.set_title("Total convergence variance")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.30, which="both")

    ax = axes[1]
    ax.axhspan(
        excess_brute - 2.0 * err_brute,
        excess_brute + 2.0 * err_brute,
        color="0.86",
        label="brute force +/- 2 sigma",
        zorder=0,
    )
    ax.axhline(excess_brute, color="0.35", ls="--", lw=1.6, zorder=1)
    ax.errorbar(
        factors_excess,
        excess,
        yerr=err,
        marker="o",
        ms=6,
        lw=2.2,
        capsize=3,
        color="#0891b2",
        label=f"paired excess, N={n_conv:,}",
        zorder=3,
    )
    ax.axvline(
        f_recommended,
        color="#dc2626",
        ls=":",
        lw=2.0,
        label=f"recommended factor = {f_recommended:.3g}",
    )
    ax.set_xscale("log")
    ax.set_xlim(factors_total.min(), factors_total.max())
    ax.set_xlabel("subhalo_factor  (clump kappa threshold / host kappa threshold)")
    ax.set_ylabel("Var(kappa) - Var(kappa_nosub)")
    ax.set_title("Paired substructure variance excess")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.30, which="both")

    fig.tight_layout()
    fig.savefig(OUT, dpi=250, facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
