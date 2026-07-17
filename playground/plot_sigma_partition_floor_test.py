#!/usr/bin/env python3
"""Floor-constraint validation figure: is the below-kappa_min rise ONLY the floor?

Compares the measured clipped (kappa_tot <= 1) sigma_total vs custom_kappathr for
two absolute background floors:
  production: kappa_min = 1.28e-7 (1e-3 * kappa_thr(N=100))    -> rises below kappa_min
  lowered:    kappa_min = 1e-11   (kappathr_flat = 1e-8 lever) -> must be flat on 1e-8..1e3

If the lowered-floor run is flat across the entire sweep (at its own, floor-
dependent plateau), the production run's below-floor rise is purely the domain
constraint (halos in (kappa_thr, kappa_min) exist in neither arm's budget), not
a defect of the weak/strong split machinery.

Inputs: data/sigma_partition_vs_kthr_bias_z1.npz (production floor, consolidated)
        playground/partition_components_zs1_floortest_seed*.npz (lowered floor)
Writes plots/sigma_partition_floor_test.{png,pdf}. Run from the repo root.
"""
import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

KMIN_PROD = 1.278e-7

d = np.load("data/sigma_partition_vs_kthr_bias_z1.npz")
kt_p = d["kthr"]
st_p = np.sqrt(d["var_t_clip1"])
sem_p = d["var_t_clip1_sem"] / (2 * st_p)

files = sorted(glob.glob("playground/partition_components_zs1_floortest_seed*.npz"))
per = [np.load(f) for f in files]
kt_f = per[0]["kthr"]
arr = np.stack([p["clip1"] for p in per])          # (nseed, nkt, 7)
vt = arr[:, :, 3]
st_f = np.sqrt(vt.mean(0))
sem_f = (vt.std(0, ddof=1) / np.sqrt(arr.shape[0])) / (2 * st_f)

core_p = kt_p >= KMIN_PROD
pl_p = st_p[core_p].mean()
pl_f = st_f.mean()
flat_f = np.ptp(st_f) / pl_f

INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.edgecolor": "#c3c2b7", "font.size": 10.5})
fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.axvline(KMIN_PROD, color=MUTED, ls=":", lw=1.0)
ax.text(KMIN_PROD * 1.4, 0.0345, r"$\kappa_{\rm min}^{\rm prod} = 1.28\times10^{-7}$",
        fontsize=8.5, color=MUTED)
ax.axhline(pl_p, color="#0f4c81", ls=":", lw=0.9, alpha=0.6)
ax.axhline(pl_f, color="#b45309", ls=":", lw=0.9, alpha=0.6)

ax.errorbar(kt_p, st_p, yerr=sem_p, fmt="o-", ms=3.4, lw=1.3, capsize=1.5,
            color="#0f4c81", label=r"production floor ($\kappa_{\rm min}=1.28\times10^{-7}$)")
ax.errorbar(kt_f, st_f, yerr=sem_f, fmt="s-", ms=3.4, lw=1.3, capsize=1.5,
            color="#b45309", label=r"lowered floor ($\kappa_{\rm min}=10^{-11}$)")

ax.set_xscale("log")
ax.set_xlabel(r"$\kappa_{\rm threshold}$")
ax.set_ylabel(r"$\sigma_\kappa$ (total, $\kappa_{\rm tot}\leq 1$ core)")
ax.set_title("total variance vs split threshold: the rise is the floor, not the split",
             fontsize=10.5)
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.grid(color="#e1e0d9", lw=.6, which="both")
ax.set_axisbelow(True)

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"plots/sigma_partition_floor_test.{ext}", dpi=200)

print(f"production floor: plateau (kt>=kappa_min) = {pl_p:.6f}, "
      f"below-floor max = {st_p[~core_p].max():.6f} ({100*(st_p[~core_p].max()/pl_p-1):+.1f}%)")
print(f"lowered floor:    plateau = {pl_f:.6f}  flatness ptp = {100*flat_f:.2f}%  "
      f"({arr.shape[0]} seeds)")
print("wrote plots/sigma_partition_floor_test.{png,pdf}")
