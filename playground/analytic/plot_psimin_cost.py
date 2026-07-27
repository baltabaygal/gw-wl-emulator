"""
Real (measured, not analytic) computational cost vs psi_min for the resolved-only
scheme (subhalo_model=1, subhalo_brute=True, psi_min_fixed swept -- the override added
2026-07-23), normalized to the no-subhalo baseline = 1.0. Data from
bench_psimin_cost_sweep.py (z_s=1.0, Nreal=500, seed=7).

Run:
  /Users/baltabay/miniforge3/envs/test/bin/python playground/analytic/plot_psimin_cost.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = json.load(open(
    '/private/tmp/claude-501/-Users-baltabay-Desktop-gw-wl-emulator/d04000ea-78eb-40d6-8254-292a7b707e7c/scratchpad/psimin_cost_sweep.json'))

psi_min = np.array([r["psi_min"] for r in DATA["rows"]])
ratio = np.array([r["ratio"] for r in DATA["rows"]])

fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
ax.plot(psi_min, ratio, "o-", color="#2563eb", ms=6, lw=2.2,
        label="resolved-only (subhalo_model=1, subhalo_brute)")
ax.axhline(1.0, color="#047857", ls="--", lw=1.6, label="no-subhalo baseline = 1.0")

ax.set_xscale("log")
ax.set_xlabel(r"$\psi_{\rm min}$  (fixed resolved floor, same for every host mass)")
ax.set_ylabel("wall-clock time  /  no-subhalo baseline")
ax.set_title(f"measured cost vs $\\psi_{{\\rm min}}$  ($z_s$={DATA['zs']:g}, "
             f"Nreal={DATA['Nreal']}, resolved clumps only)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="upper right")

for x, y in zip(psi_min, ratio):
    ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha="center", color="#1e3a5f")

fig.tight_layout()
out = Path("/Users/baltabay/Desktop/gw-wl-emulator/playground/analytic/plot_psimin_cost.png")
fig.savefig(out, dpi=180, facecolor="white", bbox_inches="tight")
print(f"saved {out}")
