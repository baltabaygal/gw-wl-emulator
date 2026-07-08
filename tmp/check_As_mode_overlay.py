"""Overlay check for the A_s normalization mode (2026-07-07).

Same seed, zs=1: sigma8-mode (0.811) vs As-mode at the matched amplitude
(As_derived => identical deltaH8 => identical draws), plus As-mode at the
Planck 2018 A_s (2.101e-9) whose derived sigma8 is 0.860 with the code's
smooth-k window (visibly higher amplitude, as expected).
-> plots/As_mode_overlay.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import gwlensing as gw

Z, N, SEED = 1.0, 400_000, 20260707
H, OM, S8 = 0.674, 0.315, 0.811

cfg = gw.get_simulator_config(h=H, OmegaM=OM, sigma8=S8)
As_matched = cfg["As_derived"]

runs = {
    f"sigma8-mode  (s8={S8})": dict(),
    f"As-mode matched (As={As_matched:.3e})": dict(As=As_matched),
    "As-mode Planck (As=2.101e-9, s8_derived=0.860)": dict(As=2.101e-9, ns=0.9649),
}
edges = np.geomspace(0.5, 30.0, 80)
ctr = np.sqrt(edges[:-1] * edges[1:]); bw = np.diff(edges)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
hists = {}
for label, kw in runs.items():
    r = gw.sample_lnmu_ml_with_diagnostics(Z, H, OM, S8, N, SEED, False, **kw)
    x = np.asarray(r["lnmu"], float); x = x[np.isfinite(x)]
    c, _ = np.histogram(np.exp(x), bins=edges)
    hists[label] = (c, x.size)
    a1.plot(ctr, c / (hists[label][1] * bw), lw=1.2, label=label)

k = list(hists)
c0, n0 = hists[k[0]]; c1, n1 = hists[k[1]]
print("sigma8-mode vs matched As-mode: identical histograms =", np.array_equal(c0, c1))
a2.plot(ctr, (c1 / n1) / np.maximum(c0 / n0, 1e-300), lw=1.2, label="matched As / sigma8")
c2, n2 = hists[k[2]]
a2.plot(ctr, (c2 / n2) / np.maximum(c0 / n0, 1e-300), lw=1.2, label="Planck As / sigma8")

a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel(r"$\mu$"); a1.set_ylabel(r"$dP/d\mu$")
a1.legend(fontsize=7); a1.grid(True, which="both", ls=":", alpha=.3)
a2.set_xscale("log"); a2.axhline(1.0, color="k", ls="--", lw=0.8)
a2.set_xlabel(r"$\mu$"); a2.set_ylabel("ratio to sigma8-mode")
a2.set_ylim(0.5, 2.0); a2.legend(fontsize=8); a2.grid(True, which="both", ls=":", alpha=.3)
fig.suptitle(f"A_s-mode vs sigma8-mode, zs={Z}, same seed ({N//1000}k samples)")
fig.tight_layout()
out = ROOT / "plots" / "As_mode_overlay.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
