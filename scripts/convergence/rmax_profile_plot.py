"""rmax vs z_l per z_s — companion to mb_profile_plot.py.

rmax(M, z_l; kappa_thr(z_s)) is the tube/encounter-disc radius (PHYSICAL kpc,
the code's internal unit) inside which a lens of mass M at z_l produces
kappa > kappa_thr. Per z_l bin it spans decades across the mass cells, so per
z_s we plot the barN-weighted (per-encounter) geometric mean with a q10-q90
band, all masses collapsed — same aggregation as mb_profile_plot.py.

Run: /Users/baltabay/miniforge3/envs/test/bin/python scripts/convergence/rmax_profile_plot.py
Output: plots/rmax_profile.png
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "convergence"))
from bias_field_prototype import Cosmo  # noqa: E402

ZS_LIST = [0.2, 1.0, 5.0, 10.0]
NHALOS = 100
COLORS = ["#4269d0", "#e05c1f", "#3ca951", "#9c6bd0"]


def wquant(x, w, qs):
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w)
    cw /= cw[-1]
    return np.interp(qs, cw, x)


def profile(C, zs):
    kt = C.find_kappathr(zs, NHALOS)
    T = C.cell_tables(zs, kt)
    rows = []
    for jz in np.unique(T["jz"]):
        s = T["jz"] == jz
        w, r = T["barN"][s], T["rmax"][s]
        if w.sum() <= 0:
            continue
        gm = 10 ** (np.sum(w * np.log10(r)) / w.sum())
        lo, hi = 10 ** wquant(np.log10(r), w, [0.10, 0.90])
        rows.append((C.zlist[jz], gm, lo, hi))
    return kt, np.array(rows)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    C = Cosmo()
    for zs, col in zip(ZS_LIST, COLORS):
        kt, P = profile(C, zs)
        zl, gm, lo, hi = P.T
        ax.fill_between(zl, lo, hi, color=col, alpha=0.12, lw=0, step="mid")
        ax.plot(zl, gm, color=col, lw=1.4, marker="o", ms=3,
                label=f"$z_s={zs:g}$  ($\\kappa_{{\\rm thr}}$={kt:.1e})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"lens redshift bin $z_l$")
    ax.set_ylabel(r"$r_{\rm max}$  [physical kpc]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.5, loc="lower center")
    ax.set_title("Encounter-disc radius $r_{\\rm max}$ per $z_l$ bin "
                 "(default grid, fixed-$\\langle N\\rangle$=100, halo only)\n"
                 "line: per-encounter geometric mean over mass cells; "
                 "band: q10–q90", fontsize=11)
    out = REPO / "plots" / "rmax_profile.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
