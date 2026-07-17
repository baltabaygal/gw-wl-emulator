#!/usr/bin/env python3
"""PDF-level effect of the explicit/weak split: JSD + shape vs <N>.

Data: playground/pdf_vs_N_zs1_seed{0..7}.npz (sweep_pdf_vs_N.py) — 240k rays
per arm, arms <N> = 300..0.1, production floor kappa_min pinned for all arms.
Reference = the <N>=300 arm. JSD in nats on an 80-bin lnmu histogram spanning
the ref [0.02%, 99.98%] quantiles + under/overflow cells; null floor = JSD
between disjoint 4-seed halves of the ref, scaled x0.5 to 240k (1/n bias law).
Also reports trimmed sigma/skew/kurtosis and tail quantiles per arm.
Output: plots/pdf_vs_N_effect.png
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper_prod.plot_style import apply_style  # noqa: E402

apply_style()
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "cm"

SEEDS = range(8)
d0 = np.load(ROOT / "playground" / "pdf_vs_N_zs1_seed0.npz")
NT = d0["N_targets"]
KT = d0["kthr"]
narm = len(NT)

arms = []          # arm -> list of per-seed lnmu arrays
for i in range(narm):
    arms.append([np.load(ROOT / "playground" / f"pdf_vs_N_zs1_seed{s}.npz")[f"lnmu_{i}"]
                 for s in SEEDS])
pool = [np.concatenate(a) for a in arms]
ref = pool[0]

edges = np.quantile(ref, np.linspace(2e-4, 1 - 2e-4, 81))


def jsd(x, y):
    p = np.histogram(x, np.r_[-np.inf, edges, np.inf])[0] / len(x)
    q = np.histogram(y, np.r_[-np.inf, edges, np.inf])[0] / len(y)
    m_ = 0.5 * (p + q)
    t = np.zeros_like(m_)
    for a in (p, q):
        nz = a > 0
        t[nz] += 0.5 * a[nz] * np.log(a[nz] / m_[nz])
    return t.sum()


half = lambda a: (np.concatenate(a[:4]), np.concatenate(a[4:]))
floor_ref = jsd(*half(arms[0])) * 0.5          # scale 120k-half null to 240k
J = np.array([jsd(pool[i], ref) for i in range(narm)])
Jfloor = np.array([jsd(*half(arms[i])) * 0.5 for i in range(narm)])

lo, hi = np.quantile(ref, [1e-3, 1 - 1e-3])
print(f"floor (ref halves, scaled): {floor_ref:.2e} nats;  "
      f"trim window lnmu = [{lo:.3f}, {hi:.3f}]")
print(f"{'<N>':>7} {'kthr':>10} {'JSD':>10} {'floor':>9} {'sig':>8} "
      f"{'skew':>7} {'exkurt':>7} {'q99.9(lnmu)':>12}")
stats = []
for i in range(narm):
    x = pool[i]
    c = x[(x >= lo) & (x <= hi)]
    sk = float(((c - c.mean()) ** 3).mean() / c.std() ** 3)
    ku = float(((c - c.mean()) ** 4).mean() / c.std() ** 4 - 3.0)
    q999 = float(np.quantile(x, 0.999))
    stats.append((c.std(), sk, ku, q999))
    print(f"{NT[i]:7.1f} {KT[i]:10.3e} {J[i]:10.3e} {Jfloor[i]:9.2e} "
          f"{c.std():8.5f} {sk:7.3f} {ku:7.3f} {q999:12.4f}")

C_REF, C_ARM, C_FIT = "#0f4c81", ["#5b8fbe", "#d97706", "#b5443c"], "#888888"
SHOW = [5, 6, 7]                               # <N> = 1, 0.3, 0.1 vs ref

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
fig.subplots_adjust(left=0.09, right=0.97, bottom=0.16, top=0.84, wspace=0.28)

# ---- left: PDF overlay, log-y
ax = axes[0]
be = np.linspace(np.quantile(ref, 5e-5), np.quantile(ref, 1 - 5e-5), 120)
bc = 0.5 * (be[1:] + be[:-1])
h = np.histogram(ref, be, density=True)[0]
ax.fill_between(bc, h, color=C_REF, alpha=0.25, lw=0)
ax.plot(bc, h, color=C_REF, lw=1.2, label=r"$\langle N\rangle = 300$ (ref)")
for c, i in zip(C_ARM, SHOW):
    h = np.histogram(pool[i], be, density=True)[0]
    ax.plot(bc, h, color=c, lw=1.1, label=rf"$\langle N\rangle = {NT[i]:g}$")
ax.set_yscale("log")
ax.set_ylim(3e-4, 30)
ax.set_xlabel(r"$\ln\mu$")
ax.set_ylabel(r"$P(\ln\mu)$")
ax.legend(fontsize=6, loc="upper right", handlelength=1.8)
ax.set_title("PDF (240k rays/arm)", fontsize=8)

# ---- right: JSD vs <N>
ax = axes[1]
ax.plot(NT[1:], J[1:], "o-", ms=3.5, lw=1.2, color="#b5443c",
        label=r"JSD vs $\langle N\rangle=300$ ref")
ax.plot(NT[1:], Jfloor[1:], "s", ms=2.8, color="#888888", alpha=0.8,
        label="per-arm null (seed halves)")
ax.axhline(floor_ref, color="#888888", lw=0.9, ls="--",
           label="ref null floor")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\langle N\rangle$ explicit halos per ray")
ax.set_ylabel("JSD [nats]")
ax.invert_xaxis()
ax.legend(fontsize=6, loc="upper left", handlelength=1.8)
ax.set_title(r"distance to the $\langle N\rangle=300$ reference", fontsize=8)

fig.suptitle(r"PDF invariance under the explicit/weak split "
             r"($z_s=1$, floor $\kappa_{\min}$ pinned, bias\_weak on)"
             if mpl.rcParams["text.usetex"] else
             r"PDF invariance under the explicit/weak split "
             r"($z_s=1$, floor $\kappa_{\rm min}$ pinned, bias_weak on)",
             fontsize=9, y=0.97)

out = ROOT / "plots" / "pdf_vs_N_effect.png"
fig.savefig(out, dpi=300, facecolor="white")
fig.savefig(out.with_suffix(".pdf"), facecolor="white")
print(f"Wrote {out}")
