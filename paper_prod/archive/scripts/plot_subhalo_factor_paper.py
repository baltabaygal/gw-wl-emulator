"""
Production copy of scripts/plot_subhalo_factor_paper.py
Defaults changed to load data from paper_prod/data and write outputs into paper_prod/plots/figures.
Also writes metadata JSON into paper_prod/metadata/.
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import json
import datetime
import subprocess
from paper_prod.plot_style import apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST

# apply production style and sizes
SIZES = apply_style()

# ROOT points to repo root
ROOT = Path(__file__).resolve().parents[2]

ADOPT = 1e-5

# Data files moved to paper_prod/data by convention
RC_PATH = ROOT / "paper_prod" / "data" / "subhalo_factor_redshift_check.npz"
CZ_PATH = ROOT / "paper_prod" / "data" / "subhalo_factor_convergence_z1.npz"
TAIL_CSV = ROOT / "paper_prod" / "data" / "subhalo_factor_z1_tail.csv"

OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def git_commit_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"

# --- cross-redshift scan ---
rc = np.load(RC_PATH, allow_pickle=True)
rows = np.asarray(rc["rows"], float)
zs_all = np.asarray(rc["zs_values"], float)
def zs_curve(zs):
    m = rows[:, 0] == zs
    o = np.argsort(rows[m, 1])
    return rows[m, 1][o], rows[m, 4][o], rows[m, 5][o]

# --- zs=1 with brute reference ---
cz = np.load(CZ_PATH, allow_pickle=True)
f1c, e1c, er1c = np.asarray(cz["factors"], float), np.asarray(cz["excess"], float), np.asarray(cz["err"], float)
brutes = [(float(cz["excess_brute"]), float(cz["err_brute"]))]

f1t, e1t, er1t = [], [], []
if TAIL_CSV.exists():
    import csv as _csv
    with open(TAIL_CSV) as fh:
        for row in _csv.DictReader(fh):
            if row["label"] == "brute":
                brutes.append((float(row["excess"]), float(row["err"])))
            else:
                f1t.append(float(row["label"]))
                e1t.append(float(row["excess"]))
                er1t.append(float(row["err"]))

w = np.array([1/s**2 for _, s in brutes])
ex_brute = np.sum(w*np.array([e for e, _ in brutes]))/w.sum()
er_brute = 1/np.sqrt(w.sum())

f1r, e1r, er1r = zs_curve(1.0)
f1 = np.concatenate([f1t, f1r, f1c]); e1 = np.concatenate([e1t, e1r, e1c]); er1 = np.concatenate([er1t, er1r, er1c])
o = np.argsort(f1); f1, e1, er1 = f1[o], e1[o], er1[o]
_, uidx = np.unique(np.round(np.log10(f1), 3), return_index=True)
f1, e1, er1 = f1[uidx], e1[uidx], er1[uidx]

def log_format(x, pos):
    val = np.log10(x)
    if np.isclose(val, np.round(val)):
        exponent = int(np.round(val))
        if exponent == 0:
            return "1"
        return fr"$10^{{{exponent}}}$"
    else:
        return f"{x:g}"

def draw_panel_a(ax, show_label=False):
    zs_show = [0.5, 1.0, 2.0, 5.0, 10.0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.82, len(zs_show)))
    for zs, col in zip(zs_show, cmap):
        f, e, er = zs_curve(zs)
        ax.errorbar(f, e, yerr=er, marker="o", ms=3.2, lw=1.1, capsize=1.6, color=col)
        ax.text(f[-1]*1.25, e[-1], fr"$z_s\!=\!{zs:g}$", color=col, fontsize=8.5,
                 va="center", ha="left")
    ax.axvline(ADOPT, color="0.35", ls=":", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(log_format))
    ax.set_xlim(6e-6, 5.5e-2)
    ax.set_xlabel(r"$\epsilon_{\rm sub}\ \ (\kappa_{\rm thr,clump}/\kappa_{\rm thr,host})$")
    ax.set_ylabel(r"$\sigma^2_{\rm sub}(\kappa)$")
    if show_label:
        ax.text(0.04, 0.94, "(a)", transform=ax.transAxes, va="top", fontweight="bold")

def draw_panel_b(ax, show_label=False):
    ax.axhspan(ex_brute - 2*er_brute, ex_brute + 2*er_brute, color="0.85", zorder=0)
    ax.axhline(ex_brute, color="0.45", ls="--", lw=1.0,
                label=r"brute force, 3 seeds ($\pm2\sigma$)")
    ax.errorbar(f1, e1, yerr=er1, marker="o", ms=3.6, lw=1.2, capsize=2.0,
                 color="#3b5bdb", zorder=3, label=r"dynamic split, $z_s=1$")
    ax.axvline(ADOPT, color="#c92a2a", ls=":", lw=1.3)
    ax.text(ADOPT*1.7, -1.3e-5, r"adopted $10^{-5}$", color="#c92a2a", fontsize=8.5,
             rotation=90, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(log_format))
    ax.set_xlabel(r"$\epsilon_{\rm sub}$")
    ax.set_ylabel(r"$\sigma^2_{\rm sub}(\kappa)$")
    if show_label:
        ax.text(0.04, 0.94, "(b)", transform=ax.transAxes, va="top", fontweight="bold")

# 1. Save original 2-paneled plot
fig, (axA, axB) = plt.subplots(1, 2, figsize=FIGURE_SIZES["double"])
fig.subplots_adjust(**SUBPLOTS_ADJUST["double"])
draw_panel_a(axA, show_label=True)
draw_panel_b(axB, show_label=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"subhalo_factor_convergence_paper.{ext}", dpi=300)
plt.close(fig)

# 2. Save individual panel A
fig_a, ax_a = plt.subplots(figsize=FIGURE_SIZES["single"])
fig_a.subplots_adjust(**SUBPLOTS_ADJUST["single"])
draw_panel_a(ax_a, show_label=False)
for ext in ("pdf", "png"):
    fig_a.savefig(OUT_DIR / f"subhalo_factor_convergence_paper_a.{ext}", dpi=300)
plt.close(fig_a)

# 3. Save individual panel B
fig_b, ax_b = plt.subplots(figsize=FIGURE_SIZES["single"])
fig_b.subplots_adjust(**SUBPLOTS_ADJUST["single"])
draw_panel_b(ax_b, show_label=False)
for ext in ("pdf", "png"):
    fig_b.savefig(OUT_DIR / f"subhalo_factor_convergence_paper_b.{ext}", dpi=300)
plt.close(fig_b)

print(f"saved {OUT_DIR}/subhalo_factor_convergence_paper.{{pdf,png}} and separate panels _a, _b")
print(f"zs=1 plateau (f<=1e-4 mean) = {e1[f1<=1e-4].mean():.3e},  brute = {ex_brute:.3e} "
      f"({100*e1[f1<=1e-4].mean()/ex_brute:.0f}% of brute)")

# Write metadata
meta = {
    "script": str(Path(__file__).relative_to(ROOT)),
    "generated": datetime.datetime.utcnow().isoformat() + "Z",
    "git_commit": git_commit_short(),
    "outputs": [str(p.relative_to(ROOT)) for p in OUT_DIR.glob("subhalo_factor_convergence_paper_b.*")],
}
MD = ROOT / "paper_prod" / "metadata"
MD.mkdir(parents=True, exist_ok=True)
MD_path = MD / "subhalo_factor_convergence_paper_b.metadata.json"
MD_path.write_text(json.dumps(meta, indent=2))
