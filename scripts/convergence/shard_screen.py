"""Cross-study shard screen + block-aware JSD floors (2026-07-13).

Follow-up to the batch-mean kappa compensation diagnosis
(data/results/floor_permutation_null/report.md): every sampler call shares an
anchor shift A_s ~= -2*kappa_max/n when the batch catches a monster ray, so
(a) which published studies were exposed?  -> screen ALL five study caches
    per shard: Delta_obs = median(shard) - median(siblings) vs
    Delta_pred = -2*kappa_max/n from the shard's own most extreme ray;
    the diagonal plot plots one against the other across every shard.
(b) sample-level permutation p-values are miscalibrated under block
    dependence -> recompute the truth-half floors with the correct resampling
    unit, the SHARD: exact enumeration of all C(16,8)=12870 ways to split the
    16 truth shards into two 8-shard groups.

Outputs: data/results/shard_screen/report.md,
         plots/batch_anchor_diagonal.png

Sandbox/Mac agnostic (numpy only, reads cached .npy + summary.json).
"""
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "data" / "results"
OUT = RES / "shard_screen"
NPROC = 8
NBINS = 120

STUDIES = ["mmin_convergence", "mmin_pd_convergence", "nz_convergence",
           "kappathr_subhalo_jsd", "subhalo_factor_jsd"]


def shards_of(x):
    per = len(x) // NPROC
    return [x[i * per:(i + 1) * per] for i in range(NPROC)], per


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    allv = np.sort(np.concatenate([a, b]))
    return float(np.abs(np.searchsorted(a, allv, "right") / a.size
                        - np.searchsorted(b, allv, "right") / b.size).max())


def jsd_counts(ca, cb, w):
    eps = 1e-300
    p = np.clip(ca / (ca.sum() * w), eps, None)
    q = np.clip(cb / (cb.sum() * w), eps, None)
    p = p / (p * w).sum(); q = q / (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * float((p * np.log(p / m) * w).sum()
                       + (q * np.log(q / m) * w).sum())


# ---------------------------------------------------------------- screen
def screen():
    rows, diag = [], []
    for study in STUDIES:
        for f in sorted(glob.glob(str(RES / study / "z*.npy"))):
            if f.endswith(("cleanfix.npy",)):
                continue
            key = Path(f).stem
            x = np.load(f)
            sh, per = shards_of(x)
            meds = np.array([np.nanmedian(s) for s in sh])
            sig_med = 1.2533 * np.nanmedian([np.nanstd(s) for s in sh]) \
                / np.sqrt(per)
            for s in range(NPROC):
                a = sh[s]; a = a[np.isfinite(a)]
                sib = np.concatenate([sh[j] for j in range(NPROC) if j != s])
                sib = sib[np.isfinite(sib)]
                mn = a.min()
                kmax = 1 + np.exp(-mn / 2) if mn < -1 else 0.0
                dpred = -2 * kmax / per
                dobs = float(np.median(a) - np.median(sib))
                n_k1 = int((a < -2 * np.log(2)) .sum())  # lnmu < -2ln2 ~ kappa>~3
                interesting = abs(dpred) > 2 * sig_med or abs(dobs) > 4 * sig_med
                ksr = ksf = np.nan
                if interesting:
                    a2 = a[a > mn]
                    ksr = ks(a2, sib)
                    ksf = ks(a2 - dpred, sib)
                diag.append((dpred, dobs, sig_med, study, key, s, interesting))
                if interesting:
                    rows.append((study, key, s, per, mn, kmax, dpred, dobs,
                                 dobs / sig_med, n_k1, ksr, ksf))
    return rows, diag


# ------------------------------------------------------- block-aware floors
def block_floors():
    out = []
    for study in STUDIES:
        summ = json.loads((RES / study / "summary.json").read_text())
        for gk, g in sorted(summ.items()):
            if "edges" not in g:
                continue
            # group key == file prefix in ALL five studies (z1_main, z1_on, z1)
            fa = RES / study / f"{gk}_truthA.npy"
            fb = RES / study / f"{gk}_truthB.npy"
            if not (fa.exists() and fb.exists()):
                continue
            edges = np.array(g["edges"]); w = np.diff(edges)
            A = np.load(fa); B = np.load(fb)
            shA, _ = shards_of(A[np.isfinite(A)])
            shB, _ = shards_of(B[np.isfinite(B)])
            counts = np.array([np.histogram(s, bins=edges)[0]
                               for s in shA + shB], float)   # (16, NBINS)
            idx = list(range(16))
            null = np.empty(12870)
            tot = counts.sum(0)
            for i, comb in enumerate(combinations(idx, 8)):
                c1 = counts[list(comb)].sum(0)
                null[i] = jsd_counts(c1, tot - c1, w)
            obs = jsd_counts(counts[:8].sum(0), counts[8:].sum(0), w)
            pctl = 100.0 * float((null <= obs).mean())
            out.append((study, gk, g.get("floor_half", np.nan),
                        float(np.median(null)),
                        float(np.percentile(null, 95)), obs, pctl))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, diag = screen()
    floors = block_floors()

    lines = ["# Cross-study shard screen + block-aware floors (2026-07-13)",
             "",
             "Mechanism under test: batch-mean kappa anchor "
             "(cpp/lensing.cpp::sample_lnmu) -> shared shard shift "
             "Delta ~= -2*kappa_max/n. Delta_pred from each shard's own most "
             "extreme ray; Delta_obs = median(shard) - median(siblings). "
             "KS_fix = KS vs siblings after dropping the ray and un-shifting.",
             "", "## Shards where the anchor shift is detectable", "",
             "| study | array | shard | n | lnmu_min | kappa_max | "
             "Delta_pred | Delta_obs | Dobs/sig | n(k>3) | KS_raw | KS_fix |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in sorted(rows, key=lambda t: t[6]):
        study, key, s, per, mn, kmax, dpred, dobs, z, n1, ksr, ksf = r
        lines.append(f"| {study} | {key} | {s} | {per} | {mn:.2f} | "
                     f"{kmax:.0f} | {dpred:+.4f} | {dobs:+.4f} | {z:+.1f} | "
                     f"{n1} | {ksr:.4f} | {ksf:.4f} |")

    lines += ["", "## Block-aware truth-half floors "
              "(exact enumeration of all C(16,8)=12870 shard splits)", "",
              "sample-level permutation destroys the shared shard shift and "
              "is too narrow; the shard is the exchangeable unit. pctl = "
              "where the ORIGINAL A|B split sits in the block null.", "",
              "| study | group | floor_half (sample) | block-null median | "
              "block-null 95% | observed A|B | pctl |",
              "|:--|:--|--:|--:|--:|--:|--:|"]
    for study, gk, fh, med, p95, obs, pctl in floors:
        lines.append(f"| {study} | {gk} | {fh:.3e} | {med:.3e} | {p95:.3e} | "
                     f"{obs:.3e} | {pctl:.0f} |")

    (OUT / "report.md").write_text("\n".join(lines))
    print(f"wrote {OUT}/report.md  ({len(rows)} detectable shards)")

    # ------------------------------------------------------ diagonal plot
    dp = np.array([d[0] for d in diag]); do = np.array([d[1] for d in diag])
    hot = np.array([d[6] for d in diag])
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=150)
    lim = max(1e-4, 1.3 * max(-dp.min(), -do.min()))
    ax.plot([-lim, 0], [-lim, 0], "-", color="#52514e", lw=1, zorder=1,
            label="Delta_obs = -2 kappa_max / n")
    ax.scatter(dp[~hot], do[~hot], s=8, alpha=0.35, color="#2a78d6",
               label="all shards (5 studies)")
    ax.scatter(dp[hot], do[hot], s=26, color="#e5484d", zorder=3,
               label="anchor shift detectable")
    ax.set(xscale="symlog", yscale="symlog")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlim(-lim, 2e-4); ax.set_ylim(-lim, 2e-4)
    ax.set_xlabel("predicted shift  -2 kappa_max / n")
    ax.set_ylabel("observed shift  median(shard) - median(siblings)")
    ax.set_title("Batch-anchor coupling across all study caches\n"
                 "(one point per shard)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPO / "plots" / "batch_anchor_diagonal.png",
                bbox_inches="tight")
    print("wrote plots/batch_anchor_diagonal.png")


if __name__ == "__main__":
    main()
