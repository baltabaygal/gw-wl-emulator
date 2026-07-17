"""Post-hoc un-shift repair of batch-anchor-contaminated shards (2026-07-13).

Follow-up to data/results/shard_screen/report.md: every sampler batch that
caught a monster ray (kappa >> 1) is uniformly shifted by ~ -2*kappa/n by the
empirical batch-mean kappa compensation in cpp/lensing.cpp::sample_lnmu.
Instead of EXCLUDING flagged shards (which over-corrects: the shards are valid
physics up to the shift), repair every shard in place:

    delta_s = -2 * sum(kappa_i, kappa_i >= 3) / n_s        (per shard)
    x_rep   = x[lnmu >= -2 ln 2] - delta_s                 (drop monsters, un-shift)

kappa_i = 1 + exp(-lnmu_i/2) for lnmu_i < -2 ln 2 (the identifiable demagnified
branch; mu ~ (kappa-1)^-2). This emulates, to first order in kappa/n, what the
robust-anchor fix (exclude kappa>1 from the mean) would have produced with the
same RNG stream. Limitations (documented, all sub-floor):
  * rays with 1 < kappa < 3 are not identifiable from lnmu (the mapping
    degenerates near kappa~2); their residual shift is <= 2*3/n ~ 2e-4 worst
    case, typically 0 (no such ray);
  * the shift is exact only in the linear lnmu ~ -2*kappa regime (the bulk);
  * shard boundaries reconstructed as len//8 blocks (same as shard_screen).

Recomputed per group, with the ORIGINAL stored bin edges (comparability):
  * validation: unrepaired JSDs must reproduce summary.json;
  * repaired floor_half, c_floor, per-config JSD, floor_pred, excess;
  * repaired block-null floor (exact C(16,8) enumeration over truth shards)
    + percentile of the repaired A|B split.

Output: data/results/unshift_repair/report.md (+ repaired_summary.json).
Numpy only; sandbox/Mac agnostic. Does NOT modify the study caches.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "data" / "results"
OUT = RES / "unshift_repair"
NPROC = 8
CUT = -2 * np.log(2.0)          # lnmu cut: identifiable monsters, kappa >= 3
STUDIES = ["mmin_convergence", "mmin_pd_convergence", "nz_convergence"]


def shards_of(x):
    per = len(x) // NPROC
    return [x[i * per:(i + 1) * per] for i in range(NPROC)], x[NPROC * per:]


def jsd_counts(ca, cb, w):
    eps = 1e-300
    p = np.clip(ca / (ca.sum() * w), eps, None)
    q = np.clip(cb / (cb.sum() * w), eps, None)
    p = p / (p * w).sum(); q = q / (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * float((p * np.log(p / m) * w).sum()
                       + (q * np.log(q / m) * w).sum())


def jsd_samples(a, b, edges):
    w = np.diff(edges)
    return jsd_counts(np.histogram(a, bins=edges)[0].astype(float),
                      np.histogram(b, bins=edges)[0].astype(float), w)


def repair_array(x, label, log):
    """Un-shift each len//8 shard by its own monster-ray anchor estimate."""
    x = x[np.isfinite(x)]
    sh, rem = shards_of(x)
    out = []
    for s, a in enumerate(sh):
        mon = a < CUT
        if mon.any():
            kap = 1.0 + np.exp(-a[mon] / 2.0)
            delta = -2.0 * kap.sum() / a.size
            out.append(a[~mon] - delta)
            log.append((label, s, a.size, int(mon.sum()),
                        float(kap.max()), float(delta)))
        else:
            out.append(a)
    if rem.size:
        out.append(rem[rem >= CUT])   # remainder: drop monsters, no shift est.
    return np.concatenate(out)


def repaired_truth_shards(arrA, arrB):
    """16 repaired truth shards (8 per half) for the block null."""
    shards = []
    for arr in (arrA, arrB):
        arr = arr[np.isfinite(arr)]
        sh, _ = shards_of(arr)
        for a in sh:
            mon = a < CUT
            if mon.any():
                kap = 1.0 + np.exp(-a[mon] / 2.0)
                a = a[~mon] + 2.0 * kap.sum() / a.size
            shards.append(a)
    return shards


def block_null(shards, edges):
    w = np.diff(edges)
    counts = np.array([np.histogram(s, bins=edges)[0]
                       for s in shards], float)
    tot = counts.sum(0)
    null = np.empty(12870)
    for i, comb in enumerate(combinations(range(16), 8)):
        c1 = counts[list(comb)].sum(0)
        null[i] = jsd_counts(c1, tot - c1, w)
    obs = jsd_counts(counts[:8].sum(0), counts[8:].sum(0), w)
    return (float(np.median(null)), float(np.percentile(null, 95)), obs,
            100.0 * float((null <= obs).mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = []          # per repaired shard: label, s, n, n_monster, kmax, delta
    results = {}
    val_lines = []

    for study in STUDIES:
        summ = json.loads((RES / study / "summary.json").read_text())
        for gk, g in sorted(summ.items()):
            if "edges" not in g:
                continue
            edges = np.array(g["edges"])
            fa = RES / study / f"{gk}_truthA.npy"
            fb = RES / study / f"{gk}_truthB.npy"
            if not (fa.exists() and fb.exists()):
                continue
            A0, B0 = np.load(fa), np.load(fb)

            # ---- validation on UNREPAIRED data: reproduce summary numbers
            fh0 = jsd_samples(A0[np.isfinite(A0)], B0[np.isfinite(B0)], edges)
            val_lines.append(
                f"| {study} | {gk} | floor_half | {g['floor_half']:.4e} | "
                f"{fh0:.4e} |")

            A = repair_array(A0, f"{study}/{gk}_truthA", log)
            B = repair_array(B0, f"{study}/{gk}_truthB", log)
            truth = np.concatenate([A, B])
            n_truth = truth.size

            fh = jsd_samples(A, B, edges)
            c = fh / (1.0 / A.size + 1.0 / B.size)
            bmed, b95, bobs, bpct = block_null(
                repaired_truth_shards(A0, B0), edges)

            grp = {"floor_half_rep": fh, "c_floor_rep": c,
                   "floor_half_orig": g["floor_half"],
                   "block_null_median_rep": bmed, "block_null_95_rep": b95,
                   "block_obs_rep": bobs, "block_pctl_rep": bpct,
                   "configs": {}}

            for ck, cfg in g["configs"].items():
                if ck == "truth":
                    continue
                fc = RES / study / f"{gk}_{ck}.npy"
                if not fc.exists():
                    continue
                X0 = np.load(fc)
                j0 = jsd_samples(X0[np.isfinite(X0)], np.concatenate(
                    [A0[np.isfinite(A0)], B0[np.isfinite(B0)]]), edges)
                val_lines.append(
                    f"| {study} | {gk} | {ck} | {cfg['jsd']:.4e} | {j0:.4e} |")
                X = repair_array(X0, f"{study}/{gk}_{ck}", log)
                j = jsd_samples(X, truth, edges)
                fp = c * (1.0 / X.size + 1.0 / n_truth)
                grp["configs"][ck] = {
                    "jsd_rep": j, "floor_pred_rep": fp,
                    "excess_rep": max(j - fp, 0.0),
                    "jsd_orig": cfg["jsd"], "excess_orig": cfg["jsd_excess"]}
            results[f"{study}/{gk}"] = grp

    # ------------------------------------------------------------ report
    lines = ["# Post-hoc un-shift repair of the batch-anchor contamination "
             "(2026-07-13)", "",
             "Every len//8 shard of every cached array un-shifted by its own "
             "monster-ray anchor estimate delta = -2*sum(kappa>=3)/n (rays "
             "dropped, batch shifted back); floors, JSDs, excesses and block "
             "nulls recomputed on the repaired arrays with the ORIGINAL bin "
             "edges. Repair emulates the robust-anchor fix to first order; "
             "study caches untouched. See module docstring for limitations.",
             "", "## Repaired shards (|delta| > 1e-4 shown)", "",
             "| array | shard | n | n_monster | kappa_max | delta |",
             "|:--|--:|--:|--:|--:|--:|"]
    for label, s, n, nm, kmax, d in sorted(log, key=lambda t: t[5]):
        if abs(d) > 1e-4:
            lines.append(f"| {label} | {s} | {n} | {nm} | {kmax:.0f} | "
                         f"{d:+.4f} |")
    ntiny = sum(1 for t in log if abs(t[5]) <= 1e-4)
    lines += ["", f"(+{ntiny} shards with |delta| <= 1e-4 also repaired)", ""]

    lines += ["## Repaired floors and excesses", "",
              "| group | floor_half orig -> rep | block med (rep) | "
              "block pctl (rep) | config | excess orig -> rep |",
              "|:--|:--|--:|--:|:--|:--|"]
    for gkey, grp in results.items():
        first = True
        for ck, cc in sorted(grp["configs"].items()):
            head = (f"| {gkey} | {grp['floor_half_orig']:.3e} -> "
                    f"{grp['floor_half_rep']:.3e} | "
                    f"{grp['block_null_median_rep']:.3e} | "
                    f"{grp['block_pctl_rep']:.0f} " if first
                    else "| | | | ")
            lines.append(head + f"| {ck} | {cc['excess_orig']:.3e} -> "
                                f"{cc['excess_rep']:.3e} |")
            first = False

    lines += ["", "## Validation (unrepaired recomputation vs summary.json)",
              "", "| study | group | quantity | summary | recomputed |",
              "|:--|:--|:--|--:|--:|"] + val_lines

    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    (OUT / "repaired_summary.json").write_text(
        json.dumps(results, indent=1, default=float))
    print(f"wrote {OUT}/report.md + repaired_summary.json "
          f"({len(log)} shards repaired)")


if __name__ == "__main__":
    main()
