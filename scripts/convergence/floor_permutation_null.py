"""Permutation-null calibration of the JSD floors (2026-07-13).

Two questions, answered from the CACHED samples only (no new MC):

1. Were the "inflated" half-split floors (mmin_pd z=1, z=5) flukes of the one
   arbitrary seed split? -> Build the null distribution of the half-split JSD
   by re-splitting the 240k truth randomly R times; report where the original
   floor_half sits in it (its percentile).

2. Is each candidate config statistically converged? -> Exact two-sample
   permutation test: under H0 (candidate distribution == truth distribution)
   pool the candidate and truth samples, re-split at random, and count how
   often the null JSD >= the observed one: p_null = P(J_perm >= J_obs).
   p_null >~ 0.05: indistinguishable from noise ("converged" in the
   statistical sense); small p_null: resolved residual (which may still be
   IMMATERIAL vs the emulator KL 7.3e-3 -- the two criteria are separate).

Trick that makes this instant: binning is deterministic per sample, so a
random split of the pooled samples induces multivariate-hypergeometric bin
counts. We draw counts directly (O(bins) per replicate) instead of shuffling
480k floats. Exactly equivalent to permuting the samples.

Reads  data/results/{mmin,mmin_pd,nz}_convergence/{summary.json,*.npy}
Writes data/results/floor_permutation_null/report.md (+ .json)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "data" / "results"
OUT = RES / "floor_permutation_null"
STUDIES = ["mmin_convergence", "mmin_pd_convergence", "nz_convergence"]
R = 4000          # permutation replicates (cheap: O(bins) each)
SEED = 20260713   # fresh namespace; analysis-only, no MC RNG touched


def jsd_counts(ca, cb, w):
    """Same estimator as convergence_scan.density()+jsd(), from counts."""
    eps = 1e-300
    p = np.clip(ca / (ca.sum() * w), eps, None)
    q = np.clip(cb / (cb.sum() * w), eps, None)
    p = p / (p * w).sum(); q = q / (q * w).sum()
    m = 0.5 * (p + q)
    return 0.5 * float((p * np.log(p / m) * w).sum()
                       + (q * np.log(q / m) * w).sum())


def perm_null(rng, pooled_counts, n_first, w, r=R):
    """Null JSD distribution for random splits of pooled counts."""
    out = np.empty(r)
    for i in range(r):
        ca = rng.multivariate_hypergeometric(pooled_counts, n_first)
        out[i] = jsd_counts(ca.astype(float),
                            (pooled_counts - ca).astype(float), w)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}
    lines = ["# Permutation-null calibration of the JSD floors",
             "",
             f"R = {R} random re-splits per test, multivariate-hypergeometric "
             "on the shared 120-bin histograms (exactly equivalent to "
             "permuting the samples). floor pctl = percentile of the ORIGINAL "
             "seed-split floor_half within the null of random truth "
             "re-splits (>=97.5 flags an unlucky split). p_null = "
             "P(J_perm >= J_obs) under candidate==truth pooling; "
             "p_null >= 0.05 -> statistically converged. Materiality vs the "
             "emulator KL 7.3e-3 is a separate (physics) criterion.", ""]

    for study in STUDIES:
        sdir = RES / study
        summ = json.loads((sdir / "summary.json").read_text())
        for gk, g in sorted(summ.items()):
            if not gk.endswith("_main"):
                continue
            z = gk.split("_")[0][1:]
            edges = np.array(g["edges"]); w = np.diff(edges)
            tA = np.load(sdir / f"z{z}_main_truthA.npy")
            tB = np.load(sdir / f"z{z}_main_truthB.npy")
            tA, tB = tA[np.isfinite(tA)], tB[np.isfinite(tB)]
            cA = np.histogram(tA, bins=edges)[0]
            cB = np.histogram(tB, bins=edges)[0]
            ct = cA + cB

            # --- (1) null of the half-split floor
            null_half = perm_null(rng, ct, int(cA.sum()), w)
            fh = g["floor_half"]
            pctl = 100.0 * float((null_half <= fh).mean())
            grp = {"floor_half": fh,
                   "null_half_median": float(np.median(null_half)),
                   "null_half_p68": [float(np.percentile(null_half, 16)),
                                     float(np.percentile(null_half, 84))],
                   "null_half_p95": float(np.percentile(null_half, 95)),
                   "floor_half_pctl": pctl, "configs": {}}

            lines += [f"## {study} / z_s={z}", "",
                      f"original floor_half = {fh:.3e}, null median = "
                      f"{np.median(null_half):.3e} "
                      f"[68%: {np.percentile(null_half,16):.2e}.."
                      f"{np.percentile(null_half,84):.2e}], "
                      f"**original split at pctl {pctl:.0f}**", "",
                      "| config | J_obs | floor_pred | excess | p_null | "
                      "verdict |", "|:--|--:|--:|--:|--:|:--|"]

            # --- (2) exact permutation p for each candidate
            for cname, d in g["configs"].items():
                if cname == "truth":
                    continue
                f = sdir / f"z{z}_main_{cname}.npy"
                if not f.exists():
                    continue
                s = np.load(f); s = s[np.isfinite(s)]
                cc = np.histogram(s, bins=edges)[0]
                null = perm_null(rng, ct + cc, int(ct.sum()), w, r=R // 2)
                p = float((null >= d["jsd"]).mean())
                # a finite permutation test never gives exactly 0
                pstr = f"<{1.0/(R//2+1):.1e}" if p == 0 else f"{p:.3f}"
                verdict = ("converged (noise-level)" if p >= 0.05 else
                           "resolved residual")
                grp["configs"][cname] = {"jsd": d["jsd"],
                                         "floor_pred": d["floor_pred"],
                                         "excess": d["jsd_excess"],
                                         "p_null": p}
                lines.append(f"| {cname} | {d['jsd']:.3e} | "
                             f"{d['floor_pred']:.3e} | {d['jsd_excess']:.3e} "
                             f"| {pstr} | {verdict} |")
            lines.append("")
            results[f"{study}/{gk}"] = grp
            print(f"{study} z={z}: floor pctl {pctl:.0f}", flush=True)

    (OUT / "report.md").write_text("\n".join(lines))
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}/report.md")


if __name__ == "__main__":
    sys.exit(main())
