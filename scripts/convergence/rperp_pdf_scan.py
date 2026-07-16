"""R_perp (clustering-scale) -> P(lnmu) scan for the correlated bias field
(bias_model=1, cpp/lensing.cpp BiasField1D; wired 2026-07-16).

Question: how does the one physical scale of the new bias layer — the
transverse window radius R_perp — shape the magnification PDF? R_perp is
parameterized by mass through the Lagrangian radius
    R_L(M) = (3 M / 4 pi rhoM0)^(1/3)   (comoving),
scanned M = 1e7 .. 1e20 Msun (one per decade; r200 alternative axis =
R_L / (200/OmegaM)^(1/3), i.e. ~8.6x smaller R at fixed M, ~1 decade shift
in M). Field discretization is the production one: L = 1.05 chi(z_s),
N_max = L/R_perp modes (exact sum).

Arms per z_s in {1, 5}: 14 field arms + anchors `legacy` (bias_model=0 iid)
and `nobias` (bias off). Protocol follows the kappa_thr/Mmin/Nz studies:
subprocess shards (NOT mp.Pool — worker segfaults hang silently), two seed
halves per config for JSD floors (shard-level blocks), fixed-<N>=100 rule,
subhalo off. kappa_anchor=1 (robust batch mean) everywhere — this is a NEW
study, so it opts into the 2026-07-13 batch-anchor fix; arms are compared
only against arms with the same anchor.

Run (Mac, test env, after `make build`):
  PY=/Users/baltabay/miniforge3/envs/test/bin/python
  $PY scripts/convergence/rperp_pdf_scan.py mc          # ~10 min at -j 8
  $PY scripts/convergence/rperp_pdf_scan.py report

Outputs: data/results/rperp_pdf_scan/{mc/*.npz, report.md},
         plots/rperp_pdf_scan.png
Seed namespace: 950_000_000 (validate smoke uses 950_000_001..; disjoint from
kappa_thr 6e8, vark 8.0-8.2e8, prototype 9.0e8).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "rperp_pdf_scan"
MCDIR = OUT / "mc"
PLOTS = REPO / "plots"

PI = np.pi
H, OM, S8 = 0.674, 0.315, 0.811
RHOM0 = OM * 277.394 * H ** 2                 # Msun / kpc^3 (cosmology.h)

ZS_LIST = [1.0, 5.0]
LOGM_LIST = list(range(7, 21))                # 1e7 .. 1e20, one per decade (user choice 2026-07-16)
NSHARD = 16                                   # shards 0-7 = half A, 8-15 = half B
NPERSHARD = 15_000                            # 240k per arm
SEED0 = 950_000_000
NBINS = 200                                   # JSD histogram bins


def RL_of_M(M):
    return (3.0 * M / (4.0 * PI * RHOM0)) ** (1.0 / 3.0)


def arms():
    out = [("legacy", dict()), ("nobias", dict(bias=False))]
    for lm in LOGM_LIST:
        out.append((f"m1e{lm:02d}",
                    dict(bias_model=1, bias_Rperp=float(RL_of_M(10.0 ** lm)))))
    return out


def shard_path(zs, arm, s):
    return MCDIR / f"z{zs:g}_{arm}_s{s:02d}.npz"


def shard_seed(zs, arm_idx, s):
    return SEED0 + 1_000_000 * int(zs * 10) + 10_000 * arm_idx + s


# ------------------------------------------------------------------ stages
def stage_shard(args):
    """Runs INSIDE a subprocess: one sampler call, one npz."""
    sys.path.insert(0, str(REPO / "build"))
    import gwlensing as gw
    kw = json.loads(args.kwargs)
    lnmu = gw.sample_lnmu(args.zs, OM, S8, H, Nreal=args.n, seed=args.seed,
                          kappa_anchor=1, **kw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, lnmu=np.asarray(lnmu, dtype=np.float32))


def stage_mc(args):
    MCDIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for zs in ZS_LIST:
        for ai, (arm, kw) in enumerate(arms()):
            for s in range(NSHARD):
                p = shard_path(zs, arm, s)
                if p.exists():
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), "shard",
                       "--zs", str(zs), "--seed", str(shard_seed(zs, ai, s)),
                       "--n", str(NPERSHARD), "--out", str(p),
                       "--kwargs", json.dumps(kw)]
                jobs.append((f"z{zs:g}/{arm}/s{s:02d}", cmd))
    print(f"{len(jobs)} shards to run, -j {args.jobs}")
    running, done, t0 = [], 0, time.time()
    while jobs or running:
        while jobs and len(running) < args.jobs:
            tag, cmd = jobs.pop(0)
            running.append((tag, subprocess.Popen(cmd)))
        for tag, proc in running[:]:
            rc = proc.poll()
            if rc is None:
                continue
            running.remove((tag, proc))
            done += 1
            if rc != 0:
                print(f"[FAIL rc={rc}] {tag}", flush=True)
            elif done % 16 == 0:
                print(f"[{done}] {tag}  ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.3)
    print(f"mc done in {time.time()-t0:.0f}s")


def _load(zs, arm, half=None):
    ss = range(NSHARD) if half is None else (
        range(NSHARD // 2) if half == "A" else range(NSHARD // 2, NSHARD))
    parts = []
    for s in ss:
        p = shard_path(zs, arm, s)
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(np.load(p)["lnmu"])
    return np.concatenate(parts).astype(np.float64)


def _jsd(x, y, edges):
    p = np.histogram(x, bins=edges)[0] + 0.0
    q = np.histogram(y, bins=edges)[0] + 0.0
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def stage_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    lines = ["# R_perp -> P(lnmu) scan (bias_model=1)\n",
             f"240k realizations/arm, kappa_anchor=1, fixed-<N>=100, "
             f"subhalo off; JSD: {NBINS} bins, floors from shard halves "
             f"(A = shards 0-7, B = 8-15).\n"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    summary = {}
    for iz, zs in enumerate(ZS_LIST):
        leg = _load(zs, "legacy")
        edges = np.linspace(np.quantile(leg, 1e-5) - 0.2,
                            max(np.quantile(leg, 1 - 1e-5) + 0.5, 3.0), NBINS + 1)
        rows = [f"\n## z_s = {zs:g}\n",
                "| arm | R_perp [Mpc] | JSD vs legacy | JSD vs nobias | floor "
                "| sigma(lnmu) | q99(mu) | q99.9(mu) | <1/mu>_I |",
                "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
        nob = _load(zs, "nobias")
        res = {}
        for arm, kw in arms():
            x = _load(zs, arm)
            fl = _jsd(_load(zs, arm, "A"), _load(zs, arm, "B"), edges)
            res[arm] = dict(
                jsd_leg=_jsd(x, leg, edges), jsd_nob=_jsd(x, nob, edges),
                floor=fl, sig=float(np.std(x)),
                q99=float(np.exp(np.quantile(x, 0.99))),
                q999=float(np.exp(np.quantile(x, 0.999))),
                invmu=float(np.mean(np.exp(-x))),
                Rp=float(kw.get("bias_Rperp", np.nan)))
            r = res[arm]
            rows.append(f"| {arm} | {r['Rp']/1e3 if np.isfinite(r['Rp']) else float('nan'):.3g} "
                        f"| {r['jsd_leg']:.2e} | {r['jsd_nob']:.2e} | {r['floor']:.2e} "
                        f"| {r['sig']:.4f} | {r['q99']:.3f} | {r['q999']:.3f} "
                        f"| {r['invmu']:.4f} |")
        lines += rows
        summary[zs] = res

        # --- plots: PDF overlays + summaries vs M
        ax = axes[iz, 0]
        ctr = 0.5 * (edges[1:] + edges[:-1])
        cmap = plt.get_cmap("viridis")
        for k, lm in enumerate(LOGM_LIST):
            x = _load(zs, f"m1e{lm:02d}")
            h = np.histogram(x, bins=edges, density=True)[0]
            ax.semilogy(ctr, h, color=cmap(k / (len(LOGM_LIST) - 1)), lw=0.9)
        h = np.histogram(leg, bins=edges, density=True)[0]
        ax.semilogy(ctr, h, "k--", lw=1.4, label="legacy iid")
        h = np.histogram(nob, bins=edges, density=True)[0]
        ax.semilogy(ctr, h, color="#e34948", ls=":", lw=1.4, label="no bias")
        ax.set_xlabel(r"$\ln\mu$"); ax.set_ylabel(r"$P(\ln\mu)$")
        ax.set_title(f"$z_s={zs:g}$: field arms, $M_{{clust}}=10^{{{LOGM_LIST[0]}}}..10^{{{LOGM_LIST[-1]}}}$")
        ax.legend(fontsize=8, frameon=False); ax.set_ylim(1e-6, None)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(
            vmin=LOGM_LIST[0], vmax=LOGM_LIST[-1]))
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label(r"$\log_{10} M_{\rm clust}/M_\odot$"
                     r"  ($R_\perp = R_L(M)$: 39 kpc $\to$ 844 Mpc)", fontsize=8)
        cb.set_ticks(LOGM_LIST[::3])
        cb.ax.set_yticklabels([f"{lm}\n{RL_of_M(10.0**lm)/1e3:.3g} Mpc"
                               for lm in LOGM_LIST[::3]], fontsize=7)

        Ms = np.array([10.0 ** lm for lm in LOGM_LIST])
        get = lambda key: np.array([res[f"m1e{lm:02d}"][key] for lm in LOGM_LIST])
        ax = axes[iz, 1]
        ax.loglog(Ms, get("jsd_leg"), "o-", color="#2a78d6", label="JSD vs legacy")
        ax.loglog(Ms, get("jsd_nob"), "s-", color="#1baf7a", label="JSD vs no-bias")
        ax.loglog(Ms, get("floor"), ":", color="#8a8776", label="floor (halves)")
        ax.set_xlabel(r"$M_{\rm clust}$ [$M_\odot$] ($R_\perp = R_L(M)$)")
        ax.set_ylabel("JSD"); ax.set_title(f"$z_s={zs:g}$: PDF distance vs scale")
        ax.legend(fontsize=8, frameon=False)
        ax = axes[iz, 2]
        ax.semilogx(Ms, get("sig"), "o-", color="#2a78d6", label=r"$\sigma(\ln\mu)$")
        ax.axhline(res["legacy"]["sig"], color="k", ls="--", lw=1, label="legacy")
        ax.axhline(res["nobias"]["sig"], color="#e34948", ls=":", lw=1, label="no bias")
        ax2 = ax.twinx()
        ax2.semilogx(Ms, get("q999"), "^-", color="#b8860b", ms=4, label=r"$q_{99.9}(\mu)$")
        ax2.set_ylabel(r"$q_{99.9}(\mu)$", color="#b8860b")
        ax.set_xlabel(r"$M_{\rm clust}$ [$M_\odot$]"); ax.set_ylabel(r"$\sigma(\ln\mu)$")
        ax.set_title(f"$z_s={zs:g}$: width / tail vs scale")
        ax.legend(fontsize=8, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(PLOTS / "rperp_pdf_scan.png", dpi=150)
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    np.savez(OUT / "summary.npz",
             **{f"z{zs:g}_{arm}_{k}": v for zs, res in summary.items()
                for arm, d in res.items() for k, v in d.items()})
    print("wrote", OUT / "report.md", "and", PLOTS / "rperp_pdf_scan.png")


def stage_pdfplot(args):
    """PDF-only detail figure: body (linear), tail (log), ratio to nobias."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(ZS_LIST), 3, figsize=(19, 5.5 * len(ZS_LIST)))
    for iz, zs in enumerate(ZS_LIST):
        leg, nob = _load(zs, "legacy"), _load(zs, "nobias")
        data = {lm: _load(zs, f"m1e{lm:02d}") for lm in LOGM_LIST}
        # body: linear y, dense bins around the peak
        blo, bhi = np.quantile(nob, 5e-4), np.quantile(leg, 0.995)
        bedges = np.linspace(blo - 0.05, bhi, 261)
        bctr = 0.5 * (bedges[1:] + bedges[:-1])
        # tail: log y, coarser bins to the far tail
        tlo, thi = blo - 0.1, max(np.quantile(data[LOGM_LIST[0]], 1 - 2e-5), 3.0)
        tedges = np.linspace(tlo, thi, 121)
        tctr = 0.5 * (tedges[1:] + tedges[:-1])
        # ratio: coarse bins for noise control
        redges = np.linspace(blo - 0.05, np.quantile(data[LOGM_LIST[0]], 1 - 5e-4), 61)
        rctr = 0.5 * (redges[1:] + redges[:-1])
        hnob_r = np.histogram(nob, bins=redges, density=True)[0]

        for k, lm in enumerate(LOGM_LIST):
            col = cmap(k / (len(LOGM_LIST) - 1))
            x = data[lm]
            axes[iz, 0].plot(bctr, np.histogram(x, bins=bedges, density=True)[0],
                             color=col, lw=1.0)
            axes[iz, 1].semilogy(tctr, np.histogram(x, bins=tedges, density=True)[0],
                                 color=col, lw=1.0)
            hr = np.histogram(x, bins=redges, density=True)[0]
            axes[iz, 2].plot(rctr, np.divide(hr, hnob_r, out=np.full_like(hr, np.nan),
                                             where=hnob_r > 0), color=col, lw=1.0)
        for j, (edges_, ctr_, ref_ls) in enumerate(
                ((bedges, bctr, None), (tedges, tctr, None), (redges, rctr, None))):
            ax = axes[iz, j]
            if j < 2:
                for arr, style, lab in ((leg, dict(color="k", ls="--", lw=1.6), "legacy iid"),
                                        (nob, dict(color="#e34948", ls=":", lw=1.6), "no bias")):
                    h = np.histogram(arr, bins=edges_, density=True)[0]
                    (ax.plot if j == 0 else ax.semilogy)(ctr_, h, **style, label=lab)
            ax.set_xlabel(r"$\ln\mu$")
        hleg_r = np.histogram(leg, bins=redges, density=True)[0]
        axes[iz, 2].plot(rctr, np.divide(hleg_r, hnob_r, out=np.full_like(hleg_r, np.nan),
                                         where=hnob_r > 0), "k--", lw=1.6, label="legacy iid")
        axes[iz, 2].axhline(1.0, color="#e34948", ls=":", lw=1.6, label="no bias")
        axes[iz, 0].set_ylabel(r"$P(\ln\mu)$")
        axes[iz, 0].set_title(f"$z_s={zs:g}$: body (linear)")
        axes[iz, 1].set_title(f"$z_s={zs:g}$: tail (log)")
        axes[iz, 1].set_ylim(1e-5, None)
        axes[iz, 2].set_title(f"$z_s={zs:g}$: ratio to no-bias")
        axes[iz, 2].set_ylabel(r"$P/P_{\rm nobias}$")
        axes[iz, 2].set_ylim(0.0, 3.5)
        axes[iz, 0].legend(fontsize=9, frameon=False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(
            vmin=LOGM_LIST[0], vmax=LOGM_LIST[-1]))
        cb = fig.colorbar(sm, ax=axes[iz, 2], pad=0.02)
        cb.set_label(r"$\log_{10} M_{\rm clust}/M_\odot$  ($R_\perp = R_L(M)$)",
                     fontsize=8)
        cb.set_ticks(LOGM_LIST[::3])
        cb.ax.set_yticklabels([f"{lm}\n{RL_of_M(10.0**lm)/1e3:.3g} Mpc"
                               for lm in LOGM_LIST[::3]], fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "rperp_pdf_only.png", dpi=150)
    print("wrote", PLOTS / "rperp_pdf_only.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["mc", "report", "shard", "pdfplot"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--zs", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--out", type=str)
    ap.add_argument("--kwargs", type=str, default="{}")
    args = ap.parse_args()
    {"mc": stage_mc, "report": stage_report, "shard": stage_shard,
     "pdfplot": stage_pdfplot}[args.stage](args)


if __name__ == "__main__":
    main()
