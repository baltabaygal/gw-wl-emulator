"""bias_window -> P(lnmu) scan for the correlated bias field (2026-07-20).

Follows the data/results/rperp_pdf_scan protocol exactly (subprocess shards,
16 shards/arm of 15k = 240k realizations, two seed halves for JSD floors,
kappa_anchor=1, fixed-<N>=100 rule, subhalo off) with the window as the new
axis:

  0 disk (legacy)   W = 2 J1(x)/x,             x = k_perp R
  1 spherical top-hat  W = 3(sin x - x cos x)/x^3,  x = |k| R
  2 Gaussian           W = exp(-x^2/2),             x = |k| R

Arms per z_s in {0.5, 1, 5}:
  legacy/nobias         anchors (bias_model=0 iid / bias off)
  t1e07..t1e20          top-hat R-scan, R = R_L(M), M = 1e7..1e20 (the §4.1 grid)
  d1e14                 disk at the production R_perp (like-for-like reference)
  tw1e14 / dw1e14       joint arms (bias_weak) at the production R_perp
  g_match               Gaussian at the variance-matched R_G (z_s = 1, 5)
  twN300                top-hat joint at <N>=300 (split-invariance gate; z_s = 1, 5)

Run (Mac, test env, after `make build`):
  PY=/Users/baltabay/miniforge3/envs/test/bin/python
  $PY scripts/convergence/bias_window_scan.py mc -j 8
  $PY scripts/convergence/bias_window_scan.py report

Outputs: data/results/bias_window/{mc/*.npz, summary.npz}, plots/bias_window_scan.png
Seed namespace: 951_000_000 (disjoint from rperp 950e6, kappa_thr 6e8,
vark 8.0-8.2e8, prototype 9.0e8).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "results" / "bias_window"
MCDIR = OUT / "mc"
PLOTS = REPO / "plots"

PI = np.pi
H, OM, S8 = 0.674, 0.315, 0.811
RHOM0 = OM * 277.394 * H ** 2                 # Msun / kpc^3 (cosmology.h)

ZS_LIST = [0.5, 1.0, 5.0]
LOGM_LIST = list(range(7, 21))                # R_L(1e7 .. 1e20), one per decade
RPERP_DEFAULT = 8441.0                        # lensing.h default = R_L(1e14)
RG_MATCHED = 3972.1                           # sigma^2_G(R_G) = sigma^2_TH(8441);
                                              # scripts/convergence/bias_window_sigmaR.py
NSHARD = 16                                   # shards 0-7 = half A, 8-15 = half B
NPERSHARD = 15_000                            # 240k per arm
SEED0 = 951_000_000
NBINS = 200                                   # JSD histogram bins


def RL_of_M(M):
    return (3.0 * M / (4.0 * PI * RHOM0)) ** (1.0 / 3.0)


def arms(zs):
    """(name, kwargs) per z_s. Order is FIXED: shard seeds key off the index,
    so append new arms at the end to keep the existing shard cache valid."""
    out = [("legacy", dict()), ("nobias", dict(bias=False))]
    for lm in LOGM_LIST:                       # top-hat R-scan
        out.append((f"t1e{lm:02d}",
                    dict(bias_model=1, bias_window=1,
                         bias_Rperp=float(RL_of_M(10.0 ** lm)))))
    out += [
        ("d1e14", dict(bias_model=1, bias_window=0, bias_Rperp=RPERP_DEFAULT)),
        ("tw1e14", dict(bias_model=1, bias_window=1, bias_Rperp=RPERP_DEFAULT,
                        bias_weak=True)),
        ("dw1e14", dict(bias_model=1, bias_window=0, bias_Rperp=RPERP_DEFAULT,
                        bias_weak=True)),
    ]
    if zs != 0.5:                              # z_s = 1, 5 only
        out += [
            ("g_match", dict(bias_model=1, bias_window=2, bias_Rperp=RG_MATCHED)),
            ("twN300", dict(bias_model=1, bias_window=1, bias_Rperp=RPERP_DEFAULT,
                            bias_weak=True, Nhalos=300)),
            # disk control for the split-invariance gate: the Cox-split residual
            # is a property of the approximation, not of the window, so the
            # top-hat's excess is only interpretable against the disk's in the
            # SAME protocol (appended last — keeps every earlier shard seed)
            ("dwN300", dict(bias_model=1, bias_window=0, bias_Rperp=RPERP_DEFAULT,
                            bias_weak=True, Nhalos=300)),
        ]
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
    # positional order is (z, OmegaM, sigma8, h) — h-first silently sets
    # sigma8 = 0.315 and kills the field power (CLAUDE.md item 13 gotcha)
    lnmu = gw.sample_lnmu(args.zs, OM, S8, H, Nreal=args.n, seed=args.seed,
                          kappa_anchor=1, **kw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, lnmu=np.asarray(lnmu, dtype=np.float32))


def stage_mc(args):
    MCDIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for zs in ZS_LIST:
        for ai, (arm, kw) in enumerate(arms(zs)):
            for s in range(NSHARD):
                p = shard_path(zs, arm, s)
                if p.exists():
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), "shard",
                       "--zs", str(zs), "--seed", str(shard_seed(zs, ai, s)),
                       "--n", str(NPERSHARD), "--out", str(p),
                       "--kwargs", json.dumps(kw)]
                jobs.append((f"z{zs:g}/{arm}/s{s:02d}", cmd))
    print(f"{len(jobs)} shards to run, -j {args.jobs}", flush=True)
    running, done, t0 = [], 0, time.time()
    fails = []
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
                fails.append(tag)
                print(f"[FAIL rc={rc}] {tag}", flush=True)
            elif done % 32 == 0:
                print(f"[{done}] {tag}  ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.3)
    print(f"mc done in {time.time()-t0:.0f}s, {len(fails)} failures")


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


def _clipped_var(x):
    """Var(lnmu) on the kappa_tot <= 1 core proxy: drop |lnmu - med| > 10 sd.
    Raw Var is monster-ray junk (one kappa~9 ray = 4e-4 shift)."""
    return float(np.var(x[np.abs(x - np.median(x)) < 10.0 * np.std(x)]))


def stage_report(args):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for zs in ZS_LIST:
        names = [a for a, _ in arms(zs)]
        have = [a for a in names if shard_path(zs, a, NSHARD - 1).exists()]
        if not have:
            continue
        leg = _load(zs, "legacy")
        edges = np.linspace(np.quantile(leg, 1e-5) - 0.2,
                            max(np.quantile(leg, 1 - 1e-5) + 0.5, 3.0), NBINS + 1)
        data = {a: _load(zs, a) for a in have}
        halves = {a: (_load(zs, a, "A"), _load(zs, a, "B")) for a in have}
        for a in have:
            x = data[a]
            key = f"z{zs:g}_{a}"
            summary[key + "_sigma"] = np.std(x)
            summary[key + "_var_clip"] = _clipped_var(x)
            summary[key + "_jsd_legacy"] = _jsd(x, data["legacy"], edges)
            summary[key + "_jsd_nobias"] = _jsd(x, data["nobias"], edges)
            summary[key + "_floor"] = _jsd(*halves[a], edges)
            summary[key + "_q01"] = np.quantile(x, 0.01)
            summary[key + "_q05"] = np.quantile(x, 0.05)
            summary[key + "_q99"] = np.exp(np.quantile(x, 0.99))
            summary[key + "_q999"] = np.exp(np.quantile(x, 0.999))
        # pairwise JSD matrix over the top-hat R-scan arms
        tn = [a for a in have if a.startswith("t1e")]
        mat = np.zeros((len(tn), len(tn)))
        for i, a in enumerate(tn):
            for j, b in enumerate(tn):
                if j > i:
                    mat[i, j] = mat[j, i] = _jsd(data[a], data[b], edges)
        summary[f"z{zs:g}_tophat_names"] = np.array(tn)
        summary[f"z{zs:g}_tophat_jsd"] = mat
        # window pairs at the production R_perp
        for tag, (a, b) in {"t_vs_d": ("t1e14", "d1e14"),
                            "tw_vs_dw": ("tw1e14", "dw1e14"),
                            "tw_vs_t": ("tw1e14", "t1e14"),
                            "dw_vs_d": ("dw1e14", "d1e14"),
                            "g_vs_t": ("g_match", "t1e14"),
                            "twN300_vs_tw": ("twN300", "tw1e14"),
                            "dwN300_vs_dw": ("dwN300", "dw1e14")}.items():
            if a in data and b in data:
                summary[f"z{zs:g}_jsd_{tag}"] = _jsd(data[a], data[b], edges)
        print(f"z_s = {zs:g}: {len(have)} arms summarized")
    np.savez(OUT / "summary.npz", **summary)
    print(f"wrote {OUT / 'summary.npz'}")
    _write_tables(summary)
    _plot(summary)


def _R_of_arm(a):
    if a.startswith("t1e"):
        return RL_of_M(10.0 ** int(a[3:]))
    if a in ("d1e14", "t1e14", "tw1e14", "dw1e14", "twN300"):
        return RPERP_DEFAULT
    if a == "g_match":
        return RG_MATCHED
    return np.nan


def _write_tables(summary):
    """MC tables for data/results/bias_window/report.md (narrative is hand-written
    there; this file is regenerated by `report` and pasted/linked from it)."""
    g = lambda k, d=np.nan: float(summary[k]) if k in summary else d
    lines = ["# bias_window scan — MC tables (auto-generated)\n",
             f"{NSHARD} shards x {NPERSHARD} = {NSHARD*NPERSHARD//1000}k "
             "realizations/arm, kappa_anchor=1, fixed-<N>=100, subhalo off; "
             f"JSD on P(lnmu), {NBINS} bins; floor = JSD between shard halves "
             "(A = 0-7, B = 8-15). Var_clip = Var(lnmu) after a 10-sd median "
             "clip (raw Var is monster-ray junk).\n"]
    for zs in ZS_LIST:
        names = [a for a, _ in arms(zs) if f"z{zs:g}_{a}_sigma" in summary]
        if not names:
            continue
        lines += [f"\n## z_s = {zs:g}\n",
                  "| arm | window | R_perp [Mpc] | sigma(lnmu) | Var_clip | "
                  "JSD vs legacy | JSD vs nobias | floor | q01(lnmu) | "
                  "q05(lnmu) | q99(mu) | q99.9(mu) |",
                  "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for a in names:
            k = f"z{zs:g}_{a}"
            win = ("disk" if a.startswith(("d", "legacy", "nobias")) else
                   "gauss" if a == "g_match" else "tophat")
            win = "-" if a in ("legacy", "nobias") else win
            R = _R_of_arm(a)
            lines.append(
                f"| {a} | {win} | {'' if np.isnan(R) else f'{R*1e-3:.3g}'} | "
                f"{g(k+'_sigma'):.4f} | {g(k+'_var_clip'):.3e} | "
                f"{g(k+'_jsd_legacy'):.2e} | {g(k+'_jsd_nobias'):.2e} | "
                f"{g(k+'_floor'):.2e} | {g(k+'_q01'):+.4f} | {g(k+'_q05'):+.4f} | "
                f"{g(k+'_q99'):.3f} | {g(k+'_q999'):.3f} |")
        # paired window / weak-arm comparisons
        lines += ["\n| comparison | JSD | floors (A/B of each arm) |",
                  "|:--|--:|:--|"]
        pairs = {"t1e14 vs d1e14 (window at fixed R)": ("t_vs_d", "t1e14", "d1e14"),
                 "tw1e14 vs dw1e14 (joint, window)": ("tw_vs_dw", "tw1e14", "dw1e14"),
                 "tw1e14 vs t1e14 (weak arm, top-hat)": ("tw_vs_t", "tw1e14", "t1e14"),
                 "dw1e14 vs d1e14 (weak arm, disk)": ("dw_vs_d", "dw1e14", "d1e14"),
                 "g_match vs t1e14 (window shape at matched var)": ("g_vs_t", "g_match", "t1e14"),
                 "twN300 vs tw1e14 (split invariance, top-hat)": ("twN300_vs_tw", "twN300", "tw1e14"),
                 "dwN300 vs dw1e14 (split invariance, DISK control)": ("dwN300_vs_dw", "dwN300", "dw1e14")}
        for label, (tag, a, b) in pairs.items():
            kk = f"z{zs:g}_jsd_{tag}"
            if kk not in summary:
                continue
            lines.append(f"| {label} | {g(kk):.2e} | "
                         f"{g(f'z{zs:g}_{a}_floor'):.1e} / {g(f'z{zs:g}_{b}_floor'):.1e} |")
    (OUT / "tables.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'tables.md'}")


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for zs, c in zip(ZS_LIST, ("C0", "C1", "C2")):
        tn = [a for a, _ in arms(zs) if a.startswith("t1e")
              and f"z{zs:g}_{a}_sigma" in summary]
        if not tn:
            continue
        R = np.array([_R_of_arm(a) for a in tn]) * 1e-3
        sig = np.array([summary[f"z{zs:g}_{a}_sigma"] for a in tn])
        vcl = np.array([summary[f"z{zs:g}_{a}_var_clip"] for a in tn])
        jnb = np.array([summary[f"z{zs:g}_{a}_jsd_nobias"] for a in tn])
        nb = float(summary[f"z{zs:g}_nobias_sigma"])
        axes[0].semilogx(R, sig / nb, "o-", color=c, label=f"$z_s={zs:g}$")
        axes[1].loglog(R, jnb, "o-", color=c)
        fl = np.array([summary[f"z{zs:g}_{a}_floor"] for a in tn])
        axes[1].semilogx(R, fl, ":", color=c, lw=1)
        axes[2].semilogx(R, vcl / float(summary[f"z{zs:g}_nobias_var_clip"]),
                         "o-", color=c)
        for ax in axes:
            ax.axvline(RPERP_DEFAULT * 1e-3, color="0.6", lw=1, ls="--", zorder=0)
    axes[0].set(xlabel=r"$R_\perp$ [Mpc]", ylabel=r"$\sigma(\ln\mu)\,/\,\sigma_{\rm nobias}$",
                title="top-hat window: PDF width vs scale")
    axes[1].set(xlabel=r"$R_\perp$ [Mpc]", ylabel="JSD vs nobias",
                title="clustering signal (dotted = shard-half floor)")
    axes[2].set(xlabel=r"$R_\perp$ [Mpc]",
                ylabel=r"Var$_{\rm clip}(\ln\mu)\,/\,$nobias",
                title="clipped variance vs scale")
    axes[0].legend(frameon=False)
    fig.suptitle(r"bias_window = 1 (spherical top-hat); dashed line = "
                 r"production $R_\perp = 8.44$ Mpc", y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS / "bias_window_scan.png", dpi=140, bbox_inches="tight")
    print(f"wrote {PLOTS / 'bias_window_scan.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("shard"); p.set_defaults(fn=stage_shard)
    p.add_argument("--zs", type=float); p.add_argument("--seed", type=int)
    p.add_argument("--n", type=int); p.add_argument("--out")
    p.add_argument("--kwargs", default="{}")
    p = sub.add_parser("mc"); p.set_defaults(fn=stage_mc)
    p.add_argument("-j", "--jobs", type=int, default=8)
    p = sub.add_parser("report"); p.set_defaults(fn=stage_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
