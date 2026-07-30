"""Driver: end-to-end HDR run.

  python -m hubble_reconstruct.run_forecast --arm bns_et --provider mock
  python -m hubble_reconstruct.run_forecast --arm both --n-samples 2400
  python -m hubble_reconstruct.run_forecast --demo-mismatch     # sec.5 bias demo
  python -m hubble_reconstruct.run_forecast --quick             # smoke, seconds

⚠ Every run on `--provider mock` is a PIPELINE TEST. The mock P(mu) is a
caricature; its width law is not the simulator's. Do not quote a sigma_8
precision from it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import catalogue as cat_mod
from . import plots
from ._repo import FIDUCIAL, PRIOR_6D, PRIOR_VASKONEN, THETA3_KEYS, THETA6_KEYS
from .likelihood import (GaussianApproxLikelihood, HDRLikelihood,
                         JointLikelihood, sensitivity)
from .mcmc import run_mcmc, summarize
from .pmu import get_provider
from .selection import RedshiftCut, SNRCut

OUT = Path(__file__).resolve().parent / "results"


def build(args):
    provider = get_provider(args.provider)
    theta_fid = dict(FIDUCIAL)
    rng = np.random.default_rng(args.seed)

    if args.selection == "snr":
        # Calibrate the horizon so the cut bites in the upper half of the z
        # range -- a threshold nothing reaches would make the demo vacuous.
        from .background import Background

        bg = Background(theta_fid)
        z_h = 1.4 if args.arm != "bmbh_lisa" else 6.0
        sel = SNRCut.from_horizon(float(bg.DL(z_h)), snr_thr=args.snr_thr)
    else:
        sel = RedshiftCut(2.0 if args.arm != "bmbh_lisa" else 10.0)

    cats = []
    if args.arm in ("bns_et", "both"):
        cats.append(cat_mod.generate_bns_et(
            provider, theta_fid, sel if args.selection == "snr" else RedshiftCut(2.0),
            n_events=args.n_bns, rng=rng))
    if args.arm in ("bmbh_lisa", "both"):
        cats.append(cat_mod.generate_bmbh_lisa(
            provider, theta_fid, sel if args.selection == "snr" else RedshiftCut(10.0),
            n_events=args.n_smbh, rng=rng))
    return provider, theta_fid, sel, cats


def make_like(cats, provider, sel, args, use_pdet=True, keys=THETA3_KEYS):
    prior = PRIOR_VASKONEN if keys == THETA3_KEYS else {
        k: PRIOR_6D[k] for k in keys
    }
    cls = GaussianApproxLikelihood if args.gaussian else HDRLikelihood
    parts = [
        cls(c, provider, sel, n_znodes=args.n_znodes, use_pdet=use_pdet,
            theta_keys=keys, prior=prior)
        for c in cats
    ]
    return parts[0] if len(parts) == 1 else JointLikelihood(parts)


def n_scan(args, n_list=(10, 20, 40, 75, 150, 300, 600, 1200, 2400), n_real=8):
    """sigma_8 precision vs catalogue size N.

    Uses the PROFILE-likelihood curvature rather than an MCMC per point: a
    parabola through log L(sigma_8) at fixed (Om, h) gives the CONDITIONAL
    error in ~15 likelihood calls, so the whole scan costs less than one chain.

    ⚠ Conditional, not marginalized. Marginalizing over Om and h widens it --
    at N = 300 the MCMC gives 9.7% where this gives ~8.4%. The SHAPE of the
    curve (the N-scaling, which is what the figure is about) is unaffected;
    the offset is reported by --n-scan so it is not silently dropped.

    ⚠ N is an ASSUMPTION about the survey, not something the data determine.
    Vaskonen asserts 300 ET bright sirens; De Leo+ (2026) decline to assert an
    absolute yield and parametrise by N, drawing nested sub-catalogues of
    N = 5-60 from a master catalogue. The real ceiling is merger rate x
    observing time x the fraction with a detectable EM counterpart, and that
    last factor is where the uncertainty lives.
    """
    provider = get_provider(args.provider)
    theta = dict(FIDUCIAL)
    sel = RedshiftCut(2.0)
    s8_0 = theta["sigma8"]
    grid = np.linspace(s8_0 - 0.07, s8_0 + 0.07, 9)

    out = dict(N=[], median=[], lo=[], hi=[], n_real=n_real)
    print(f"[n-scan] provider={args.provider}, {n_real} realizations per N")
    for N in n_list:
        errs = []
        for r in range(n_real):
            rng = np.random.default_rng(9000 + r)
            cat = cat_mod.generate_bns_et(provider, theta, n_events=N, rng=rng)
            like = HDRLikelihood(cat, provider, sel, n_znodes=args.n_znodes)
            v = np.array([like.log_likelihood([theta["Om"], s, theta["h"]])
                          for s in grid])
            c = np.polyfit(grid, v, 2)[0]
            if c < 0:
                errs.append(1.0 / np.sqrt(-2.0 * c) / s8_0)
        if not errs:
            continue
        e = np.array(errs)
        out["N"].append(N)
        out["median"].append(float(np.median(e)))
        out["lo"].append(float(np.percentile(e, 16)))
        out["hi"].append(float(np.percentile(e, 84)))
        print(f"  N={N:5d}  sigma8 err {100 * np.median(e):5.2f}%  "
              f"[{100 * np.percentile(e, 16):.2f}, "
              f"{100 * np.percentile(e, 84):.2f}]  "
              f"x sqrt(N/300) = {100 * np.median(e) * np.sqrt(N / 300):.2f}%")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mock",
                    choices=("mock", "dataset", "ace", "emulator"),
                    help="mock = analytic caricature; dataset = calibrated to "
                         "the stored June-2026 simulator datasets (1+3d, "
                         "measured parameter response); ace / emulator see "
                         "pmu.py")
    ap.add_argument("--arm", default="bns_et",
                    choices=("bns_et", "bmbh_lisa", "both"))
    ap.add_argument("--selection", default="redshift", choices=("redshift", "snr"))
    ap.add_argument("--snr-thr", type=float, default=20.0)
    ap.add_argument("--params", default="3d", choices=("3d", "6d"))
    ap.add_argument("--n-bns", type=int, default=300)
    ap.add_argument("--n-smbh", type=int, default=12)
    ap.add_argument("--n-znodes", type=int, default=24)
    ap.add_argument("--n-samples", type=int, default=2400)
    ap.add_argument("--n-burn", type=int, default=200)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--f-step", type=float, default=0.06,
                    help="proposal width as a fraction of the prior range; "
                         "0.06 gives ~22%% acceptance on the 300-event ET arm "
                         "(Vaskonen's target band is 10-28%%). Retune for 6d.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gaussian", action="store_true",
                    help="Gaussian-approximation control likelihood")
    ap.add_argument("--demo-mismatch", action="store_true",
                    help="sec.5 demo: SNR-selected catalogue analysed with and "
                         "without the matching P_det")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--n-scan", action="store_true",
                    help="sigma_8 precision vs catalogue size N (no MCMC); "
                         "writes sigma8_vs_N.png + .json")
    ap.add_argument("--n-scan-real", type=int, default=8,
                    help="catalogue realizations per N in --n-scan")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args(argv)

    if args.quick:
        args.n_bns, args.n_samples, args.n_burn = 60, 200, 50
        args.n_chains, args.n_znodes = 2, 8
    if args.demo_mismatch:
        args.selection = "snr"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    keys = THETA3_KEYS if args.params == "3d" else THETA6_KEYS

    t0 = time.time()

    if args.n_scan:
        scan = n_scan(args, n_real=args.n_scan_real)
        (outdir / "sigma8_vs_N.json").write_text(json.dumps(scan, indent=2))
        plots.sigma8_vs_N(
            scan, path=outdir / "sigma8_vs_N.png",
            title=f"HDR ET/BNS, provider={args.provider} "
                  f"(profile-likelihood, conditional)",
            banner=("MOCK P(mu) -- NOT a forecast" if args.provider == "mock"
                    else "June-2026 simulator physics via DatasetPDF -- "
                         "NOT the paper's config"),
        )
        print(f"\n[done] {time.time() - t0:.1f}s -> {outdir}")
        print("⚠ N is a SURVEY ASSUMPTION, not a measurement. Vaskonen asserts "
              "300; De Leo+ parametrise\n  by N (sub-catalogues of 5-60) rather "
              "than assert an absolute yield. Errors here are\n  CONDITIONAL "
              "(Om, h fixed); marginalizing widens them ~15% at N=300.")
        return scan

    provider, theta_fid, sel, cats = build(args)
    for c in cats:
        print(f"[cat] {c.arm}: {len(c)} events kept of {c.n_accepted} detected "
              f"/ {c.n_drawn} drawn (selection passes "
              f"{100 * c.n_accepted / max(c.n_drawn, 1):.1f}%), "
              f"rule={c.selection['name']} (mu_coupled="
              f"{c.selection['mu_coupled']})")
        if "mu" in c.truth and len(c.truth["mu"]):
            print(f"       <mu> in catalogue = {float(np.mean(c.truth['mu'])):.4f} "
                  f"(fair sample => ~<mu>_S; skewed high => selection bias)")
        c.save(outdir / f"catalogue_{c.arm}.npz")

    prior = PRIOR_VASKONEN if keys == THETA3_KEYS else {k: PRIOR_6D[k] for k in keys}
    ranges = [prior[k] for k in keys]
    x0 = [theta_fid[k] for k in keys]

    # Which parameters does the likelihood actually respond to? Inert ones make
    # R-hat look broken for a reason that is not a sampling problem.
    probe = make_like(cats, provider, sel, args, use_pdet=True, keys=keys)
    sens = sensitivity(probe, keys=keys)
    print("\n[sensitivity] max |dlogL| across each prior range:")
    for k in keys:
        tag = "  <-- INERT (posterior = prior)" if sens[k] < 1e-6 else ""
        print(f"  {k:>8s} {sens[k]:10.3f}{tag}")
    inert = [k for k in keys if sens[k] < 1e-6]
    if inert:
        print(f"  ⚠ {len(inert)} inert parameter(s): {', '.join(inert)}. Their "
              f"R-hat reflects random-walk mixing over the prior,\n"
              f"    not a convergence failure. With the mock provider Ob/ns are "
              f"inert BY CONSTRUCTION and zeq nearly so.")

    runs = {}
    variants = [("matched", True)]
    if args.demo_mismatch:
        variants.append(("mismatched_no_pdet", False))

    for label, use_pdet in variants:
        print(f"\n[mcmc] {label} (use_pdet={use_pdet}, params={args.params})")
        like = make_like(cats, provider, sel, args, use_pdet=use_pdet, keys=keys)
        res = run_mcmc(
            like, x0, prior_ranges=ranges, n_chains=args.n_chains,
            n_samples=args.n_samples, n_burn=args.n_burn, f_step=args.f_step,
            seed=args.seed + 1,
        )
        print(summarize(res["flat"], keys, truth=theta_fid))
        runs[label] = res
        np.savez(outdir / f"chains_{label}.npz", samples=res["samples"],
                 rhat=res["rhat"], keys=np.array(keys))
        plots.corner_plot(
            res["flat"], keys, truth=theta_fid,
            path=outdir / f"corner_{label}.png",
            title=f"HDR {args.arm} / {args.params} / {label}",
        )

    plots.hubble_diagram(cats[0], theta_fid, provider=provider,
                         path=outdir / "hubble_diagram.png")
    plots.pmu_panel(provider, theta_fid, path=outdir / "pmu_panel.png")

    if args.demo_mismatch:
        print("\n[sec.5 demo] sigma_8 pull, matched vs P_det omitted:")
        i8 = list(keys).index("sigma8")
        for label, res in runs.items():
            x = res["flat"][:, i8]
            m, s = float(x.mean()), float(x.std(ddof=1))
            print(f"  {label:>20s}: sigma8 = {m:.4f} +- {s:.4f}  "
                  f"pull = {(m - theta_fid['sigma8']) / s:+.2f} sd")
        print("  ⚠ ONE catalogue cannot show this. A single injection-recovery "
              "pull is N(0,1) by\n     construction, so these two numbers will "
              "usually look alike. The effect is\n     measured over many "
              "realizations by tests/run_gates.py::gate_selection\n"
              "     (HDR_GATE_NREAL=96: bias +0.0284+-0.0054 = +5.3 sigma when "
              "P_det is omitted,\n     +0.0023+-0.0048 = +0.5 sigma when "
              "matched). This demo shows the MACHINERY.")

    meta = dict(vars(args), keys=list(keys), theta_fid=theta_fid,
                runtime_s=round(time.time() - t0, 1))
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[done] {time.time() - t0:.1f}s -> {outdir}")
    if args.provider == "mock":
        print("⚠ MOCK P(mu): pipeline diagnostic only, NOT a forecast.")
    return runs


if __name__ == "__main__":
    main()
