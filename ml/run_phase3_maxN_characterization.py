"""Phase 3 max-N characterization: how many events before the emulator's small
systematic density bias dominates the statistical posterior width?

The simulator has ZERO systematic bias against itself (only MC seed noise that
averages out), so its posterior converges on truth as ~1/sqrt(N). The NSF carries
a small but FIXED systematic density bias, so as N grows its posterior mean
settles at a constant offset from the converged-simulator mean while the width
keeps shrinking. The crossover -- where |nsf_mean - sim_mean| (bias) equals the
posterior std (width) -- is the practical max usable N per redshift slice.

Uses a converged simulator PMF grid (high nsim, computed once) and re-weights by
nested-subsample catalog counts for each N, so the reference is identical across
the sweep.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ml.phase3_common import build_grid, ensure_dir, load_nsf_model, normalize_log_grid
from ml.posterior_metrics import posterior_summary
from ml.nsf_likelihood import nsf_mixed_catalog_log_likelihood
from ml.run_phase3_posterior_grid import load_bin_edges
from ml.run_phase3_nscaling_control import sim_pmf_grid, make_truth_catalog, subsamples_for_N, loglik_from_pmf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=12)
    ap.add_argument("--zs", type=float, nargs="+", default=[0.5, 1.5, 2.5])
    ap.add_argument("--Ns", type=int, nargs="+", default=[150, 300, 600, 1200, 2400, 4800, 9600])
    ap.add_argument("--nsim", type=int, default=40000, help="converged simulator statistics for the reference grid")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--reps", type=int, default=16, help="independent catalog realizations to average over per N")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--model_path", default="data/models/conditional_nsf_backend_current.pt")
    ap.add_argument("--out_json", default="data/results/phase3_maxN_characterization.json")
    ap.add_argument("--out_md", default="docs/phase3_maxN_characterization.md")
    args = ap.parse_args()

    truth = (0.67, 0.30, 0.85)
    truth_d = {"h": truth[0], "OmegaM": truth[1], "sigma8": truth[2]}
    zs = list(args.zs)
    bin_edges = load_bin_edges()
    grid = build_grid("2d", args.resolution, fixed_h=truth[0])
    axes = grid["axes"]
    shape = tuple(len(v) for v in axes.values())
    params = [p for p in ("OmegaM", "sigma8")]

    print(f"Converged simulator PMF grid (nsim={args.nsim})...", flush=True)
    pmf = sim_pmf_grid(grid["theta"], zs, bin_edges, args.nsim, args.seed, args.workers)

    model = load_nsf_model(args.model_path)
    n_max = max(args.Ns) // len(zs) + 1

    # Draw `reps` independent truth catalogs so single-realization / grid-quantization
    # jitter averages out and the SYSTEMATIC emulator bias is isolated.
    print(f"Drawing {args.reps} independent truth catalogs...", flush=True)
    realizations = [make_truth_catalog(zs, n_max, truth, bin_edges, seed=700000 + 31 * r) for r in range(args.reps)]

    rows = []
    for N in args.Ns:
        # per-realization signed mean-offsets and widths
        diffs = {p: [] for p in params}
        widths = {p: [] for p in params}
        for per_z in realizations:
            subs = subsamples_for_N(per_z, zs, N)
            counts = []
            z_arr, lnmu_arr = [], []
            for z in zs:
                clamped = np.clip(subs[z], bin_edges[0] + 1e-9, bin_edges[-1] - 1e-9)
                c, _ = np.histogram(clamped, bins=bin_edges)
                counts.append(c)
                z_arr += [z] * len(subs[z]); lnmu_arr += list(subs[z])
            sim_post = normalize_log_grid(loglik_from_pmf(pmf, np.array(counts)).reshape(shape))
            nsf_ll = nsf_mixed_catalog_log_likelihood(model, np.array(z_arr), np.array(lnmu_arr), grid["theta"],
                                                      likelihood_mode="simulator_compatible", bin_edges=bin_edges).reshape(shape)
            nsf_post = normalize_log_grid(nsf_ll)
            sim_s = posterior_summary(axes, sim_post)
            nsf_s = posterior_summary(axes, nsf_post)
            for p in params:
                diffs[p].append(float(nsf_s[p]["mean"]) - float(sim_s[p]["mean"]))
                widths[p].append(float(sim_s[p]["std"]))
        rec = {"N": N}
        for p in params:
            d = np.array(diffs[p])
            syst = abs(float(np.mean(d)))           # systematic bias (realization noise averaged out)
            rms = float(np.sqrt(np.mean(d ** 2)))   # total NSF-vs-sim mean scatter
            width = float(np.mean(widths[p]))
            rec[p] = {
                "truth": truth_d[p],
                "systematic_bias": round(syst, 4),
                "rms_offset": round(rms, 4),
                "width_sim_std": round(width, 4),
                "syst_over_width": round(syst / max(width, 1e-9), 2),
            }
        rows.append(rec)
        msg = "  ".join(f"{p}: syst_bias={rec[p]['systematic_bias']:.4f} width={rec[p]['width_sim_std']:.4f} ratio={rec[p]['syst_over_width']:.2f}" for p in params)
        print(f"N={N:5d}  {msg}", flush=True)

    # crossover N per param: first N beyond which systematic bias STAYS >= width
    crossover = {}
    for p in params:
        cross = None
        for i, r in enumerate(rows):
            if all(rows[j][p]["syst_over_width"] >= 1.0 for j in range(i, len(rows))):
                cross = r["N"]; break
        crossover[p] = cross

    ensure_dir(Path(args.out_json).parent)
    Path(args.out_json).write_text(json.dumps({"truth": truth_d, "zs": zs, "nsim": args.nsim, "reps": args.reps,
                                               "model_path": args.model_path, "rows": rows,
                                               "crossover_N": crossover}, indent=2))

    L = ["# Phase 3 Max-N Characterization", "",
         f"Model: `{args.model_path}`  |  converged sim nsim={args.nsim}  |  {args.reps} catalog realizations averaged  |  z slices {zs}  |  grid 2d res {args.resolution}", "",
         "Systematic bias = |mean over realizations of (NSF mean - converged-sim mean)| (realization noise averaged out).",
         "Width = mean simulator posterior std. Emulator bias is sub-dominant while ratio < 1; it dominates once ratio >= 1.", "",
         "| N | OmegaM syst-bias | OmegaM width | OmegaM ratio | sigma8 syst-bias | sigma8 width | sigma8 ratio |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['N']} | {r['OmegaM']['systematic_bias']} | {r['OmegaM']['width_sim_std']} | {r['OmegaM']['syst_over_width']} "
                 f"| {r['sigma8']['systematic_bias']} | {r['sigma8']['width_sim_std']} | {r['sigma8']['syst_over_width']} |")
    L += ["", "## Crossover (max usable N before emulator bias persistently dominates statistical width)",
          f"- OmegaM: {crossover['OmegaM']}", f"- sigma8: {crossover['sigma8']}", "",
          "Below the crossover, the emulator is safe for inference (systematic bias < statistical width).", ""]
    ensure_dir(Path(args.out_md).parent)
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print("\nCrossover N:", crossover)
    print(f"Wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
