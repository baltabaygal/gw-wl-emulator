"""
MC: the substructure boost sigma_ON/sigma_OFF vs source redshift, z_s in [0.2, 10].

WHY THIS EXISTS. The 2026-07-27 threshold sweep
(`playground/sigma_on_off_vs_subkappathr_model5_crn.json`, figure
`paper_prod/plots/figures/fig_subhalo_sigma_ratio_vs_subkappathr.*`) measured the
substructure boost at z_s = 1 only. The analytic companion
(`playground/analytic/sweep_subkappathr_zs.py`) predicts the THRESHOLD DEPENDENCE at any
z_s exactly, but it cannot predict the boost AMPLITUDE: that needs the host + smooth-field
variance, which the MC measures on a kappa <= 1 clipped estimator while the Campbell
integral is unclipped. So the amplitude has to be measured, per z_s. This driver does it.

MUST RUN ON THE MAC in the `test` conda env -- build/gwlensing is a
cpython-312-darwin .so:
    PY=/Users/baltabay/miniforge3/envs/test/bin/python
    $PY scripts/convergence/subhalo_kappathr_zs_mc.py

ESTIMATOR (user-confirmed 2026-07-27, matching the existing z_s=1 json):
    x     = sample_lensing_raw_ml(...) -> raw kappa, NOT anchored
    sigma = std(kappa[kappa <= 1])     -> the certified kappa_tot <= 1 core mask
    mean  = mean(kappa)                -> unanchored, reported for provenance
CLAUDE.md item 12/13: raw sigma_kappa over the full support is monster-ray junk; the
kappa <= 1 clip is the certified estimator. Do NOT switch to unclipped sigma here.

FOUR CONFIGS PER z_s (user-chosen scope):
    off      substructure off                          -> sigma_OFF
    deep     model 5, factor 1e-3                      -> saturated substructure
    prod     model 5, factor 0.1 (production default)  -> the number that ships
    null     model 5, factor 1e5                       -> nothing rendered; ratio == 1 by
             construction (Msum = 0, uncarved host, cpp/lensing.cpp:975), so it
             calibrates the common-mode offset of the shared sigma_OFF at THIS z_s
The null point is what makes each z_s self-calibrating; without it the boost inherits an
uncontrolled ~0.3% normalization error (see the docstring of
paper_prod/scripts/plot_fig_subhalo_sigma_ratio_vs_subkappathr.py).

PARALLELISM: subprocess shards, never mp.Pool -- CLAUDE.md item 12, worker faults in
mp.Pool surface as silent hangs. Shards are seed-disjoint and pooled by concatenation.

REGRESSION GATE: at z_s = 1 the prod/off ratio must reproduce the cached sweep's
kappa_thr,sub = 0.1*kappa_thr point (ratio 1.0425, sigma_OFF 0.0272769) to within the
sampling error printed by the script. If it does not, the estimator or the config drifted
and nothing downstream should be trusted.

Writes data/results/subkappathr_zs_mc/boost_vs_zs.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable

Z_SOURCES = (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
N_RAY = 300_000
N_SHARD = 8                      # 8 beats 10 on the 10-core box (CLAUDE.md item 11)
SEED0 = 20260727

# factor = subhalo_kappathr_factor, i.e. kappa_thr,sub / kappa_thr(z_s)
CONFIGS = (("off", None), ("deep", 1e-3), ("prod", 0.1), ("null", 1e5))

OUT_DIR = REPO / "data" / "results" / "subkappathr_zs_mc"
SCRATCH = REPO / "tmp" / "subkappathr_zs_mc"

SHARD_SRC = '''
import sys, numpy as np
sys.path.insert(0, "build")
import gwlensing as gw
zs, nray, seed, out = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
factor = sys.argv[5]
kw = dict(kappa_anchor=1)
if factor == "off":
    kw.update(subhalo=False)
else:
    kw.update(subhalo=True, subhalo_carve=True, subhalo_model=5,
              subhalo_kappathr_factor=float(factor))
raw = gw.sample_lensing_raw_ml(zs, 0.315, 0.811, 0.674, nray, seed, **kw)
np.save(out, np.asarray(raw["kappa"], dtype=float))
'''


def clipped_sigma(kappa: np.ndarray):
    """sigma on the certified kappa <= 1 core mask, plus the unclipped mean."""
    core = kappa[kappa <= 1.0]
    return float(core.std(ddof=1)), float(kappa.mean()), int(kappa.size - core.size)


def run_config(zs: float, tag: str, factor, shard_py: Path) -> dict:
    """Launch N_SHARD seed-disjoint subprocesses, pool their kappa arrays."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    per = N_RAY // N_SHARD
    arg = "off" if factor is None else repr(float(factor))
    procs, outs = [], []
    for s in range(N_SHARD):
        o = SCRATCH / f"zs{zs}_{tag}_s{s}.npy"
        outs.append(o)
        procs.append(subprocess.Popen(
            [PY, str(shard_py), str(zs), str(per),
             str(SEED0 + 1000 * s + hash(tag) % 97), str(o), arg],
            cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
    for s, p in enumerate(procs):
        _, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"shard {s} of z_s={zs} {tag} failed:\n"
                               f"{err.decode()[-2000:]}")
    kappa = np.concatenate([np.load(o) for o in outs])
    sig, mean, n_clip = clipped_sigma(kappa)
    # seed-split spread -> the honest per-config error, no distributional assumption
    per_shard = [clipped_sigma(np.load(o))[0] for o in outs]
    return {"sigma": sig, "mean": mean, "n_rays": int(kappa.size),
            "n_clipped": n_clip, "sigma_shards": per_shard,
            "sigma_sem": float(np.std(per_shard, ddof=1) / np.sqrt(len(per_shard)))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    shard_py = SCRATCH / "_shard.py"
    shard_py.write_text(SHARD_SRC)

    out_path = OUT_DIR / "boost_vs_zs.json"
    results = json.loads(out_path.read_text())["results"] if out_path.exists() else {}

    for zs in Z_SOURCES:
        if str(zs) in results:
            print(f"z_s={zs}: cached, skipping")
            continue
        row, t0 = {}, time.time()
        for tag, factor in CONFIGS:
            row[tag] = run_config(zs, tag, factor, shard_py)
            print(f"  z_s={zs:<5g} {tag:<5} sigma={row[tag]['sigma']:.6f} "
                  f"+-{row[tag]['sigma_sem']:.6f}  mean={row[tag]['mean']:.6f}  "
                  f"clipped={row[tag]['n_clipped']}  ({time.time() - t0:.0f}s)",
                  flush=True)

        s_off = row["off"]["sigma"]
        c = row["null"]["sigma"] / s_off          # common-mode calibration at THIS z_s
        for tag in ("deep", "prod", "null"):
            row[tag]["ratio_raw"] = row[tag]["sigma"] / s_off
            row[tag]["ratio_cal"] = row[tag]["sigma"] / s_off / c
        row["c_common_mode"] = c
        # error on c vs 1: the shared sigma_OFF term does not average down
        row["c_err"] = float(np.hypot(row["null"]["sigma_sem"] / s_off,
                                      row["off"]["sigma_sem"] / s_off))
        print(f"  -> z_s={zs}: null calib c={c:.5f}+-{row['c_err']:.5f}, "
              f"boost(prod)={100 * (row['prod']['ratio_cal'] - 1):.2f}%, "
              f"boost(deep)={100 * (row['deep']['ratio_cal'] - 1):.2f}%, "
              f"prod/deep={row['prod']['ratio_cal'] / row['deep']['ratio_cal']:.5f}")
        results[str(zs)] = row
        out_path.write_text(json.dumps(
            {"n_ray": N_RAY, "n_shard": N_SHARD, "seed0": SEED0,
             "estimator": "std(kappa[kappa<=1]) from sample_lensing_raw_ml, kappa_anchor=1",
             "configs": {t: f for t, f in CONFIGS}, "results": results}, indent=2))
        print(f"  saved {out_path}", flush=True)

    # ---- regression gate vs the cached z_s = 1 threshold sweep ---------------------
    ref = REPO / "playground" / "sigma_on_off_vs_subkappathr_model5_crn.json"
    if "1.0" in results and ref.exists():
        rj = json.loads(ref.read_text())
        kthr = results["1.0"].get("kappa_thr")
        target = 0.1 * (kthr if kthr else 1.2777483020e-04)
        i = int(np.argmin([abs(np.log(r["kthr_sub"] / target)) for r in rj["rows"]]))
        r_ref, r_new = rj["rows"][i]["ratio"], results["1.0"]["prod"]["ratio_raw"]
        err = np.hypot(results["1.0"]["prod"]["sigma_sem"],
                       results["1.0"]["off"]["sigma_sem"]) / results["1.0"]["off"]["sigma"]
        print(f"\nREGRESSION GATE (z_s=1, raw ratios, no calibration):")
        print(f"  cached sweep at kappa_thr,sub={rj['rows'][i]['kthr_sub']:.3e}: {r_ref:.5f}")
        print(f"  this run:                                    {r_new:.5f} +- {err:.5f}")
        print(f"  sigma_OFF: cached {rj['sig_off']:.6f}  vs  "
              f"{results['1.0']['off']['sigma']:.6f}")
        print(f"  -> {'PASS' if abs(r_ref - r_new) < 3 * max(err, 1e-9) else 'INVESTIGATE'}")

    print("\n  z_s    sigma_OFF   c      boost(prod)  boost(deep)  prod/deep")
    for z in sorted(float(k) for k in results):
        r = results[str(z)]
        print(f"{z:6.2f}  {r['off']['sigma']:.6f}  {r['c_common_mode']:.4f}  "
              f"{100 * (r['prod']['ratio_cal'] - 1):9.2f}%  "
              f"{100 * (r['deep']['ratio_cal'] - 1):9.2f}%  "
              f"{r['prod']['ratio_cal'] / r['deep']['ratio_cal']:.5f}")


if __name__ == "__main__":
    main()
