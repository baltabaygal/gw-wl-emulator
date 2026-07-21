#!/usr/bin/env python3
"""Per-ray weak/strong/total variance components vs custom_kappathr, bias era.

Production MC (halo-only: filaments/ell/subhalo OFF) with bias_model=1
(Rperp = production 8441 kpc) and bias_weak=True, sweeping custom_kappathr with
the floor-consistent background injection. Uses the kappa_weak diagnostic field
(2026-07-16) to split each ray into
    kappa_weak (background arm: conditional Cox draw)  and
    kappa_strong = kappa - kappa_weak (explicit encounters),
so Var_w + Var_s + 2 Cov = Var_tot holds EXACTLY on any shared ray mask.

Per kappa_thr, three estimators are recorded (same rays, shared masks):
  raw    — all rays. NOT a certified statistic: the formal clustering variance
           is dominated by >5 sigma field excursions (see the clamp scan in
           playground/sigma_partition_bias_vs_kthr.cpp) and by monster kappa
           rays, so raw Var is seed junk (standing rule).
  clip1  — rays with kappa_tot <= 1 (weak-lensing validity core, matches the
           kappa_anchor_cut convention).
  q999   — rays with kappa_tot <= per-sample q99.9 (tight body trim).

Writes playground/partition_components_zs<zs>[_deep]_seed<seedbase>.npz.
Run (repo root, test env):
  python playground/sweep_sigma_partition_components.py [zs=1] [nsamples=100000] [seedbase=0] [mode=main] [crn=0]
mode=main: 33-point grid 1e-5..1e3, fixed nsamples.
mode=deep: 6 half-decade points 1e-8..10^-5.5 with per-point adaptive nsamples
           (<N> ~ 1/kappa_thr makes 1e-8 cost ~0.2 s/ray); extends the main grid
           below the absolute background floor kappa_min = 1.28e-7, where the
           weak arm is empty by construction and the explicit arm picks up the
           sub-floor band the default path drops.
Launch several seedbases in parallel processes for ensembles.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "build")
import gwlensing  # noqa: E402

ZS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
SEEDBASE = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODE = sys.argv[4] if len(sys.argv) > 4 else "main"
CRN = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
RPERP = 8441.0

if MODE == "main":
    kthr = 10.0 ** (-5.0 + 8.0 * np.arange(33) / 32.0)
    nsamp_pt = np.full(kthr.size, NSAMP, dtype=int)
    seed0 = 20260800 + 100000 * SEEDBASE
    OUT = Path(f"playground/partition_components_zs{ZS:g}_seed{SEEDBASE}.npz")
elif MODE == "deep":
    kthr = 10.0 ** np.array([-8.0, -7.5, -7.0, -6.5, -6.0, -5.5])
    nsamp_pt = np.array([4000, 5000, 6000, 8000, 10000, 30000])
    seed0 = 20260800 + 100000 * SEEDBASE + 100   # offset: never overlaps main's i=0..32
    OUT = Path(f"playground/partition_components_zs{ZS:g}_deep_seed{SEEDBASE}.npz")
elif MODE == "deepcore":
    # The three deep points that are INSIDE the model domain (kappa_thr >=
    # kappa_min = 1.28e-7) and therefore actually plotted, at 4x the rays of
    # mode=deep. These carried the worst SEM on the figure (0.6-1.4% vs ~0.3%
    # on the main grid) purely because mode=deep sizes its samples for the
    # sub-floor points, which cost ~0.2 s/ray and are npz-only documentation.
    # In-domain they are cheap (~97 s/seed at the old sizes), so 4x is nearly
    # free. Sub-floor points are NOT regenerated here - take them from the
    # existing mode=deep files.
    kthr = 10.0 ** np.array([-6.5, -6.0, -5.5])
    nsamp_pt = np.array([32000, 40000, 120000])
    seed0 = 20260800 + 100000 * SEEDBASE + 300   # never overlaps main (i<=32) or deep (+100)
    OUT = Path(f"playground/partition_components_zs{ZS:g}_deepcore_seed{SEEDBASE}.npz")
elif MODE == "floortest":
    # Floor-constraint validation: lower the ABSOLUTE background floor to
    # kappa_min = 1e-3*KTHR_FLAT = 1e-11 (kappathr_flat only feeds the floor when
    # custom_kappathr drives the split — lensing.cpp:676-692,772). With the whole
    # sweep now INSIDE the model domain, sigma_total must be flat across
    # 1e-8..1e3 at a NEW (floor-dependent) plateau — proving the below-floor rise
    # of the production-floor run is purely the domain constraint, not a defect
    # in the split machinery.
    kthr = 10.0 ** np.array([-8.0, -7.5, -7.0, -6.5, -6.0, -5.5,
                             -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    nsamp_pt = np.array([4000, 5000, 6000, 8000, 10000, 30000,
                         100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000])
    seed0 = 20260800 + 100000 * SEEDBASE + 200
    OUT = Path(f"playground/partition_components_zs{ZS:g}_floortest_seed{SEEDBASE}.npz")
else:
    raise SystemExit(f"unknown mode {MODE!r}")
KTHR_FLAT = 1.0e-8 if MODE == "floortest" else -1.0

if CRN:
    # Common random numbers: one seed for the WHOLE kappa_thr grid instead of
    # seed0+i, so every point shares the same field/encounter random stream and
    # the estimation errors become positively correlated ALONG the sweep. Each
    # point stays individually unbiased; what shrinks is the point-to-point
    # scatter, which is what the sweep is actually measuring (flatness of
    # sigma_total). Measured on a 4-seed x 20k pilot: rms second-difference
    # roughness 2.71% -> 1.36% at identical cost (= a free 4x in samples).
    # CAVEAT: the ensemble now carries only <nseed> independent field
    # realizations rather than nseed*nkt, so the absolute PLATEAU level needs
    # the seed count for its error bar (>=32 seedbases) even though the shape
    # is well determined. Off by default: the default path is unchanged.
    OUT = OUT.with_name(OUT.stem + "_crn.npz")


def stats(kw, ks, mask):
    """(var_w, var_s, cov, var_t, mean_w, mean_s, n) on the masked rays."""
    w, s = kw[mask], ks[mask]
    t = w + s
    return (w.var(), s.var(), np.mean((w - w.mean()) * (s - s.mean())),
            t.var(), w.mean(), s.mean(), int(mask.sum()))


rows = {k: [] for k in ("raw", "clip1", "q999")}
qtot = []
for i, kt in enumerate(kthr):
    t0 = time.time()
    r = gwlensing.sample_lensing_raw_ml(
        z=ZS, h=0.674, OmegaM=0.315, sigma8=0.811,
        nsamples=int(nsamp_pt[i]), seed=seed0 if CRN else seed0 + i,
        filaments=False, bias=True, ell=False, subhalo=False,
        bias_model=1, bias_Rperp=RPERP, bias_weak=True,
        custom_kappathr=float(kt), kappathr_flat=KTHR_FLAT)
    ktot = np.asarray(r["kappa"])
    kw = np.asarray(r["kappa_weak"])
    ks = ktot - kw

    rows["raw"].append(stats(kw, ks, np.ones_like(ktot, bool)))
    rows["clip1"].append(stats(kw, ks, ktot <= 1.0))
    rows["q999"].append(stats(kw, ks, ktot <= np.quantile(ktot, 0.999)))
    qtot.append(np.quantile(ktot, [0.001, 0.5, 0.99, 0.999, 1.0]))

    st = rows["clip1"][-1]
    print(f"kt={kt:.3e}  clip1: sig_w={np.sqrt(st[0]):.6f} sig_s={np.sqrt(st[1]):.6f} "
          f"sig_t={np.sqrt(st[3]):.6f} (n={st[6]})  [{time.time()-t0:.0f}s]", flush=True)

save = {"kthr": kthr, "zs": ZS, "nsamples": NSAMP, "nsamples_pt": nsamp_pt,
        "rperp": RPERP, "kappathr_flat": KTHR_FLAT, "qtot": np.array(qtot),
        "crn": CRN}
for k, v in rows.items():
    save[k] = np.array(v)  # (nkt, 7): var_w var_s cov var_t mean_w mean_s n
np.savez(OUT, **save)
print("wrote", OUT)
