"""Calibrate a fast P(mu) provider against the stored simulator datasets.

    python -m hubble_reconstruct.calibrate summarize   # datasets -> cache npz
    python -m hubble_reconstruct.calibrate fit         # cache -> calibration json

Two stages so the expensive pass (reading ~30M float32 lnmu samples and taking
per-config quantiles) is paid once and the fit can be iterated cheaply.

WHAT THIS IS. The `datasets/` HDF5 files are real C++ engine output: each config
is a Latin-hypercube point in (z, Om, h, sigma8) with 10k-100k lnmu samples. We
do NOT interpolate them directly -- 3308 scattered points in 4-D is ~7 per
dimension, so nearest-neighbour lookup would give a discontinuous likelihood
whose sigma8 response is dominated by design noise. Instead we fit a SMOOTH
parametric description (a width law + a standardized shape template) and use
that. The result is still a parametric family, but its parameter response is
MEASURED rather than guessed -- which is the mock's main defect.

⚠⚠ THE DATASETS ARE OLD PHYSICS. Generated 2026-06-15/17, schema 1.1, i.e.
BEFORE the paper-defaults flip: subhalo_model=3 (not 5), bias_model=0 (legacy
iid, not the correlated field), no fil_bias/subhalo_virial, and kappa_anchor=0
-- the raw batch-mean anchoring whose monster-ray contamination CLAUDE.md
documents. Measured here: median <1/mu>_I = 1.006, 37% of configs above 1.01,
worst 1.39. So this calibration describes the JUNE simulator, not the paper's.
It is a much better stand-in than the analytic mock and it is still not the
production model. Recalibrate after the retrain.

⚠ Parameter space is 1+3d (z, Om, h, sigma8). Ob, ns and zeq are absent from
these datasets, so a provider built from them is inert in those three.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ._repo import PKG_ROOT, REPO_ROOT

CACHE = PKG_ROOT / "cache"
SUMMARY_NPZ = CACHE / "dataset_summaries.npz"
CALIB_JSON = CACHE / "calibration_1p3d.json"

# Quantiles summarizing each config. The low side (q01..q50) is tail-free and
# carries the width; the high side pins the tail.
QLEVELS = (0.5, 1.0, 2.5, 5.0, 10.0, 16.0, 25.0, 50.0,
           75.0, 84.0, 90.0, 95.0, 97.5, 99.0, 99.5, 99.9)
# Everything on disk is summarized into the cache; only CALIB_DATASETS are FIT.
DATASETS = ("tailrich", "logz_1k", "backend_current_1k", "large_1k", "large",
            "medium")

# ⚠ THE DATASETS ARE TWO DIFFERENT BACKEND GENERATIONS AND MUST NOT BE POOLED.
# Fitting all six together gives a 12.6% width residual; per-dataset offsets
# from the joint fit are
#     backend_current_1k  +9.5%   (2026-06-15)
#     tailrich            +4.5%   (2026-06-17)
#     logz_1k             +4.2%   (2026-06-16)
#     large               +0.2%   (n=16, uninformative)
#     large_1k           -11.8%   (2026-06-05)
#     medium             -11.8%   (2026-06-05)
# i.e. a ~21% systematic step in sigma(ln mu) between the 2026-06-05 and
# 2026-06-15+ engines -- a backend change, not scatter. Using only the newer
# generation drops the residual to 6.9%. The older sets stay in the cache so the
# offset stays visible and re-checkable; they are excluded from the fit.
CALIB_DATASETS = ("tailrich", "logz_1k", "backend_current_1k")
SPLITS = ("train", "validation", "test")
MIN_VALID = 1000          # configs with fewer valid samples are unusable
MIN_Z = 0.1               # below this the engine returns essentially no lensing
MIN_WIDTH = 0.005         # degenerate/zero-width configs


def summarize() -> None:
    """Pass 1: datasets -> per-config quantiles + flux, cached as npz."""
    import h5py

    rows, meta = [], []
    for name in DATASETS:
        for split in SPLITS:
            p = REPO_ROOT / "datasets" / name / split / f"dataset_{split}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                s = f["samples"]
                z, Om, h, s8 = (np.asarray(s[k]) for k in
                                ("z", "OmegaM", "h", "sigma8"))
                vc = np.asarray(s["valid_counts"])
                L = s["lnmu"]
                for i in range(len(z)):
                    n = int(vc[i])
                    if n < MIN_VALID:
                        continue
                    x = np.asarray(L[i, :n], dtype=np.float64)
                    q = np.percentile(x, QLEVELS)
                    rows.append([z[i], Om[i], h[i], s8[i], n,
                                 float(np.mean(np.exp(-x))), *q])
                    meta.append(f"{name}/{split}")
            print(f"  {name}/{split}: {len(z)} configs", flush=True)

    if not rows:
        raise SystemExit("no datasets found under datasets/")
    R = np.array(rows, dtype=float)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(SUMMARY_NPZ, table=R, qlevels=np.array(QLEVELS),
             source=np.array(meta))
    print(f"[summarize] {len(R)} configs -> {SUMMARY_NPZ}")


# ---------------------------------------------------------------------------
def _design(z, Om, h, s8):
    """Regression basis for ln(width). Chosen to reproduce the engine's
    z^{3/2}-ish growth with saturation, plus power laws in the three
    cosmological parameters."""
    one = np.ones_like(z)
    return np.column_stack([one, np.log(z), np.log1p(z),
                            np.log(s8), np.log(Om), np.log(h)])


BASIS_NAMES = ("const", "ln z", "ln(1+z)", "ln sigma8", "ln Om", "ln h")


def fit() -> dict:
    """Pass 2: cache -> calibration json (width law + shape template)."""
    d = np.load(SUMMARY_NPZ, allow_pickle=False)
    R, ql = d["table"], d["qlevels"]
    src = np.array([str(s).split("/")[0] for s in d["source"]])
    gen = np.isin(src, CALIB_DATASETS)
    print(f"[fit] backend generation filter: keeping "
          f"{', '.join(CALIB_DATASETS)} -> {gen.sum()} of {len(src)} configs "
          f"(see CALIB_DATASETS: the older sets differ ~21% in width)")
    R = R[gen]
    z, Om, h, s8, n, inv = (R[:, j] for j in range(6))
    Q = R[:, 6:]
    i50 = int(np.argmin(np.abs(ql - 50.0)))
    i05 = int(np.argmin(np.abs(ql - 5.0)))
    width = Q[:, i50] - Q[:, i05]

    keep = (z > MIN_Z) & (width > MIN_WIDTH) & np.isfinite(width)
    print(f"[fit] {keep.sum()} of {len(z)} configs pass "
          f"(z > {MIN_Z}, width > {MIN_WIDTH})")

    zk, Omk, hk, s8k, Qk, wk = z[keep], Om[keep], h[keep], s8[keep], Q[keep], width[keep]
    X = _design(zk, Omk, hk, s8k)
    beta, *_ = np.linalg.lstsq(X, np.log(wk), rcond=None)
    resid = np.log(wk) - X @ beta
    # robust second pass: drop >3 sigma outliers, refit
    ok = np.abs(resid) < 3.0 * resid.std()
    beta, *_ = np.linalg.lstsq(X[ok], np.log(wk[ok]), rcond=None)
    resid = np.log(wk) - X @ beta
    rms = float(np.exp(resid[ok].std()) - 1.0)
    print("[fit] ln width = " + " ".join(
        f"{b:+.4f}*{nm}" for b, nm in zip(beta, BASIS_NAMES)))
    print(f"[fit] width residual {100 * rms:.1f}% rms "
          f"({ok.sum()} inliers, {(~ok).sum()} dropped)")

    # Standardized shape: (q - q50) / width, per config, then a z-dependent
    # template (the shape is NOT z-universal -- the tail grows with z).
    S = (Qk - Qk[:, [i50]]) / wk[:, None]
    zedges = np.array([0.1, 0.35, 0.7, 1.25, 2.0, 3.5, 6.0, 10.5])
    zc, tmpl, cnt = [], [], []
    for a, b in zip(zedges[:-1], zedges[1:]):
        m = (zk >= a) & (zk < b)
        if m.sum() < 10:
            continue
        zc.append(float(np.sqrt(a * b)))
        tmpl.append(np.median(S[m], axis=0))
        cnt.append(int(m.sum()))
    tmpl = np.array(tmpl)
    spread = float(np.median(np.std(S, axis=0)))
    print(f"[fit] shape template on {len(zc)} z-bins "
          f"(counts {cnt}); pooled config-to-config scatter "
          f"{spread:.3f} in units of the width")

    calib = dict(
        created="2026-07-29",
        source=list(CALIB_DATASETS),
        excluded=("large_1k, large, medium: 2026-06-05 backend generation, "
                  "~21% narrower ln-mu width; pooling raises the residual "
                  "6.9% -> 12.6%"),
        physics_warning=("June-2026 engine: subhalo_model=3, bias_model=0, "
                         "kappa_anchor=0. NOT the paper/production config."),
        n_configs=int(keep.sum()),
        params=["z", "Om", "h", "sigma8"],
        basis=list(BASIS_NAMES),
        width_beta=[float(b) for b in beta],
        width_resid_rms=rms,
        qlevels=[float(x) for x in ql],
        z_centers=[float(x) for x in zc],
        shape_template=[[float(v) for v in row] for row in tmpl],
        shape_scatter=spread,
        flux_median=float(np.median(inv[keep])),
        flux_frac_above_1p01=float(np.mean(inv[keep] > 1.01)),
        z_range=[float(zk.min()), float(zk.max())],
        prior_box={k: [float(v.min()), float(v.max())]
                   for k, v in (("Om", Omk), ("h", hk), ("sigma8", s8k))},
    )
    CALIB_JSON.write_text(json.dumps(calib, indent=2))
    print(f"[fit] -> {CALIB_JSON}")
    return calib


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("summarize", "all"):
        summarize()
    if cmd in ("fit", "all"):
        fit()
