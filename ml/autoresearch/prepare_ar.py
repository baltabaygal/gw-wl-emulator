"""FIXED harness for the autoresearch loop — DO NOT MODIFY during experiments.

Provides the honest metric against simulator ground truth:

  - TLSE   : Tail Log-Survival Error (primary). Mean over high-z eval points and
             thresholds t of |log10 S_nsf(mu>t) - log10 S_sim(mu>t)|.
  - BodyNLL: mean NLL of the flow on simulator samples in the science window
             lnmu in [-0.5, 2.5] (guard against trading body for tail).
  - SCORE  : TLSE + 0.3 * max(0, BodyNLL - BODY_BUDGET).   (lower is better)

Ground truth (simulator survival + body samples) is generated once and cached.
The metric consumes a model-agnostic callable:

    log_prob_fn(lnmu: np.ndarray[N], z: float, theta=(h,om,s8)) -> np.ndarray[N]

returning log density in *raw* lnmu space. This decouples the metric from the
model architecture so train_ar.py can change the model freely.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import numpy as np

AR_DIR = Path(__file__).resolve().parent
REPO_ROOT = AR_DIR.parents[1]
CACHE_PATH = AR_DIR / "cache" / "groundtruth.npz"

# ---- Evaluation panel -------------------------------------------------------
CENTRAL = (0.67, 0.30, 0.85)      # prior-center cosmology
STRESS = (0.72, 0.38, 1.00)       # high-structure corner (the reported failure)
TAIL_ZS = (2.0, 3.5, 5.0, 8.0)    # high-z where the far tail is fattest
BODY_ANCHORS = ((0.5, CENTRAL), (1.0, CENTRAL))  # extra low/mid-z body checks
THRESHOLDS = (2.0, 3.0, 4.0, 5.0)  # mu survival thresholds for TLSE
N_SIM = 400_000                    # simulator samples per eval point
SEED = 100
BODY_LO, BODY_HI = -0.5, 2.5       # science window in lnmu (mu ~ [0.61, 12.18])
N_BODY_KEEP = 60_000               # subsample of body samples kept for NLL eval

# Calibrated once from the baseline run (BodyNLL=-0.209): allow ~0.1 nat of body
# regression before penalizing. Body fidelity is a guard; TLSE is the driver.
BODY_BUDGET = -0.10
EPS = 1e-6                         # survival floor for log10

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _eval_points():
    """Returns list of (tag, z, theta, is_tail) for the full panel."""
    pts = []
    for cosmo_name, theta in (("central", CENTRAL), ("stress", STRESS)):
        for z in TAIL_ZS:
            pts.append((f"{cosmo_name}_z{z}", float(z), theta, True))
    for z, theta in BODY_ANCHORS:
        pts.append((f"body_z{z}", float(z), theta, False))
    return pts


def _import_simulator():
    build = str(REPO_ROOT / "build")
    if build not in sys.path:
        sys.path.insert(0, build)
    import gwlensing  # type: ignore
    return gwlensing


def build_ground_truth(force: bool = False) -> dict:
    """Sample the simulator across the panel and cache survivals + body samples."""
    if CACHE_PATH.exists() and not force:
        return load_ground_truth()

    gw = _import_simulator()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    tags, zs, thetas, is_tail = [], [], [], []
    surv = []                 # (P, len(THRESHOLDS)) simulator survivals
    body_arrays = {}          # tag -> body lnmu samples (subsampled)
    rng = np.random.default_rng(0)

    for tag, z, theta, tail in _eval_points():
        h, om, s8 = theta
        res = gw.sample_lnmu_ml_with_diagnostics(float(z), float(h), float(om), float(s8),
                                                 int(N_SIM), int(SEED), False)
        lnmu = np.asarray(res["lnmu"], dtype=np.float64)
        lnmu = lnmu[np.isfinite(lnmu)]
        mu = np.exp(lnmu)
        s = np.array([float(np.mean(mu > t)) for t in THRESHOLDS], dtype=np.float64)

        body = lnmu[(lnmu >= BODY_LO) & (lnmu <= BODY_HI)]
        if body.size > N_BODY_KEEP:
            body = rng.choice(body, size=N_BODY_KEEP, replace=False)

        tags.append(tag); zs.append(z); thetas.append(theta)
        is_tail.append(tail); surv.append(s); body_arrays[f"body_{tag}"] = body
        print(f"[gt] {tag:14s} z={z:<4} n={lnmu.size:>7} "
              f"S(mu>2,3,4,5)={np.array2string(s, precision=4, floatmode='fixed')}")

    meta = dict(
        tags=np.array(tags), zs=np.array(zs, dtype=np.float64),
        thetas=np.array(thetas, dtype=np.float64), is_tail=np.array(is_tail),
        surv=np.array(surv, dtype=np.float64),
        thresholds=np.array(THRESHOLDS, dtype=np.float64),
    )
    np.savez(CACHE_PATH, **meta, **body_arrays)
    print(f"[gt] cached -> {CACHE_PATH}")
    return load_ground_truth()


def load_ground_truth() -> dict:
    if not CACHE_PATH.exists():
        return build_ground_truth()
    d = np.load(CACHE_PATH, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["tags"] = [str(t) for t in out["tags"]]
    return out


def _survival_nsf(log_prob_fn, z, theta) -> np.ndarray:
    """Flow survival S(mu>t) = integral_{ln t}^{inf} p(lnmu) dlnmu, via fine grid."""
    grid = np.linspace(-3.0, 7.0, 5000)         # mu up to ~1100
    logp = log_prob_fn(grid, float(z), tuple(float(x) for x in theta))
    p = np.exp(np.asarray(logp, dtype=np.float64))
    out = []
    for t in THRESHOLDS:
        m = grid > np.log(t)
        out.append(float(_trapz(p[m], grid[m])) if m.any() else 0.0)
    return np.array(out, dtype=np.float64)


def evaluate(log_prob_fn, verbose: bool = True) -> dict:
    """Compute TLSE, BodyNLL, SCORE for a model exposed as log_prob_fn."""
    gt = load_ground_truth()
    tags = gt["tags"]; surv_sim = gt["surv"]; is_tail = gt["is_tail"]
    thetas = gt["thetas"]; zs = gt["zs"]

    tlse_terms, body_nlls, rows = [], [], []
    for i, tag in enumerate(tags):
        z = float(zs[i]); theta = thetas[i]
        # body NLL
        body = gt[f"body_{tag}"]
        nll = float("nan")
        if body.size:
            lp = np.asarray(log_prob_fn(body, z, tuple(float(x) for x in theta)), dtype=np.float64)
            nll = float(-np.mean(lp))
            body_nlls.append(nll)
        # tail log-survival error (tail points only)
        if bool(is_tail[i]):
            s_nsf = _survival_nsf(log_prob_fn, z, theta)
            s_sim = surv_sim[i]
            for j, _t in enumerate(THRESHOLDS):
                if s_sim[j] <= 0:   # not measurable from sim
                    continue
                err = abs(np.log10(max(s_nsf[j], EPS)) - np.log10(max(s_sim[j], EPS)))
                tlse_terms.append(err)
            rows.append((tag, s_sim, s_nsf, nll))

    tlse = float(np.mean(tlse_terms)) if tlse_terms else float("nan")
    body_nll = float(np.mean(body_nlls)) if body_nlls else float("nan")
    score = tlse + 0.3 * max(0.0, body_nll - BODY_BUDGET)

    if verbose:
        print("\n--- per-point tail survival (sim vs nsf) ---")
        print(f"{'point':16s} | {'mu>t':>5} | {'S_sim':>9} {'S_nsf':>9} | bodyNLL")
        for tag, s_sim, s_nsf, nll in rows:
            for j, t in enumerate(THRESHOLDS):
                lead = tag if j == 0 else ""
                nlls = f"{nll:7.3f}" if j == 0 else ""
                print(f"{lead:16s} | {t:5.1f} | {s_sim[j]:9.5f} {s_nsf[j]:9.5f} | {nlls}")
        print("---")
    return dict(SCORE=score, TLSE=tlse, BodyNLL=body_nll)


if __name__ == "__main__":
    build_ground_truth(force="--force" in sys.argv)
