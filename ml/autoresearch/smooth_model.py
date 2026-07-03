"""Production smooth density: body flow + C-inf tanh blend to an analytic power law,
with a physical low-mu cutoff at the empty-beam demagnification edge.

p(lnmu) = W(lnmu) * [ (1-w)*p_flow + w*p_tail ],  renormalized on a wide grid, where
  w = 0.5*(1 + tanh((lnmu - ln muc)/width))       -- tail blend
  p_tail(lnmu) = p_flow(ln muc) * exp(-(alpha-1)(lnmu - ln muc))  (density-anchored)
  W = sigmoid((lnmu - edge(z,theta))/EDGE_W)      -- low-mu cutoff window (exp19):
      the simulator PDF ends at the empty-beam bound; the flow leaks below it.
      edge(z,theta) = ridge fit of the per-config 0.1% lnmu quantile of
      datasets_logz_1k (resid std 0.008; see prepare_fix.py), minus EDGE_MARGIN.

Smooth by construction: no splice kink, no spline-bound edge (blend takes over
before mu ~ 13.35), C-inf cutoff, conditional on (z, theta) for free.

Usage:
    from ml.autoresearch.smooth_model import load_smooth_fn
    fn = load_smooth_fn()            # log p(lnmu | z, theta), raw lnmu space
    lp = fn(lnmu_array, z, (h, om, s8))
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path
import numpy as np
import torch

from ml.autoresearch import train_ar

AR = Path(__file__).resolve().parent
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# exp20 body (low-z augmented; exp18 recipe). Alternatives kept: flow_smooth.pt
# (exp18), flow_smooth_zmask.pt (exp21: z=1 shoulder ~25% better, high-z tail worse).
DEFAULT_MODEL = AR / "models" / "flow_smooth_lowz.pt"
EDGE_FIT = json.loads((AR / "cache" / "edge_alpha_fit.json").read_text())
EDGE_MARGIN = 0.05   # place the window this far below the fitted 0.1% quantile
EDGE_W = 0.02        # window softness in lnmu (sharp but C-inf)


def edge_lnmu(z, h, om, s8):
    """Fitted physical lower edge of the lnmu distribution (see prepare_fix.py)."""
    lz = np.log1p(z)
    feats = np.array([1.0, lz, lz**2, lz**3, om*lz, h*lz, s8*lz, om, h, s8])
    return float(feats @ np.array(EDGE_FIT["edge_coef"]))
# image-plane mu^-2 tail. muc=8: keep the flow's learned (accurate) tail through the
# data-rich range, hand over to the analytic power law before the spline-bound kink at
# mu=13.35 and before the StudentT base's too-shallow (lnmu-power) asymptotics.
# Scanned muc=3 (SCORE 0.238, overshoots: anchored on the shoulder), 8/10/12 (all ~0.106);
# muc=8 best TLSE (0.192) and DenseRough (3.79). See REPORT.md 2026-07-02.
BLEND = dict(alpha=2.0, muc=8.0, width=0.35)


def load_body_fn(path=DEFAULT_MODEL, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    flow = train_ar.build_flow(ck["config"])
    flow.load_state_dict(ck["state_dict"]); flow.eval()
    return train_ar.make_log_prob_fn(flow, ck["stats"], torch.device(device)), ck


def make_blend_log_prob_fn(body_fn, alpha=None, muc=None, width=None,
                           grid_lo=-4.0, grid_hi=np.log(400.0), grid_n=3000):
    alpha = BLEND["alpha"] if alpha is None else alpha
    muc = BLEND["muc"] if muc is None else muc
    width = BLEND["width"] if width is None else width
    lnc = float(np.log(muc)); k = float(alpha - 1.0)
    gb = np.linspace(grid_lo, grid_hi, grid_n)
    wg = 0.5 * (1.0 + np.tanh((gb - lnc) / width))

    def fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, dtype=np.float64).ravel()
        h, om, s8 = (float(x) for x in theta)
        e = edge_lnmu(z, h, om, s8) - EDGE_MARGIN
        lp_c = float(np.asarray(body_fn(np.array([lnc]), z, theta), np.float64)[0])
        # normalization of the cut, blended density on the wide grid (per context)
        bdg = np.exp(np.asarray(body_fn(gb, z, theta), np.float64))
        plg = np.exp(lp_c - k * (gb - lnc))
        Wg = 1.0 / (1.0 + np.exp(-(gb - e) / EDGE_W))
        norm = float(_trapz(Wg * ((1.0 - wg) * bdg + wg * plg), gb))
        bd = np.exp(np.asarray(body_fn(lnmu, z, theta), np.float64))
        pl = np.exp(lp_c - k * (lnmu - lnc))
        w = 0.5 * (1.0 + np.tanh((lnmu - lnc) / width))
        W = 1.0 / (1.0 + np.exp(-(lnmu - e) / EDGE_W))
        p = W * ((1.0 - w) * bd + w * pl)
        return np.log(np.maximum(p / norm, 1e-300))

    return fn


# ---------------------------------------------------------------------------
# exp23: conditional POT tail (see RESEARCH_tails.md round 2, fit_tail_amplitude.py)
# The flow's tail AMPLITUDE is uncalibrated per context (measured: ~constant vs a
# x140 true range). Replace the density-anchored power law with a survival model
# built from Poisson-regressed exceedance fractions S3(ctx)=P(mu>3), S8(ctx)=P(mu>8):
#   ln S(x) = ln S3 - k (x - ln3) + (k-1) w sp((x - ln8)/w),  x = lnmu,
#   k = ln(S3/S8)/ln(8/3) (transition exponent), sp = softplus, w = 0.3
# -> survival slope k over mu in [3,8], smoothly -> 1 (i.e. dP/dmu ~ mu^-2) beyond.
#   p_tail(x) = [k - (k-1) sigmoid((x - ln8)/w)] * S(x)   (positive, C-inf)
# Tanh-blended with the flow body at mu_c=3.5; edge window as before.
# ---------------------------------------------------------------------------
TAIL_FIT = json.loads((AR / "cache" / "tail_amp_fit.json").read_text())
# exp24: three anchors S2/S3/S8 (u=2 extends the calibrated tail over the z~1
# shoulder); piecewise power-law survival with softplus-smoothed corners, anchored
# exactly at S3 (best-fit threshold; "u2" anchor tested and dominated), asymptotically
# dP/dmu ~ mu^-2. muc scan (SCORE/extMAE/z=1): 1.5 .059/.071/.080, 1.6 .054/.064/.082,
# 1.7 .051/.061/.095 (SHIPPED), 1.8 .048/.064/.116, 2.0 .043/.074/.155, 2.4 .045/.097/.219.
POT = dict(u2=2.0, u3=3.0, u8=8.0, muc=1.7, width=0.30, wsp=0.25,
           k_min=1.0, k_max=8.0, anchor="u3")


def _tail_features(z, h, om, s8):
    lz = np.log1p(z)
    return np.array([1.0, lz, lz**2, lz**3, om, h, s8, om*lz, h*lz, s8*lz,
                     s8**2, om*s8, s8**2*lz, om*s8*lz])


def _sp(t):
    return np.where(t > 30, t, np.log1p(np.exp(np.minimum(t, 30.0))))


def pot_tail_logp(lnmu, z, h, om, s8):
    """log p_tail(lnmu): calibrated POT tail carrying survival mass S2 above u2.
    ln S is piecewise-linear in lnmu (slopes k1 over [2,3], k2 over [3,8], 1 beyond)
    with softplus-smoothed corners, anchored exactly at u2."""
    f = _tail_features(z, h, om, s8)
    lnS2 = float(f @ np.array(TAIL_FIT["coef"]["2.0"]))
    lnS3 = float(f @ np.array(TAIL_FIT["coef"]["3.0"]))
    lnS8 = float(f @ np.array(TAIL_FIT["coef"]["8.0"]))
    x2, x3, x8 = np.log(POT["u2"]), np.log(POT["u3"]), np.log(POT["u8"])
    w = POT["wsp"]
    k1 = np.clip((lnS2 - lnS3) / (x3 - x2), POT["k_min"], POT["k_max"])
    k2 = np.clip((lnS3 - lnS8) / (x8 - x3), POT["k_min"], POT["k_max"])
    t3 = (lnmu - x3) / w; t8 = (lnmu - x8) / w
    if POT.get("anchor", "u3") == "u3":     # anchor exactly at S3 (best-fit threshold)
        xa, lnSa = x3, lnS3
    else:                                    # anchor exactly at S2
        xa, lnSa = x2, lnS2
    lnS = (lnSa - k1 * (lnmu - xa)
           + (k1 - k2) * w * (_sp(t3) - _sp((xa - x3) / w))
           + (k2 - 1.0) * w * (_sp(t8) - _sp((xa - x8) / w)))
    slope = (k1 - (k1 - k2) / (1.0 + np.exp(-t3))
                - (k2 - 1.0) / (1.0 + np.exp(-t8)))
    return np.log(np.maximum(slope, 1e-10)) + lnS


def make_pot_log_prob_fn(body_fn, grid_lo=-4.0, grid_hi=np.log(400.0), grid_n=3000):
    lnc = float(np.log(POT["muc"])); width = POT["width"]
    gb = np.linspace(grid_lo, grid_hi, grid_n)
    wg = 0.5 * (1.0 + np.tanh((gb - lnc) / width))

    def fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, dtype=np.float64).ravel()
        h, om, s8 = (float(x) for x in theta)
        e = edge_lnmu(z, h, om, s8) - EDGE_MARGIN
        bdg = np.exp(np.asarray(body_fn(gb, z, theta), np.float64))
        plg = np.exp(pot_tail_logp(gb, z, h, om, s8))
        Wg = 1.0 / (1.0 + np.exp(-(gb - e) / EDGE_W))
        norm = float(_trapz(Wg * ((1.0 - wg) * bdg + wg * plg), gb))
        bd = np.exp(np.asarray(body_fn(lnmu, z, theta), np.float64))
        pl = np.exp(pot_tail_logp(lnmu, z, h, om, s8))
        w = 0.5 * (1.0 + np.tanh((lnmu - lnc) / width))
        W = 1.0 / (1.0 + np.exp(-(lnmu - e) / EDGE_W))
        p = W * ((1.0 - w) * bd + w * pl)
        return np.log(np.maximum(p / norm, 1e-300))

    return fn


def load_smooth_fn(path=DEFAULT_MODEL, device="cpu", tail="pot", **blend_kw):
    body_fn, _ = load_body_fn(path, device)
    if tail == "pot":
        return make_pot_log_prob_fn(body_fn, **blend_kw)
    return make_blend_log_prob_fn(body_fn, **blend_kw)   # legacy exp18-22 tail
