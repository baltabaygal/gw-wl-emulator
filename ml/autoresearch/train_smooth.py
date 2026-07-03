"""Smoothness experiment: SIMPLER spline body + dense curvature penalty.

Motivation (2026-07-02, see smoothness_refresh.png): flow_ar.pt has knot-scale
wiggles in the local log-log slope over mu ~ 1-13 (6 transforms x 20 RQS bins
overfit MC noise) and a kink at the spline bound edge mu=13.35. This trainer:
  - transforms 6->3, bins 20->10 (4x fewer knots),
  - curvature penalty: mean (2nd diff of log p)^2 on a dense grid over
    lnmu in [0.35, 2.6] (mu ~ 1.4-13.5, the wiggly region; peak left free),
  - NO tail slope pinning (fixed the z=2 overshoot flow_ar had),
  - StudentT df=2 base kept for tail mass.
Production tail is handled by the smooth tanh blend (see smooth_model.py).
Saves models/flow_smooth.pt. Fork of train_ar.py (hackable-trainer pattern).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ml.autoresearch import train_ar, prepare_ar

# ============================= CONFIG (edit me) =============================
CONFIG = dict(
    # --- architecture (simpler than flow_ar: fewer knots = smoother) ---
    bins=10,
    bound=16.0,
    transforms=3,
    hidden=128,
    base="studentt",
    df=2.0,
    # --- optimization ---
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=65536,
    epochs=60,
    patience=12,
    grad_clip=5.0,
    # --- disabled flow_ar extras (kept for build_flow compat) ---
    tail_weight=0.0, tail_thresh=1.5, tail_tilt=0.0,
    smooth_weight=0.0, slope_alpha=2.0,
    smooth_lo=4.0, smooth_hi=12.0, smooth_npts=16, smooth_ctx=128,
    # --- roughness penalty ---
    # exp18 (WINNER recipe): order 2, weight 30, grid mu ~ 1.4-13.5.
    # exp19 (order 3, weight 60, wider grid) was WORSE everywhere incl. low z ->
    # the low-z shoulder overshoot is data sparsity, not penalty bias (see exp20).
    curv_order=2,
    curv_weight=30.0,    # weight on mean (d^k log p)^2 over the curv grid
    curv_lo=0.35,        # grid lower edge, raw lnmu (mu ~ 1.42)
    curv_hi=2.60,        # grid upper edge, raw lnmu (mu ~ 13.5)
    curv_npts=48,
    curv_ctx=128,        # contexts per batch used for the penalty
    # --- exp20: low-z data augmentation (cache/lowz_aug.npz from gen_lowz_data.py) ---
    aug_path="cache/lowz_aug.npz",   # "" disables
    aug_n=1200000,                   # subsample of the augmentation set to append
    # --- exp21/22: down-weight the penalty at low z. At low z the conditional is
    # narrow and data-free on the penalty grid, so the penalty (not data) was setting
    # the shoulder shape -> z~1 overshoot. exp21 hard mask z>1.6 fixed z=1 but hurt
    # high z; exp22 uses a smooth per-context ramp sigmoid((z-curv_zmin)/curv_zw). ---
    curv_zmin=1.6,
    curv_zw=0.5,
    # --- runtime ---
    device="mps",
    seed=0,
    n_subsample=1500000,
)
AR_DIR = Path(__file__).resolve().parent
MODEL_PATH = AR_DIR / "models" / "flow_smooth_zramp.pt"  # exp22; exp18 winner kept at flow_smooth.pt
STATS_PATH = AR_DIR / "cache" / "stats_smooth.json"
# ===========================================================================


def train(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(cfg["device"])
    train_ar.STATS_PATH = STATS_PATH  # keep flow_ar's stats file untouched
    (Xtr, Ytr), (Xva, Yva), stats = train_ar.load_standardized(cfg)
    if cfg.get("n_subsample", 0) and cfg["n_subsample"] < Xtr.shape[0]:
        sel = torch.from_numpy(np.random.default_rng(cfg["seed"]).choice(
            Xtr.shape[0], cfg["n_subsample"], replace=False))
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    if cfg.get("aug_path"):
        d = np.load(AR_DIR / cfg["aug_path"])
        Xa, Ya = d["X"].astype(np.float64), d["Y"].astype(np.float64)
        if cfg.get("aug_n", 0) and cfg["aug_n"] < Xa.shape[0]:
            ai = np.random.default_rng(cfg["seed"] + 1).choice(Xa.shape[0], cfg["aug_n"], replace=False)
            Xa, Ya = Xa[ai], Ya[ai]
        cm = np.array(stats["context_mean"]); cs = np.array(stats["context_std"])
        Xa = torch.from_numpy(((Xa - cm) / cs).astype(np.float32))
        Ya = torch.from_numpy(((Ya - stats["lnmu_mean"]) / stats["lnmu_std"])
                              .astype(np.float32)).reshape(-1, 1)
        Xtr = torch.cat([Xtr, Xa]); Ytr = torch.cat([Ytr, Ya])
        print(f"augmented train set: +{Xa.shape[0]} low-z samples -> {Xtr.shape[0]} total")
    if Xva.shape[0] > 100000:
        vs = torch.from_numpy(np.random.default_rng(1).choice(Xva.shape[0], 100000, replace=False))
        Xva, Yva = Xva[vs], Yva[vs]
    Xtr, Ytr, Xva, Yva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev), Yva.to(dev)
    lmean = stats["lnmu_mean"]; lstd = stats["lnmu_std"]

    flow = train_ar.build_flow(cfg).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = Xtr.shape[0]; bs = cfg["batch_size"]

    # dense curvature grid in standardized lnmu
    cgrid = torch.linspace((cfg["curv_lo"] - lmean) / lstd,
                           (cfg["curv_hi"] - lmean) / lstd,
                           cfg["curv_npts"], device=dev).reshape(-1, 1)

    best_val = float("inf"); best_state = None; no_improve = 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        flow.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            loss = -flow(xb).log_prob(yb).mean()
            if cfg["curv_weight"] > 0:
                cs = xb[:cfg["curv_ctx"]]
                G = cgrid.shape[0]; Nc = cs.shape[0]
                gx = cgrid.repeat(Nc, 1)
                cx = cs.repeat_interleave(G, dim=0)
                lg = flow(cx).log_prob(gx).reshape(Nc, G)
                d = lg
                for _ in range(cfg.get("curv_order", 2)):
                    d = d[:, 1:] - d[:, :-1]
                pen = (d ** 2).mean(dim=1)                       # per context
                if cfg.get("curv_zmin", 0) > 0:
                    zraw = cs[:, 0] * stats["context_std"][0] + stats["context_mean"][0]
                    wz = torch.sigmoid((zraw - cfg["curv_zmin"]) / cfg.get("curv_zw", 0.5))
                    pen = (wz * pen).sum() / (wz.sum() + 1e-12)
                else:
                    pen = pen.mean()
                loss = loss + cfg["curv_weight"] * pen
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(flow.parameters(), cfg["grad_clip"])
            opt.step()

        flow.eval()
        with torch.no_grad():
            vlp = []
            for i in range(0, Xva.shape[0], 65536):
                vlp.append(flow(Xva[i:i+65536]).log_prob(Yva[i:i+65536]))
            val = float(-torch.cat(vlp).mean())
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:3d} | val NLL {val:.5f} | {time.time()-t0:.0f}s", flush=True)
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            no_improve = 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "config": cfg, "stats": stats, "epoch": epoch}, MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"early stop at epoch {epoch}", flush=True); break

    if best_state is not None:
        flow.load_state_dict(best_state)
    torch.save({"state_dict": flow.state_dict(), "config": cfg, "stats": stats}, MODEL_PATH)
    return flow, stats, best_val


def main():
    cfg = CONFIG
    t0 = time.time()
    flow, stats, val_nll = train(cfg)
    fn = train_ar.make_log_prob_fn(flow, stats, torch.device(cfg["device"]))
    m = prepare_ar.evaluate(fn, verbose=True)
    print("\n=== RESULT (flow_smooth alone, no tail blend) ===")
    for k in ("SCORE", "TailShape", "Rough", "TLSE", "BodyNLL"):
        print(f"{k}: {m[k]:.4f}")
    print(f"ValNLL:  {val_nll:.4f}")
    print(f"config:  {cfg}")
    print(f"wall:    {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
