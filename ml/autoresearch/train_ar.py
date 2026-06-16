"""HACKABLE training script for the autoresearch loop.

Edit the CONFIG block (and anything below) to run one experiment. Trains a
conditional spline flow on datasets_logz_1k, saves a checkpoint under models/,
then prints the honest metric block from prepare_ar.evaluate:

    SCORE:   <float>
    TLSE:    <float>
    BodyNLL: <float>
    ValNLL:  <float>

Keep an experiment if SCORE drops; otherwise revert. See program.md.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, StudentT

from zuko.flows.autoregressive import MAF
from zuko.transforms import MonotonicRQSTransform
from zuko.lazy import UnconditionalDistribution

from ml.data import load_dataset, flatten_dataset
from ml.autoresearch import prepare_ar

# ============================= CONFIG (edit me) =============================
CONFIG = dict(
    # --- architecture ---
    bins=16,             # RQS bins
    bound=12.0,          # RQS spline domain [-bound, bound] in standardized lnmu
    transforms=6,        # number of autoregressive transforms
    hidden=128,          # hypernetwork hidden width (x2 layers)
    base="studentt",     # "normal" | "studentt"
    df=4.0,              # StudentT degrees of freedom (heavier tail = smaller df)
    # --- optimization ---
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=32768,
    epochs=120,
    patience=15,
    grad_clip=5.0,
    # --- tail-aware loss (0 disables) ---
    tail_weight=0.0,     # extra weight on samples with lnmu > tail_thresh
    tail_thresh=1.5,
    # --- runtime ---
    time_budget_s=360,   # wall-clock training cap (fixed budget, comparable runs)
    device="mps",
    seed=0,
)
DATASET_DIR = "datasets_logz_1k"
AR_DIR = Path(__file__).resolve().parent
STATS_PATH = AR_DIR / "cache" / "stats_logz.json"
MODEL_PATH = AR_DIR / "models" / "flow_ar.pt"
# ===========================================================================


def build_flow(cfg) -> nn.Module:
    flow = MAF(
        features=1, context=4, transforms=cfg["transforms"],
        hidden_features=[cfg["hidden"], cfg["hidden"]],
        univariate=partial(MonotonicRQSTransform, slope=1e-3, bound=cfg["bound"]),
        shapes=[(cfg["bins"],), (cfg["bins"],), (cfg["bins"] - 1,)],
    )
    if cfg["base"] == "studentt":
        flow.base = UnconditionalDistribution(
            lambda df, loc, scale: Independent(StudentT(df, loc, scale), 1),
            torch.tensor([float(cfg["df"])]), torch.zeros(1), torch.ones(1),
            buffer=True,
        )
    elif cfg["base"] != "normal":
        raise ValueError(f"unknown base {cfg['base']}")
    return flow


def load_standardized(cfg):
    ds = load_dataset(DATASET_DIR)
    Xtr, Ytr = flatten_dataset(ds["train"], normalize=False)
    Xva, Yva = (flatten_dataset(ds["validation"], normalize=False)
                if "validation" in ds else (Xtr, Ytr))
    cmean = Xtr.mean(0, dtype=np.float64); cstd = Xtr.std(0, dtype=np.float64)
    cstd[cstd == 0] = 1.0
    lmean = float(Ytr.mean(dtype=np.float64)); lstd = float(Ytr.std(dtype=np.float64)) or 1.0
    stats = dict(context_mean=cmean.tolist(), context_std=cstd.tolist(),
                 lnmu_mean=lmean, lnmu_std=lstd)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    def norm(X, Y):
        Xn = ((X.astype(np.float64) - cmean) / cstd).astype(np.float32)
        Yn = ((Y.astype(np.float64) - lmean) / lstd).astype(np.float32)
        return torch.from_numpy(Xn), torch.from_numpy(Yn)

    return norm(Xtr, Ytr), norm(Xva, Yva), stats


def train(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(cfg["device"])
    (Xtr, Ytr), (Xva, Yva), stats = load_standardized(cfg)
    Xtr, Ytr, Xva, Yva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev), Yva.to(dev)
    lstd = stats["lnmu_std"]; lmean = stats["lnmu_mean"]
    tail_thr_norm = (cfg["tail_thresh"] - lmean) / lstd

    flow = build_flow(cfg).to(dev)
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = Xtr.shape[0]; bs = cfg["batch_size"]
    best_val = float("inf"); best_state = None; no_improve = 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        flow.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            lp = flow(xb).log_prob(yb)
            if cfg["tail_weight"] > 0:
                w = 1.0 + cfg["tail_weight"] * (yb.squeeze(-1) > tail_thr_norm).float()
                loss = -(w * lp).sum() / w.sum()
            else:
                loss = -lp.mean()
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
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d} | val NLL {val:.5f} | {time.time()-t0:.0f}s")
        if val < best_val - 1e-5:
            best_val = val; best_state = {k: v.cpu().clone() for k, v in flow.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"early stop at epoch {epoch}"); break
        if time.time() - t0 > cfg["time_budget_s"]:
            print(f"time budget hit at epoch {epoch}"); break

    if best_state is not None:
        flow.load_state_dict(best_state)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": flow.state_dict(), "config": cfg, "stats": stats}, MODEL_PATH)
    return flow, stats, best_val


def make_log_prob_fn(flow, stats, dev):
    cmean = torch.tensor(stats["context_mean"], dtype=torch.float32, device=dev)
    cstd = torch.tensor(stats["context_std"], dtype=torch.float32, device=dev)
    lmean = float(stats["lnmu_mean"]); lstd = float(stats["lnmu_std"])
    flow.eval()

    def fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, dtype=np.float32).reshape(-1, 1)
        x = torch.from_numpy(lnmu).to(dev)
        ctx = torch.tensor([[z, *theta]], dtype=torch.float32, device=dev).repeat(x.shape[0], 1)
        with torch.no_grad():
            xn = (x - lmean) / lstd
            cn = (ctx - cmean) / cstd
            lp = flow(cn).log_prob(xn) - np.log(lstd)
        return lp.cpu().numpy().ravel()

    return fn


def main():
    cfg = CONFIG
    t0 = time.time()
    flow, stats, val_nll = train(cfg)
    fn = make_log_prob_fn(flow, stats, torch.device(cfg["device"]))
    m = prepare_ar.evaluate(fn, verbose=True)
    print("\n=== RESULT ===")
    print(f"SCORE:   {m['SCORE']:.4f}")
    print(f"TLSE:    {m['TLSE']:.4f}")
    print(f"BodyNLL: {m['BodyNLL']:.4f}")
    print(f"ValNLL:  {val_nll:.4f}")
    print(f"config:  {cfg}")
    print(f"wall:    {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
