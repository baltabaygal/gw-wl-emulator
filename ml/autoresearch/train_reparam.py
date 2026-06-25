"""Data-side tail reparametrization (smooth, no splice).

Model the flow in v = g(y) where y = standardized lnmu and g smoothly compresses the
heavy upper tail into a near-Gaussian shape:
    g(y) = y                                   for y <= t0
    g(y) = t0 + (2/k)(sqrt(1+k(y-t0)) - 1)     for y > t0   (C1 at t0)
This turns the exponential lnmu tail into a Gaussian one, so a plain Normal-base NSF
fits v (incl. the tail) easily; g^-1 restores the heavy tail. One continuous curve,
no handoff/discontinuity, no spline forced to manufacture the tail.

p(lnmu) = p_flow(g(y)) * |g'(y)| / lnmu_std ;   log|g'(y)| = -0.5 log(1+k(y-t0)) for y>t0.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from ml.autoresearch import train_ar, prepare_ar

AR = Path(__file__).resolve().parent
MODEL_PATH = AR / "models" / "flow_reparam.pt"

CFG = dict(bins=20, bound=16.0, transforms=6, hidden=128, base="normal",
           lr=5e-4, weight_decay=1e-5, batch_size=32768, epochs=int(__import__("os").environ.get("EPOCHS","50")), patience=40,
           grad_clip=5.0, device="mps", seed=0, n_subsample=600000,
           muc=2.0, k=1.0,            # tail-compression: start at mu=muc, strength k
           df=4.0, tail_weight=0.0, tail_tilt=0.0, smooth_weight=0.0)  # (unused knobs for build_flow)


def g_and_logdet(y, t0, k):
    """v = g(y), and log|dg/dy|. Safe for autograd (relu guards the sqrt)."""
    over = torch.relu(y - t0)
    s = torch.sqrt(1.0 + k * over)
    v = torch.where(y <= t0, y, t0 + (2.0 / k) * (s - 1.0))
    logdet = torch.where(y <= t0, torch.zeros_like(y), -0.5 * torch.log(1.0 + k * over))
    return v, logdet


def main():
    cfg = CFG
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = torch.device(cfg["device"])
    (Xtr, Ytr), (Xva, Yva), stats = train_ar.load_standardized(cfg)
    if cfg["n_subsample"] < Xtr.shape[0]:
        sel = torch.from_numpy(np.random.default_rng(cfg["seed"]).choice(Xtr.shape[0], cfg["n_subsample"], replace=False))
        Xtr, Ytr = Xtr[sel], Ytr[sel]
    lmean = float(stats["lnmu_mean"]); lstd = float(stats["lnmu_std"])
    t0 = (np.log(cfg["muc"]) - lmean) / lstd
    cfg["t0"] = float(t0)
    # subsample validation: 4.3M rows/epoch is the dominant per-epoch cost (and memory)
    nval = min(100000, Xva.shape[0])
    vsel = torch.from_numpy(np.random.default_rng(1).choice(Xva.shape[0], nval, replace=False))
    Xva, Yva = Xva[vsel], Yva[vsel]
    Xtr, Ytr, Xva, Yva = Xtr.to(dev), Ytr.to(dev), Xva.to(dev), Yva.to(dev)

    flow = train_ar.build_flow(cfg).to(dev)
    if MODEL_PATH.exists():   # resume: accumulate progress across kills/stalls
        try:
            prev = torch.load(MODEL_PATH, map_location=dev, weights_only=False)
            flow.load_state_dict(prev["state_dict"]); print("resumed from existing checkpoint", flush=True)
        except Exception as e:
            print("resume failed, fresh init:", e, flush=True)
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = Xtr.shape[0]; bs = cfg["batch_size"]
    best = float("inf"); best_state = None; bad = 0; t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        flow.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i+bs]; yb = Ytr[idx].squeeze(-1)
            v, ld = g_and_logdet(yb, t0, cfg["k"])
            opt.zero_grad()
            lp = flow(Xtr[idx]).log_prob(v.unsqueeze(-1)) + ld     # density in y-space
            loss = -lp.mean()
            if not torch.isfinite(loss): raise ValueError("nan loss")
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), cfg["grad_clip"]); opt.step()
        flow.eval()
        with torch.no_grad():
            vv, ldv = g_and_logdet(Yva.squeeze(-1), t0, cfg["k"])
            val = float(-(flow(Xva).log_prob(vv.unsqueeze(-1)) + ldv).mean())
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d} | val NLL {val:.5f} | {time.time()-t_start:.0f}s")
        if val < best - 1e-5:
            best = val; best_state = {k_: v_.cpu().clone() for k_, v_ in flow.state_dict().items()}; bad = 0
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": best_state, "config": cfg, "stats": stats}, MODEL_PATH)
        else:
            bad += 1
            if bad >= cfg["patience"]: print(f"early stop at {epoch}"); break
    flow.load_state_dict(best_state)

    # model-agnostic log_prob_fn (applies g + jacobian)
    cmean = torch.tensor(stats["context_mean"], dtype=torch.float32, device=dev)
    cstd = torch.tensor(stats["context_std"], dtype=torch.float32, device=dev)
    def log_prob_fn(lnmu, z, theta):
        lnmu = np.asarray(lnmu, np.float32).reshape(-1, 1)
        y = (torch.from_numpy(lnmu).to(dev) - lmean) / lstd
        v, ld = g_and_logdet(y.squeeze(-1), t0, cfg["k"])
        ctx = torch.tensor([[z, *theta]], dtype=torch.float32, device=dev).repeat(y.shape[0], 1)
        with torch.no_grad():
            lp = flow((ctx - cmean) / cstd).log_prob(v.unsqueeze(-1)) + ld - np.log(lstd)
        return lp.cpu().numpy().ravel()

    m = prepare_ar.evaluate(log_prob_fn, verbose=True)
    print("\n=== RESULT (reparam k=%.2f muc=%.1f) ===" % (cfg["k"], cfg["muc"]))
    for kk in ("SCORE", "TailShape", "Rough", "TLSE", "BodyNLL"): print(f"{kk}: {m[kk]:.4f}")
    print("ValNLL:", round(best, 4))


if __name__ == "__main__":
    main()
