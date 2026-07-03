"""Head-to-head: flow_ar vs flow_smooth, each alone and with the tanh tail blend.

Reports the frozen harness metrics (prepare_ar.evaluate) plus DenseRough:
mean |2nd diff of log p| on a 2000-pt grid over mu in [1, 100] (step-normalized
curvature proxy, catches knot wiggles the coarse harness bins miss), averaged
over the 8 tail panel points. Writes smoothness_compare.png for the winner vs
flow_ar (PDF + local slope panels).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from ml.autoresearch import prepare_ar, smooth_model

AR = Path(__file__).resolve().parent
PANEL = [(z, th) for th in (prepare_ar.CENTRAL, prepare_ar.STRESS) for z in prepare_ar.TAIL_ZS]
LG = np.linspace(np.log(1.0), np.log(100.0), 2000)


def dense_rough(fn):
    r = []
    for z, th in PANEL:
        lp = np.asarray(fn(LG, z, th), np.float64)
        r.append(np.mean(np.abs(np.diff(lp, 2))) / (LG[1] - LG[0]) ** 2)  # ~|d2 logp/dlnmu^2|
    return float(np.mean(r))


def main():
    cands = {}
    body_ar, _ = smooth_model.load_body_fn(AR / "models" / "flow_ar.pt")
    cands["flow_ar"] = body_ar
    cands["flow_ar+blend"] = smooth_model.make_blend_log_prob_fn(body_ar)
    body_sm, ck = smooth_model.load_body_fn(AR / "models" / "flow_smooth.pt")
    cands["flow_smooth"] = body_sm
    cands["flow_smooth+blend"] = smooth_model.make_blend_log_prob_fn(body_sm)

    print(f"{'candidate':20s} {'SCORE':>7} {'TailShape':>9} {'Rough':>7} {'TLSE':>7} "
          f"{'BodyNLL':>8} {'DenseRough':>10}")
    results = {}
    for name, fn in cands.items():
        m = prepare_ar.evaluate(fn, verbose=False)
        m["DenseRough"] = dense_rough(fn)
        results[name] = m
        print(f"{name:20s} {m['SCORE']:7.4f} {m['TailShape']:9.4f} {m['Rough']:7.4f} "
              f"{m['TLSE']:7.4f} {m['BodyNLL']:8.4f} {m['DenseRough']:10.3f}", flush=True)

    # figure: old production vs new blend, PDF + local slope
    ref = np.load(AR / "cache" / "clean_tail_ref.npz")
    edges = ref["edges"]; ctr = np.sqrt(edges[:-1] * edges[1:]); bw = np.diff(edges)
    ZS = [2.0, 3.5, 5.0, 8.0]; THETA = (0.72, 0.38, 1.00)
    lg = np.linspace(np.log(0.5), np.log(300), 2000); mug = np.exp(lg)
    show = [("flow_ar (old)", cands["flow_ar"], "#2ca02c"),
            ("flow_smooth+blend (new)", cands["flow_smooth+blend"], "#d62728")]
    fig, ax = plt.subplots(2, 4, figsize=(19, 8), sharex="col")
    for i, z in enumerate(ZS):
        a, b = ax[0][i], ax[1][i]
        c = ref[f"c_{z}"]; N = int(ref[f"N_{z}"]); m = c >= 20
        a.scatter(ctr[m], c[m] / (N * bw[m]), s=12, color="#1f77b4", zorder=3, label="simulator 10M")
        for lbl, fn, col in show:
            lp = np.asarray(fn(lg, z, THETA), np.float64)
            a.plot(mug, np.exp(lp) / mug, color=col, lw=1.4, label=lbl)
            b.plot(mug, np.gradient(lp, lg) - 1.0, color=col, lw=1.2)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(1e-7, 5)
        a.set_title(f"z={z}"); a.grid(True, which="both", ls=":", alpha=.35)
        if i == 0:
            a.set_ylabel(r"$dP/d\mu$"); a.legend(fontsize=8, loc="lower left")
        b.axhline(-2.0, color="k", ls="--", lw=1, alpha=.6)
        b.set_xscale("log"); b.set_ylim(-8, 4); b.set_xlabel(r"$\mu$")
        b.grid(True, which="both", ls=":", alpha=.35)
        if i == 0:
            b.set_ylabel("local log-log slope")
    fig.suptitle("smoothness fix: old flow_ar vs new flow_smooth + tanh blend "
                 f"(body 3x10 knots, curvature penalty; blend at mu={smooth_model.BLEND['muc']})")
    fig.tight_layout()
    out = AR / "smoothness_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
