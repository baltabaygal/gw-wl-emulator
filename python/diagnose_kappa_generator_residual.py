import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose residual structure in additive-field kappa generator validation."
    )
    parser.add_argument("npz", help="NPZ file written by additive_kappa_generator_validation.py")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    path = Path(args.npz)
    data = np.load(path)

    q = np.asarray(data["q"], dtype=float)
    lam_emp = np.asarray(data["lam_emp"], dtype=np.complex128)
    lam_theory = np.asarray(data["lam_theory"], dtype=np.complex128)
    mask = np.asarray(data["reliable_mask"], dtype=bool)

    delta = lam_emp - lam_theory
    q_abs = np.abs(q)
    delta_abs = np.abs(delta)

    reliable = mask & np.isfinite(delta_abs)
    positive_q = reliable & (q_abs > 0)

    if not np.any(positive_q):
        raise RuntimeError("No reliable nonzero q modes found")

    q_rel = q_abs[positive_q]
    d_rel = delta_abs[positive_q]
    order = np.argsort(q_rel)
    q_rel = q_rel[order]
    d_rel = d_rel[order]

    n = q_rel.size
    split = max(1, n // 3)
    low = d_rel[:split]
    high = d_rel[-split:]

    slope = np.polyfit(q_rel, d_rel, deg=1)[0]
    corr = np.corrcoef(q_rel, d_rel)[0, 1] if n > 1 else np.nan

    summary = {
        "input_npz": str(path),
        "reliable_nonzero_modes": int(n),
        "median_abs_delta_low_q": float(np.median(low)),
        "median_abs_delta_high_q": float(np.median(high)),
        "high_to_low_median_ratio": float(np.median(high) / np.median(low)) if np.median(low) > 0 else None,
        "abs_delta_vs_abs_q_slope": float(slope),
        "abs_delta_vs_abs_q_corr": float(corr),
        "max_abs_delta": float(np.max(d_rel)),
        "q_at_max_abs_delta": float(q_rel[np.argmax(d_rel)]),
    }

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = str(path.with_suffix("")) + "_residual"

    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True)

    ax[0].plot(q[reliable], delta.real[reliable], lw=1.8, label="Re deltaLambda")
    ax[0].plot(q[reliable], delta.imag[reliable], lw=1.8, label="Im deltaLambda")
    ax[0].axhline(0.0, color="k", ls="--", lw=0.9)
    ax[0].set_ylabel("deltaLambda")
    ax[0].set_title("Residual structure in additive kappa generator test")
    ax[0].legend(frameon=False)

    ax[1].plot(q_rel, d_rel, "o-", ms=3.0, lw=1.4)
    ax[1].set_xlabel("|q_kappa|")
    ax[1].set_ylabel("|deltaLambda|")
    ax[1].set_title(
        f"high/low median={summary['high_to_low_median_ratio']:.3f}, "
        f"corr={summary['abs_delta_vs_abs_q_corr']:.3f}"
    )

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
