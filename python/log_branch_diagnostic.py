import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npz(npz_path: Path):
    data = np.load(npz_path)
    summary_path = npz_path.with_name(npz_path.stem + "_summary.json")
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return data, summary


def first_winding_location(q, phi):
    order = np.argsort(q)
    q_sorted = q[order]
    phase_wrapped = np.angle(phi[order])
    dphi = np.diff(phase_wrapped)
    crossings = np.where(np.abs(dphi) > np.pi)[0]
    if crossings.size == 0:
        return None
    idx = crossings[0] + 1
    return float(abs(q_sorted[idx]))


def lambda_unwrapped(phi, lam_theory, z):
    n = phi.size
    q_index = np.arange(n)
    center = n // 2

    phase = np.angle(phi)
    phase_unwrapped = phase.copy()
    theory_phase = np.imag(lam_theory) * z

    pos = q_index >= center
    neg = q_index <= center

    phase_unwrapped[pos] = np.unwrap(phase[pos])
    phase_unwrapped[neg] = np.unwrap(phase[neg][::-1])[::-1]

    def align_branch(mask):
        idx = np.where(mask)[0]
        if idx.size <= 1:
            return
        anchor = idx[: min(16, idx.size)]
        d = np.median((phase_unwrapped[anchor] - theory_phase[anchor]) / (2.0 * np.pi))
        shift = np.round(d)
        phase_unwrapped[idx] -= shift * 2.0 * np.pi

    align_branch((q_index >= center) & (q_index <= center + 32))
    align_branch((q_index <= center) & (q_index >= center - 32))

    return (np.log(np.abs(phi)) + 1j * phase_unwrapped) / z


def band_stats(abs_residual, mask):
    if np.count_nonzero(mask) == 0:
        return {"n_modes": 0, "median_residual": None}
    return {
        "n_modes": int(np.count_nonzero(mask)),
        "median_residual": float(np.median(abs_residual[mask])),
        "mean_residual": float(np.mean(abs_residual[mask])),
        "max_residual": float(np.max(abs_residual[mask])),
    }


def mc_floor_from_pair(res100, res400, mask):
    if np.count_nonzero(mask) == 0:
        return {"mc_floor": None, "fixed_floor": None, "ratio_to_mc": None}
    d100 = float(np.median(res100[mask]))
    d400 = float(np.median(res400[mask]))
    mc2 = max((4.0 / 3.0) * max(d100 * d100 - d400 * d400, 0.0), 0.0)
    fixed2 = max(d400 * d400 - mc2 / 4.0, 0.0)
    mc = float(np.sqrt(mc2))
    fixed = float(np.sqrt(fixed2))
    ratio = float(d400 / mc) if mc > 0 else None
    return {
        "mc_floor": mc,
        "fixed_floor": fixed,
        "ratio_to_mc": ratio,
        "median_100k": d100,
        "median_400k": d400,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose log-branch winding and compare wrapped vs unwrapped transport residuals."
    )
    parser.add_argument("npz_400k", help="Primary widened-domain 400k NPZ")
    parser.add_argument("--npz-100k", required=True, help="Matched widened-domain 100k NPZ")
    parser.add_argument("--out-prefix", default="log_branch_diagnostic")
    args = parser.parse_args()

    data400, summary400 = load_npz(Path(args.npz_400k))
    data100, summary100 = load_npz(Path(args.npz_100k))
    if summary400 is None or summary100 is None:
        raise RuntimeError("Missing summary JSON for one of the NPZ inputs")

    q400 = np.asarray(data400["q"], dtype=float)
    q100 = np.asarray(data100["q"], dtype=float)
    if q400.shape != q100.shape or not np.allclose(q400, q100):
        raise RuntimeError("Input q grids do not match")
    q = q400

    phi400 = np.asarray(data400["phi_emp"], dtype=np.complex128)
    phi100 = np.asarray(data100["phi_emp"], dtype=np.complex128)
    lam_theory400 = np.asarray(data400["lam_theory"], dtype=np.complex128)
    lam_theory100 = np.asarray(data100["lam_theory"], dtype=np.complex128)
    lam_wrapped400 = np.asarray(data400["lam_emp"], dtype=np.complex128)
    lam_wrapped100 = np.asarray(data100["lam_emp"], dtype=np.complex128)

    z = float(summary400["z"])
    q_wind = first_winding_location(q, phi400)

    lam_unwrapped400 = lambda_unwrapped(phi400, lam_theory400, z)
    lam_unwrapped100 = lambda_unwrapped(phi100, lam_theory100, z)

    res_wrapped400 = np.abs(lam_wrapped400 - lam_theory400)
    res_wrapped100 = np.abs(lam_wrapped100 - lam_theory100)
    res_unwrapped400 = np.abs(lam_unwrapped400 - lam_theory400)
    res_unwrapped100 = np.abs(lam_unwrapped100 - lam_theory100)

    pre_mask = np.abs(q) < q_wind if q_wind is not None else np.ones_like(q, dtype=bool)
    full_mask = np.ones_like(q, dtype=bool)

    verdict = {
        "input_400k": args.npz_400k,
        "input_100k": args.npz_100k,
        "q_wind": q_wind,
        "pre_winding_band": {
            "wrapped": {
                **band_stats(res_wrapped400, pre_mask),
                **mc_floor_from_pair(res_wrapped100, res_wrapped400, pre_mask),
            },
            "unwrapped": {
                **band_stats(res_unwrapped400, pre_mask),
                **mc_floor_from_pair(res_unwrapped100, res_unwrapped400, pre_mask),
            },
        },
        "full_unwrapped_band": {
            **band_stats(res_unwrapped400, full_mask),
            **mc_floor_from_pair(res_unwrapped100, res_unwrapped400, full_mask),
        },
    }

    out_prefix = args.out_prefix
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(verdict, f, indent=2, sort_keys=True)

    order = np.argsort(q)
    q_sorted = q[order]
    wrapped_phase = np.angle(phi400[order])
    unwrapped_phase = np.unwrap(wrapped_phase)

    fig, ax = plt.subplots(2, 2, figsize=(10.5, 8.0))

    ax[0, 0].plot(phi400.real, phi400.imag, lw=1.3)
    ax[0, 0].scatter([phi400.real[0]], [phi400.imag[0]], s=18, color="tab:green", label="start")
    ax[0, 0].scatter([phi400.real[-1]], [phi400.imag[-1]], s=18, color="tab:red", label="end")
    ax[0, 0].set_xlabel("Re(P_hat)")
    ax[0, 0].set_ylabel("Im(P_hat)")
    ax[0, 0].set_title("Complex P_hat trajectory")
    ax[0, 0].legend(frameon=False, fontsize=8)

    ax[0, 1].plot(q_sorted, wrapped_phase, lw=1.4, label="wrapped")
    if q_wind is not None:
        ax[0, 1].axvline(q_wind, color="tab:red", ls="--", lw=1.0, label=f"q_wind={q_wind:.1f}")
        ax[0, 1].axvline(-q_wind, color="tab:red", ls="--", lw=1.0)
    ax[0, 1].set_xlabel("q")
    ax[0, 1].set_ylabel("arg(P_hat)")
    ax[0, 1].set_title("Wrapped phase")
    ax[0, 1].legend(frameon=False, fontsize=8)

    ax[1, 0].plot(q_sorted, unwrapped_phase, lw=1.4, label="unwrapped")
    if q_wind is not None:
        ax[1, 0].axvline(q_wind, color="tab:red", ls="--", lw=1.0)
        ax[1, 0].axvline(-q_wind, color="tab:red", ls="--", lw=1.0)
    ax[1, 0].set_xlabel("q")
    ax[1, 0].set_ylabel("unwrap(arg(P_hat))")
    ax[1, 0].set_title("Unwrapped phase")

    absq = np.abs(q)
    order_abs = np.argsort(absq)
    ax[1, 1].plot(absq[order_abs], res_wrapped400[order_abs], lw=1.3, label="wrapped residual")
    ax[1, 1].plot(absq[order_abs], res_unwrapped400[order_abs], lw=1.3, label="unwrapped residual")
    if q_wind is not None:
        ax[1, 1].axvline(q_wind, color="tab:red", ls="--", lw=1.0, label="q_wind")
    ax[1, 1].set_xlabel("|q|")
    ax[1, 1].set_ylabel("|Lambda_emp - Lambda_theory|")
    ax[1, 1].set_title("Residual before/after phase unwrapping")
    ax[1, 1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
