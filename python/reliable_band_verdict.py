import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = [0.30, 0.20, 0.10, 0.05, 0.02]


def load_npz_with_summary(npz_path: Path):
    data = np.load(npz_path)
    summary_path = npz_path.with_name(npz_path.stem + "_summary.json")
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return data, summary, summary_path


def compute_stats(abs_residual, phi_abs, q, thresholds):
    rows = []
    for thr in thresholds:
        mask = phi_abs > thr
        if np.count_nonzero(mask) == 0:
            rows.append(
                {
                    "phi_threshold": thr,
                    "n_modes": 0,
                    "median_abs_error": None,
                    "mean_abs_error": None,
                    "max_abs_error": None,
                    "corr_absq_absres": None,
                }
            )
            continue

        corr = None
        if np.count_nonzero(mask) > 1:
            corr = float(np.corrcoef(np.abs(q[mask]), abs_residual[mask])[0, 1])
        rows.append(
            {
                "phi_threshold": thr,
                "n_modes": int(np.count_nonzero(mask)),
                "median_abs_error": float(np.median(abs_residual[mask])),
                "mean_abs_error": float(np.mean(abs_residual[mask])),
                "max_abs_error": float(np.max(abs_residual[mask])),
                "corr_absq_absres": corr,
            }
        )
    return rows


def summarize_noise_floor(npz_100k: Path, npz_400k: Path):
    data_100k, summary_100k, _ = load_npz_with_summary(npz_100k)
    data_400k, summary_400k, _ = load_npz_with_summary(npz_400k)

    if summary_100k is None or summary_400k is None:
        raise RuntimeError("Need matching summary JSON files for both 100k and 400k artifacts")

    res100 = np.abs(np.asarray(data_100k["lam_emp"]) - np.asarray(data_100k["lam_theory"]))
    res400 = np.abs(np.asarray(data_400k["lam_emp"]) - np.asarray(data_400k["lam_theory"]))
    phi100 = np.abs(np.asarray(data_100k["phi_emp"]))
    phi400 = np.abs(np.asarray(data_400k["phi_emp"]))
    q100 = np.asarray(data_100k["q"], dtype=float)
    q400 = np.asarray(data_400k["q"], dtype=float)

    if q100.shape != q400.shape or not np.allclose(q100, q400):
        raise RuntimeError("100k and 400k q grids do not match")

    rows = []
    for thr in THRESHOLDS:
        mask = (phi100 > thr) & (phi400 > thr)
        if np.count_nonzero(mask) == 0:
            rows.append(
                {
                    "phi_threshold": thr,
                    "n_modes": 0,
                    "delta_fixed_median": None,
                    "delta_mc_median": None,
                    "res100_median": None,
                    "res400_median": None,
                }
            )
            continue

        d100 = float(np.median(res100[mask]))
        d400 = float(np.median(res400[mask]))
        mc2 = max((4.0 / 3.0) * max(d100 * d100 - d400 * d400, 0.0), 0.0)
        fixed2 = max(d400 * d400 - mc2 / 4.0, 0.0)
        rows.append(
            {
                "phi_threshold": thr,
                "n_modes": int(np.count_nonzero(mask)),
                "res100_median": d100,
                "res400_median": d400,
                "delta_mc_median": float(np.sqrt(mc2)),
                "delta_fixed_median": float(np.sqrt(fixed2)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Reliable-band verdict test for additive-field kappa generator residuals."
    )
    parser.add_argument("npz", help="Primary validation NPZ file")
    parser.add_argument("--npz-100k", required=True, help="100k exact-Poisson NPZ for MC floor estimate")
    parser.add_argument("--npz-400k", required=True, help="400k exact-Poisson NPZ for MC floor estimate")
    parser.add_argument("--out-prefix", default="reliable_band_verdict")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data, summary, _ = load_npz_with_summary(npz_path)
    if summary is None:
        raise RuntimeError(f"Missing summary JSON for {npz_path}")

    q = np.asarray(data["q"], dtype=float)
    phi_emp = np.asarray(data["phi_emp"], dtype=np.complex128)
    lam_emp = np.asarray(data["lam_emp"], dtype=np.complex128)
    lam_theory = np.asarray(data["lam_theory"], dtype=np.complex128)

    phi_abs = np.abs(phi_emp)
    residual = lam_emp - lam_theory
    abs_residual = np.abs(residual)

    band_rows = compute_stats(abs_residual, phi_abs, q, THRESHOLDS)
    noise_rows = summarize_noise_floor(Path(args.npz_100k), Path(args.npz_400k))
    noise_by_thr = {row["phi_threshold"]: row for row in noise_rows}

    verdict_rows = []
    for row in band_rows:
        thr = row["phi_threshold"]
        merged = dict(row)
        merged.update(noise_by_thr.get(thr, {}))
        if (
            merged.get("median_abs_error") is not None
            and merged.get("delta_mc_median") is not None
            and merged["delta_mc_median"] > 0
        ):
            merged["median_to_mc_ratio"] = merged["median_abs_error"] / merged["delta_mc_median"]
        else:
            merged["median_to_mc_ratio"] = None
        verdict_rows.append(merged)

    summary_out = {
        "input_npz": str(npz_path),
        "theory_mode": summary.get("theory_mode"),
        "exact_poisson": summary.get("exact_poisson"),
        "rows": verdict_rows,
    }

    out_prefix = args.out_prefix
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots(2, 1, figsize=(9.0, 8.0))

    x = np.arange(len(THRESHOLDS))
    labels = [f"{thr:.2f}" for thr in THRESHOLDS]
    med = [row["median_abs_error"] if row["median_abs_error"] is not None else np.nan for row in verdict_rows]
    mean = [row["mean_abs_error"] if row["mean_abs_error"] is not None else np.nan for row in verdict_rows]
    mx = [row["max_abs_error"] if row["max_abs_error"] is not None else np.nan for row in verdict_rows]
    mc = [row["delta_mc_median"] if row["delta_mc_median"] is not None else np.nan for row in verdict_rows]
    fixed = [row["delta_fixed_median"] if row["delta_fixed_median"] is not None else np.nan for row in verdict_rows]
    corr = [row["corr_absq_absres"] if row["corr_absq_absres"] is not None else np.nan for row in verdict_rows]

    ax[0].plot(x, med, "o-", lw=1.8, label="median abs error")
    ax[0].plot(x, mean, "o-", lw=1.8, label="mean abs error")
    ax[0].plot(x, mx, "o-", lw=1.8, label="max abs error")
    ax[0].plot(x, mc, "o--", lw=1.5, label="MC floor (median)")
    ax[0].plot(x, fixed, "o--", lw=1.5, label="fixed floor (median)")
    ax[0].set_xticks(x, labels)
    ax[0].set_xlabel("|P_hat| threshold")
    ax[0].set_ylabel("abs residual")
    ax[0].set_title("Residual statistics versus reliable-band threshold")
    ax[0].legend(frameon=False, fontsize=8)

    for thr in THRESHOLDS:
        mask = phi_abs > thr
        if np.count_nonzero(mask) == 0:
            continue
        order = np.argsort(np.abs(q[mask]))
        ax[1].plot(np.abs(q[mask])[order], abs_residual[mask][order], lw=1.4, label=f"|P_hat|>{thr:.2f}")
    ax[1].set_xlabel("|q|")
    ax[1].set_ylabel("|residual|")
    ax[1].set_title("Residual versus |q| in nested reliable bands")
    ax[1].legend(frameon=False, fontsize=8)

    ax_corr = ax[1].twinx()
    ax_corr.plot(x, corr, "ks--", lw=1.2, ms=4, label="corr(|q|,|res|)")
    ax_corr.set_ylabel("corr(|q|, |residual|)")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    print(json.dumps(summary_out, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
