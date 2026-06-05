import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_runs():
    sigma8_values = [0.75, 0.81, 0.87]
    z_values = [0.5, 1.0, 2.0, 5.0]
    files = {s: Path(f"collapse_sigma8_{s:.2f}_n200k.npz") for s in sigma8_values}
    for f in files.values():
        if not f.exists():
            raise FileNotFoundError(f"Missing input NPZ: {f}")

    data = {s: np.load(files[s]) for s in sigma8_values}

    # Global common k band across all sigma8 runs.
    k_sets = [set(np.round(data[s]["k"], 12).tolist()) for s in sigma8_values]
    k_common = np.array(sorted(set.intersection(*k_sets)), dtype=float)
    if k_common.size == 0:
        raise RuntimeError("No common k modes found across NPZ inputs.")

    idx_map = {}
    for s in sigma8_values:
        ks = np.round(data[s]["k"], 12).tolist()
        idx_map[s] = {float(kv): i for i, kv in enumerate(ks)}

    # Build per-run real Lambda(k,z,sigma8).
    runs = []
    for s in sigma8_values:
        M = data[s]["M"]  # shape: nk x nz, entries are Re Lambda
        for zi, z in enumerate(z_values):
            lam = np.array(
                [M[idx_map[s][float(np.round(kv, 12))], zi] for kv in k_common],
                dtype=float,
            )
            runs.append({"sigma8": s, "z": z, "lambda_real": lam})
    return k_common, runs


def load_threshold_sweep():
    p = Path("nu_threshold_sweep_z1.json")
    if not p.exists():
        raise FileNotFoundError("Missing nu_threshold_sweep_z1.json")
    with p.open() as f:
        d = json.load(f)
    rows = d["rows"]
    kappa = np.array([r["kappa_thr"] for r in rows], dtype=float)
    gamma = np.array([r["gamma_pred"] for r in rows], dtype=float)
    order = np.argsort(kappa)
    return kappa[order], gamma[order]


def interp_gamma_logkappa(kappa_eff, kappa_grid, gamma_grid):
    # Piecewise linear in log(kappa). Clamp to range.
    x = np.log(kappa_grid)
    y = gamma_grid
    xq = np.log(np.clip(kappa_eff, kappa_grid.min(), kappa_grid.max()))
    return float(np.interp(xq, x, y))


def main():
    mask_levels = [0.10, 0.05, 0.02]
    gamma_obs = 1.46

    k, runs = load_runs()
    kappa_grid, gamma_grid = load_threshold_sweep()

    rows = []
    for mthr in mask_levels:
        # We do not have complex P_hat directly in the saved NPZs.
        # Approximation: |P_hat(k,z)| ~= exp(z * Re[Lambda(k,z)]).
        # This is exact if Im[Lambda] is small and phase oscillations are subdominant.
        joint = np.ones_like(k, dtype=bool)
        for r in runs:
            phat_abs = np.exp(r["z"] * r["lambda_real"])
            joint &= phat_abs > mthr

        if not np.any(joint):
            rows.append(
                {
                    "mask_threshold": mthr,
                    "joint_modes": 0,
                    "k_max": None,
                    "kappa_thr_eff": None,
                    "gamma_pred_implied": None,
                    "gamma_obs": gamma_obs,
                }
            )
            continue

        kmax = float(np.max(np.abs(k[joint])))
        kappa_eff = 1.0 / (2.0 * kmax)
        gamma_imp = interp_gamma_logkappa(kappa_eff, kappa_grid, gamma_grid)
        rows.append(
            {
                "mask_threshold": mthr,
                "joint_modes": int(np.count_nonzero(joint)),
                "k_max": kmax,
                "kappa_thr_eff": kappa_eff,
                "gamma_pred_implied": gamma_imp,
                "gamma_obs": gamma_obs,
            }
        )

    # Save summary JSON.
    out = {
        "note": (
            "Using |P_hat| ~= exp(z*Re[Lambda]) from stored transport spectra. "
            "Gamma interpolation is linear in log(kappa_thr) using nu_threshold_sweep_z1.json."
        ),
        "mask_levels": mask_levels,
        "rows": rows,
    }
    with open("effective_threshold_from_band_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    # Plot kappa_eff vs mask threshold.
    valid = [r for r in rows if r["kappa_thr_eff"] is not None]
    x = np.array([r["mask_threshold"] for r in valid], dtype=float)
    yk = np.array([r["kappa_thr_eff"] for r in valid], dtype=float)
    yg = np.array([r["gamma_pred_implied"] for r in valid], dtype=float)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, yk, "o-", lw=1.8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Mask threshold on |P_hat|")
    plt.ylabel("kappa_thr_eff ~ 1/(2 k_max)")
    plt.title("Effective threshold from spectral band edge")
    plt.tight_layout()
    plt.savefig("effective_threshold_vs_mask.png", dpi=170)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, yg, "o-", lw=1.8, label="Implied gamma_pred")
    plt.axhline(gamma_obs, color="k", ls="--", lw=1.0, label="gamma_obs=1.46")
    plt.xscale("log")
    plt.xlabel("Mask threshold on |P_hat|")
    plt.ylabel("Implied gamma_pred")
    plt.title("Implied gamma vs spectral mask threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("implied_gamma_vs_mask.png", dpi=170)

    print(json.dumps(rows, indent=2))
    print(
        "WROTE effective_threshold_from_band_summary.json "
        "effective_threshold_vs_mask.png implied_gamma_vs_mask.png"
    )


if __name__ == "__main__":
    main()
