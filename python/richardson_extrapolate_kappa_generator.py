import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_gwlensing():
    repo_root = Path(__file__).resolve().parents[1]
    build_dir = repo_root / "build"
    if build_dir.exists():
        sys.path.insert(0, str(build_dir))
    import gwlensing
    return gwlensing


def richardson_limit(values, hs):
    x = hs
    y = values
    A = np.column_stack([np.ones_like(x), x])
    coeff, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeff[0], coeff[1]


def main():
    parser = argparse.ArgumentParser(
        description="Richardson-style extrapolation of theory-side quadrature error in the kappa generator test."
    )
    parser.add_argument("npz", help="NPZ written by additive_kappa_generator_validation.py")
    parser.add_argument("--summary-json", required=True, help="Matching summary JSON from the same run")
    parser.add_argument("--n-u-list", default="128,256,512,1024,2048")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()

    gw = load_gwlensing()
    data = np.load(args.npz)
    with open(args.summary_json) as f:
        summary = json.load(f)

    q = np.asarray(data["q"], dtype=float)
    lam_emp = np.asarray(data["lam_emp"], dtype=np.complex128)
    mask = np.asarray(data["reliable_mask"], dtype=bool)

    n_u_values = [int(x) for x in args.n_u_list.split(",") if x.strip()]
    hs = np.array([1.0 / n for n in n_u_values], dtype=float)

    lam_theory_by_nu = {}
    for n_u in n_u_values:
        lam_theory_by_nu[n_u] = np.asarray(
            gw.theory_kappa_generator(
                q_values=q,
                z=summary["z"],
                OmegaM=summary["OmegaM"],
                sigma8=summary["sigma8"],
                h=summary["h"],
                Nreal=summary["Nreal"],
                seed=summary["seed"],
                filaments=False,
                bias=False,
                ell=False,
                exact_poisson=summary.get("exact_poisson", False),
                Nhalos=summary["Nhalos"],
                n_u=n_u,
            ),
            dtype=np.complex128,
        )

    lam_stack = np.stack([lam_theory_by_nu[n] for n in n_u_values], axis=0)
    lam_inf = np.zeros_like(lam_stack[0])
    slope = np.zeros_like(lam_stack[0])

    for iq in range(lam_stack.shape[1]):
        a_re, b_re = richardson_limit(lam_stack[:, iq].real, hs)
        a_im, b_im = richardson_limit(lam_stack[:, iq].imag, hs)
        lam_inf[iq] = a_re + 1j * a_im
        slope[iq] = b_re + 1j * b_im

    delta_inf = lam_emp - lam_inf
    abs_delta_inf = np.abs(delta_inf)

    records = []
    for i, n_u in enumerate(n_u_values):
        delta = lam_emp - lam_theory_by_nu[n_u]
        abs_delta = np.abs(delta)
        theory_diff_to_inf = np.abs(lam_theory_by_nu[n_u] - lam_inf)
        records.append(
            {
                "n_u": n_u,
                "median_abs_error_reliable": float(np.median(abs_delta[mask])),
                "median_abs_theory_to_inf_reliable": float(np.median(theory_diff_to_inf[mask])),
                "edge_abs_error": float(abs_delta[np.argmax(np.abs(q))]),
                "edge_abs_theory_to_inf": float(theory_diff_to_inf[np.argmax(np.abs(q))]),
            }
        )

    out = {
        "input_npz": args.npz,
        "input_summary_json": args.summary_json,
        "n_u_values": n_u_values,
        "median_abs_delta_inf_reliable": float(np.median(abs_delta_inf[mask])),
        "max_abs_delta_inf_reliable": float(np.max(abs_delta_inf[mask])),
        "edge_abs_delta_inf": float(abs_delta_inf[np.argmax(np.abs(q))]),
        "q_at_edge": float(q[np.argmax(np.abs(q))]),
        "records": records,
    }

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = str(Path(args.npz).with_suffix("")) + "_richardson"

    np.savez(
        f"{out_prefix}.npz",
        q=q,
        lam_emp=lam_emp,
        lam_inf=lam_inf,
        delta_inf=delta_inf,
        reliable_mask=mask,
    )
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)
    q_abs = np.abs(q)
    order = np.argsort(q_abs)

    ax[0].plot(q_abs[order], abs_delta_inf[order], lw=1.8, label="|Lambda_emp - Lambda_inf|")
    ax[0].set_ylabel("abs residual")
    ax[0].set_title("Richardson-extrapolated residual")
    ax[0].legend(frameon=False)

    ax[1].plot(
        [r["n_u"] for r in records],
        [r["median_abs_error_reliable"] for r in records],
        "o-",
        lw=1.8,
        label="sim-theory error",
    )
    ax[1].plot(
        [r["n_u"] for r in records],
        [r["median_abs_theory_to_inf_reliable"] for r in records],
        "o-",
        lw=1.8,
        label="theory to extrapolated limit",
    )
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("n_u")
    ax[1].set_ylabel("median abs error")
    ax[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.npz {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
