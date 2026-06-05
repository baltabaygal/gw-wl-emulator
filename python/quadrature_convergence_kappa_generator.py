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


def main():
    parser = argparse.ArgumentParser(
        description="Check theory-side quadrature convergence for the additive kappa generator."
    )
    parser.add_argument("npz", help="NPZ written by additive_kappa_generator_validation.py")
    parser.add_argument("--summary-json", required=True, help="Matching summary JSON from the same run")
    parser.add_argument("--n-u-list", default="64,128,256,512,1024")
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
    records = []
    lam_theory_by_nu = {}

    for n_u in n_u_values:
        lam_theory = np.asarray(
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
        lam_theory_by_nu[n_u] = lam_theory
        delta = lam_emp - lam_theory
        abs_delta = np.abs(delta)
        rec = {
            "n_u": n_u,
            "median_abs_error_reliable": float(np.median(abs_delta[mask])),
            "max_abs_error_reliable": float(np.max(abs_delta[mask])),
            "edge_abs_error": float(abs_delta[np.argmax(np.abs(q))]),
            "edge_q": float(q[np.argmax(np.abs(q))]),
        }
        records.append(rec)

    ref_nu = n_u_values[-1]
    ref = lam_theory_by_nu[ref_nu]
    q_abs = np.abs(q)
    order = np.argsort(q_abs)

    out_prefix = args.out_prefix
    if out_prefix is None:
        out_prefix = str(Path(args.npz).with_suffix("")) + "_quadrature"

    plot_rows = []
    for n_u in n_u_values[:-1]:
        diff = np.abs(lam_theory_by_nu[n_u] - ref)
        plot_rows.append((n_u, diff))

    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.2), sharex=True)

    ax[0].plot(
        [r["n_u"] for r in records],
        [r["median_abs_error_reliable"] for r in records],
        "o-",
        lw=1.8,
        label="median |Lambda_emp - Lambda_theory|",
    )
    ax[0].plot(
        [r["n_u"] for r in records],
        [r["edge_abs_error"] for r in records],
        "o-",
        lw=1.8,
        label=f"|deltaLambda| at |q|={records[0]['edge_q']:.1f}",
    )
    ax[0].set_xscale("log", base=2)
    ax[0].set_ylabel("abs error")
    ax[0].set_title("Quadrature convergence diagnostic")
    ax[0].legend(frameon=False)

    for n_u, diff in plot_rows:
        ax[1].plot(q_abs[order], diff[order], lw=1.5, label=f"n_u={n_u} vs {ref_nu}")
    ax[1].set_xlabel("|q_kappa|")
    ax[1].set_ylabel("|Lambda_theory(n_u)-Lambda_theory(ref)|")
    ax[1].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=170)

    out = {
        "input_npz": args.npz,
        "input_summary_json": args.summary_json,
        "reference_n_u": ref_nu,
        "records": records,
    }
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.png {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
