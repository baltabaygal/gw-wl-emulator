import argparse
import json
import sys
from pathlib import Path

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
        description="Validate the exact additive-field kappa generator against raw Monte Carlo samples."
    )
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--OmegaM", type=float, default=0.315)
    parser.add_argument("--sigma8", type=float, default=0.811)
    parser.add_argument("--h", type=float, default=0.674)
    parser.add_argument("--Nreal", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--Nhalos", type=int, default=100)
    parser.add_argument("--qmax", type=float, default=80.0)
    parser.add_argument("--nq", type=int, default=201)
    parser.add_argument("--n-u", dest="n_u", type=int, default=256)
    parser.add_argument("--phi-floor", dest="phi_floor", type=float, default=2e-2)
    parser.add_argument("--out-prefix", default="kappa_generator_baseline")
    parser.add_argument("--exact-poisson", action="store_true")
    parser.add_argument("--theory-mode", choices=["midpoint", "kspace"], default="midpoint")
    parser.add_argument("--n-kappa", dest="n_kappa", type=int, default=256)
    parser.add_argument("--cluster-power", dest="cluster_power", type=float, default=4.0)
    args = parser.parse_args()

    gw = load_gwlensing()

    q = np.linspace(-args.qmax, args.qmax, args.nq)

    raw = gw.sample_lensing_raw(
        z=args.z,
        OmegaM=args.OmegaM,
        sigma8=args.sigma8,
        h=args.h,
        Nreal=args.Nreal,
        seed=args.seed,
        filaments=False,
        bias=False,
        ell=False,
        exact_poisson=args.exact_poisson,
        Nhalos=args.Nhalos,
    )
    kappa = np.asarray(raw["kappa"], dtype=float)
    phi_emp = np.exp(-1j * np.outer(q, kappa)).mean(axis=1)
    lam_emp = np.log(phi_emp)

    theory_fn = gw.theory_kappa_generator if args.theory_mode == "midpoint" else gw.theory_kappa_generator_kspace
    theory_kwargs = dict(
        q_values=q,
        z=args.z,
        OmegaM=args.OmegaM,
        sigma8=args.sigma8,
        h=args.h,
        Nreal=args.Nreal,
        seed=args.seed,
        filaments=False,
        bias=False,
        ell=False,
        exact_poisson=args.exact_poisson,
        Nhalos=args.Nhalos,
        n_u=args.n_u,
    )
    if args.theory_mode == "kspace":
        theory_kwargs["n_kappa"] = args.n_kappa
        theory_kwargs["cluster_power"] = args.cluster_power
    lam_theory = np.asarray(theory_fn(**theory_kwargs), dtype=np.complex128)

    mask = np.abs(phi_emp) > args.phi_floor
    abs_err = np.abs(lam_emp - lam_theory)
    rel_err = abs_err / np.maximum(np.abs(lam_theory), 1e-12)

    summary = {
        "z": args.z,
        "OmegaM": args.OmegaM,
        "sigma8": args.sigma8,
        "h": args.h,
        "Nreal": args.Nreal,
        "seed": args.seed,
        "Nhalos": args.Nhalos,
        "nq": args.nq,
        "qmax": args.qmax,
        "n_u": args.n_u,
        "theory_mode": args.theory_mode,
        "n_kappa": args.n_kappa if args.theory_mode == "kspace" else None,
        "cluster_power": args.cluster_power if args.theory_mode == "kspace" else None,
        "phi_floor": args.phi_floor,
        "exact_poisson": bool(args.exact_poisson),
        "mean_kappa_raw": float(np.mean(kappa)),
        "std_kappa_raw": float(np.std(kappa)),
        "reliable_modes": int(mask.sum()),
        "median_abs_error_reliable": float(np.median(abs_err[mask])) if np.any(mask) else None,
        "max_abs_error_reliable": float(np.max(abs_err[mask])) if np.any(mask) else None,
        "median_rel_error_reliable": float(np.median(rel_err[mask])) if np.any(mask) else None,
        "max_rel_error_reliable": float(np.max(rel_err[mask])) if np.any(mask) else None,
        "corr_real_reliable": float(np.corrcoef(lam_emp.real[mask], lam_theory.real[mask])[0, 1]) if np.count_nonzero(mask) > 1 else None,
        "corr_imag_reliable": float(np.corrcoef(lam_emp.imag[mask], lam_theory.imag[mask])[0, 1]) if np.count_nonzero(mask) > 1 else None,
    }

    out_prefix = Path(args.out_prefix)
    np.savez(
        f"{out_prefix}.npz",
        q=q,
        kappa=kappa,
        phi_emp=phi_emp,
        lam_emp=lam_emp,
        lam_theory=lam_theory,
        reliable_mask=mask,
    )
    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"WROTE {out_prefix}.npz {out_prefix}_summary.json")


if __name__ == "__main__":
    main()
