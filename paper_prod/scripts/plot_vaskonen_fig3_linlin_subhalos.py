"""
Production copy of scripts/plot_vaskonen_fig3_linlin_subhalos.py
Defaults changed to write cache and outputs into paper_prod folders and
to emit a metadata JSON into paper_prod/metadata/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from paper_prod.plot_style import apply_style, FIGURE_SIZES, SUBPLOTS_ADJUST

# apply production style and sizes
SIZES = apply_style()

# When run from paper_prod/scripts, parents[2] is repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ace_lensing"))
sys.path.insert(0, str(ROOT / "build"))

import gwlensing as gw  # noqa: E402
import ace_lensing.model as ace_model  # noqa: E402

COSMO = {
    "h": 0.674,
    "OmegaM": 0.315,
    "sigma8": 0.811,
    "w": -1.0,
}

CURVES = [
    {
        "key": "spherical_halos",
        "label": "spherical halos",
        "color": "#b85bb8",
        "linestyle": "-",
        "kwargs": {"filaments": False, "bias": False, "ell": False, "subhalo": False},
    },
    {
        "key": "spherical_halos_bias",
        "label": "spherical halos + bias",
        "color": "#00e51a",
        "linestyle": "-",
        "kwargs": {"filaments": False, "bias": True, "ell": False, "subhalo": False},
    },
    {
        "key": "elliptical_halos_bias",
        "label": "elliptical halos + bias",
        "color": "#ff8c00",
        "linestyle": "-",
        "kwargs": {"filaments": False, "bias": True, "ell": True, "subhalo": False},
    },
    {
        "key": "elliptical_halos_bias_filaments",
        "label": "ell. halos + bias + filam.",
        "color": "#000000",
        "linestyle": "-",
        "kwargs": {"filaments": True, "bias": True, "ell": True, "subhalo": False},
    },
    {
        "key": "elliptical_halos_bias_filaments_subhalos",
        "label": "ell. halos + bias + filam. + subhalos",
        "color": "#00b7ff",
        "linestyle": "--",
        "kwargs": {"filaments": True, "bias": True, "ell": True, "subhalo": True},
    },
]


def sigma_dl_from_lnmu(lnmu: np.ndarray) -> tuple[float, float, int]:
    valid = np.asarray(lnmu, dtype=float)
    valid = valid[np.isfinite(valid)]
    valid = valid[valid >= -2.302585]
    dl_factor = np.exp(-0.5 * valid)
    w = np.exp(-valid)
    wsum = float(np.sum(w))
    mean = float(np.sum(w * dl_factor) / wsum)
    var = float(np.sum(w * (dl_factor - mean) ** 2) / wsum)
    sigma = float(np.sqrt(max(0.0, var)))
    n_eff = wsum * wsum / float(np.sum(w * w))
    err = float(sigma / np.sqrt(2.0 * max(n_eff - 1.0, 1.0)))
    return sigma, err, int(len(dl_factor))


def run_simulation_task(task: tuple) -> tuple[int, int, float, float, int]:
    c_idx, z_idx, z, curve_kwargs, args_dict, seed, label = task
    import gwlensing as gw
    import numpy as np

    print(f"[Start] {label}: z={z:g}, seed={seed}, nreal={args_dict['nreal']}", flush=True)
    lnmu = gw.sample_lnmu(
        float(z),
        COSMO["OmegaM"],
        COSMO["sigma8"],
        COSMO["h"],
        Nreal=args_dict["nreal"],
        seed=seed,
        Nhalos=args_dict["nhalos"],
        strict_weak_lensing=args_dict["strict_weak_lensing"],
        Mmin=args_dict["mmin"],
        m_floor=args_dict["m_floor"],
        subhalo_threads=args_dict["subhalo_threads"],
        subhalo_parallel_threshold=args_dict["subhalo_parallel_threshold"],
        subhalo_model=args_dict["subhalo_model"],
        subhalo_brute=args_dict["subhalo_brute"],
        subhalo_factor=args_dict["subhalo_factor"],
        **curve_kwargs,
    )

    sigma, err, n = sigma_dl_from_lnmu(lnmu)
    print(f"[Finished] {label}: z={z:g}, sigma_DL/DL={sigma:.6f}", flush=True)
    return c_idx, z_idx, sigma, err, n


def run_curve(z_values: np.ndarray, curve: dict, args: argparse.Namespace) -> dict:
    y, yerr, nvalid = [], [], []
    for i, z in enumerate(z_values):
        seed = args.seed + 1009 * i
        lnmu = gw.sample_lnmu(
            float(z),
            COSMO["OmegaM"],
            COSMO["sigma8"],
            COSMO["h"],
            Nreal=args.nreal,
            seed=seed,
            Nhalos=args.nhalos,
            strict_weak_lensing=args.strict_weak_lensing,
            Mmin=args.mmin,
            m_floor=args.m_floor,
            subhalo_threads=args.subhalo_threads,
            subhalo_parallel_threshold=args.subhalo_parallel_threshold,
            subhalo_model=args.subhalo_model,
            subhalo_brute=args.subhalo_brute,
            subhalo_factor=args.subhalo_factor,
            **curve["kwargs"],
        )
        sigma, err, n = sigma_dl_from_lnmu(lnmu)
        y.append(sigma)
        yerr.append(err)
        nvalid.append(n)
    return {
        "label": curve["label"],
        "color": curve["color"],
        "linestyle": curve["linestyle"],
        "y": y,
        "yerr": yerr,
        "nvalid": nvalid,
    }


def sigma_dl_from_ace_pdf(z: float) -> float:
    mu_edges, pdf = ace_model.predict_pdf(
        Om=COSMO["OmegaM"],
        h=COSMO["h"],
        w=COSMO["w"],
        s8=COSMO["sigma8"],
        z=float(z),
        verbose=False,
    )
    mu_edges = np.asarray(mu_edges, dtype=float)
    pdf = np.asarray(pdf, dtype=float)
    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    width = np.diff(mu_edges)
    prob = pdf[:-1] * width
    prob_sum = np.sum(prob)
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        raise ValueError(f"ACE returned a non-normalizable PDF at z={z:g}")
    prob = prob / prob_sum
    mean_inv_mu = np.sum(prob / mu_mid)
    mean_inv_sqrt_mu = np.sum(prob / np.sqrt(mu_mid))
    var_inv_sqrt_mu = mean_inv_mu - mean_inv_sqrt_mu**2
    return float(np.sqrt(max(0.0, var_inv_sqrt_mu)))


def run_ace_curve(z_values: np.ndarray) -> dict:
    y = []
    for z in z_values:
        try:
            y.append(sigma_dl_from_ace_pdf(float(z)))
        except ValueError:
            y.append(float("nan"))
    return {
        "label": "ACE lensing",
        "color": "#bfbfbf",
        "linestyle": ":",
        "y": y,
        "yerr": [0.0] * len(y),
        "nvalid": [0] * len(y),
    }


def save_plot(results: dict, png_path: Path, pdf_path: Path, ylim: tuple[float, float] = None) -> None:
    z = np.asarray(results["z_values"], dtype=float)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single"])
    fig.subplots_adjust(**SUBPLOTS_ADJUST["single"])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for curve in results["curves"]:
        if curve["label"] == "ACE lensing":
            continue
        color = curve["color"]
        linestyle = curve.get("linestyle", "-")
        for c in CURVES:
            if c["label"] == curve["label"]:
                color = c["color"]
                linestyle = c["linestyle"]
                break

        ax.plot(
            z,
            curve["y"],
            label=curve["label"],
            color=color,
            linestyle=linestyle,
            lw=1.5,
        )
        if results.get("show_error_band", False):
            ax.fill_between(
                z,
                np.asarray(curve["y"]) - np.asarray(curve["yerr"]),
                np.asarray(curve["y"]) + np.asarray(curve["yerr"]),
                color=color,
                alpha=0.10,
                lw=0,
            )

    ax.set_xlim(0.0, 10.0)
    active_curves = [c for c in results["curves"] if c["label"] != "ACE lensing"]
    if ylim is not None:
        ax.set_ylim(ylim)
    elif active_curves:
        ax.set_ylim(0.0, max(0.12, 1.08 * max(max(c["y"]) for c in active_curves)))
    else:
        ax.set_ylim(0.0, 0.12)
    ax.set_xlabel(r"$z$", color="black")
    ax.set_ylabel(r"$\sigma_{D_L}/D_L$", color="black")
    ax.tick_params(colors="black", which="both")
    for spine in ax.spines.values():
        spine.set_color("black")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize=6.5, handlelength=1.5, handletextpad=0.4, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_color("black")
    ax.grid(False)
    fig.savefig(png_path, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nreal", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=240706)
    parser.add_argument("--nhalos", type=int, default=100)
    parser.add_argument("--mmin", type=float, default=1.0e7)
    parser.add_argument("--m-floor", type=float, default=1.0e7)
    parser.add_argument("--subhalo-factor", type=float, default=1.0e-5)
    parser.add_argument("--subhalo-model", type=int, default=1)
    parser.add_argument("--subhalo-brute", action="store_true")
    parser.add_argument("--strict-weak-lensing", action="store_true")
    parser.add_argument("--show-error-band", action="store_true")
    parser.add_argument("--no-ace", action="store_true")
    parser.add_argument("--subhalo-threads", type=int, default=4)
    parser.add_argument("--subhalo-parallel-threshold", type=int, default=1000)
    parser.add_argument("--z-values", default="0.2,0.5,1,1.5,2,3,4,5,6,8,10")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "paper_prod" / "data" / "vaskonen_fig3_linlin_subhalos.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_prod" / "plots" / "figures" / "vaskonen_fig3_linlin_subhalos.png",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel processes to run (1 to disable)")
    parser.add_argument("--ylim-max", type=float, default=None, help="Force maximum y-axis limit")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_metadata(out_png: Path, args: argparse.Namespace):
    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "command": " ".join(sys.argv),
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "git_commit": (_git_rev_short() if True else "unknown"),
        "seed": int(args.seed),
        "nreal": int(args.nreal),
    }
    md_dir = ROOT / "paper_prod" / "metadata"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / (out_png.stem + ".metadata.json")
    md_path.write_text(json.dumps(meta, indent=2))


def _git_rev_short():
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    z_values = np.array([float(z) for z in args.z_values.split(",")], dtype=float)

    if args.cache.exists() and not args.force:
        results = json.loads(args.cache.read_text())
        has_ace = any(curve.get("label") == "ACE lensing" for curve in results["curves"])
        if not args.no_ace and not has_ace:
            results["curves"].append(run_ace_curve(z_values))
            args.cache.write_text(json.dumps(results, indent=2))
    else:
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            tasks = []
            for c_idx, curve in enumerate(CURVES):
                for z_idx, z in enumerate(z_values):
                    seed = args.seed + 1009 * z_idx
                    tasks.append((c_idx, z_idx, z, curve["kwargs"], vars(args), seed, curve["label"]))
            tasks.sort(key=lambda t: (t[3].get("subhalo", False), t[2]), reverse=True)
            curves_data = {
                c_idx: {"y": [0.0] * len(z_values), "yerr": [0.0] * len(z_values), "nvalid": [0] * len(z_values)}
                for c_idx in range(len(CURVES))
            }
            with ProcessPoolExecutor(max_workers=args.jobs, max_tasks_per_child=1) as executor:
                for c_idx, z_idx, sigma, err, n in executor.map(run_simulation_task, tasks):
                    curves_data[c_idx]["y"][z_idx] = sigma
                    curves_data[c_idx]["yerr"][z_idx] = err
                    curves_data[c_idx]["nvalid"][z_idx] = n

            curves_list = []
            for c_idx, curve in enumerate(CURVES):
                curves_list.append({
                    "label": curve["label"],
                    "color": curve["color"],
                    "linestyle": curve["linestyle"],
                    "y": curves_data[c_idx]["y"],
                    "yerr": curves_data[c_idx]["yerr"],
                    "nvalid": curves_data[c_idx]["nvalid"],
                })

            results = {
                "cosmology": COSMO,
                "z_values": z_values.tolist(),
                "nreal": args.nreal,
                "seed": args.seed,
                "nhalos": args.nhalos,
                "mmin": args.mmin,
                "m_floor": args.m_floor,
                "subhalo_factor": args.subhalo_factor,
                "subhalo_model": args.subhalo_model,
                "subhalo_brute": args.subhalo_brute,
                "strict_weak_lensing": args.strict_weak_lensing,
                "show_error_band": args.show_error_band,
                "curves": curves_list,
            }
        else:
            results = {
                "cosmology": COSMO,
                "z_values": z_values.tolist(),
                "nreal": args.nreal,
                "seed": args.seed,
                "nhalos": args.nhalos,
                "mmin": args.mmin,
                "m_floor": args.m_floor,
                "subhalo_factor": args.subhalo_factor,
                "subhalo_model": args.subhalo_model,
                "subhalo_brute": args.subhalo_brute,
                "strict_weak_lensing": args.strict_weak_lensing,
                "show_error_band": args.show_error_band,
                "curves": [run_curve(z_values, curve, args) for curve in CURVES],
            }
        if not args.no_ace:
            results["curves"].append(run_ace_curve(z_values))
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(results, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output.with_suffix(".pdf")
    ylim = (0.0, args.ylim_max) if args.ylim_max is not None else None
    save_plot(results, args.output, pdf_path, ylim=ylim)
    write_metadata(args.output, args)
    print(f"Saved {args.output}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
