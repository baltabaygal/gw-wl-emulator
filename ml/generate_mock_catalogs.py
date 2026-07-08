import argparse
import json
from pathlib import Path

import numpy as np

from ml.phase3_common import (
    TRUE_COSMOLOGIES,
    catalog_hash,
    ensure_dir,
    import_gwlensing,
    make_redshifts,
    save_npz_with_metadata,
    simulator_git_commit,
    stable_json_hash,
    utc_timestamp,
)


def generate_lnmu_for_redshifts(z: np.ndarray, h: float, omega_m: float, sigma8: float, seed: int) -> np.ndarray:
    gw = import_gwlensing()
    rng = np.random.default_rng(seed)
    lnmu = np.empty_like(z, dtype=np.float64)
    rounded_z = np.round(z, 3)
    for z_val in np.unique(rounded_z):
        idx = np.where(rounded_z == z_val)[0]
        finite_chunks = []
        needed = int(len(idx))
        attempts = 0
        while sum(chunk.size for chunk in finite_chunks) < needed and attempts < 8:
            attempts += 1
            sim_seed = int(rng.integers(1, 2**31 - 1))
            request_n = int(np.ceil((needed - sum(chunk.size for chunk in finite_chunks)) * 1.10)) + 8
            res = gw.sample_lnmu_ml_with_diagnostics(float(z_val), h, omega_m, sigma8, request_n, sim_seed, False)
            samples = np.asarray(res["lnmu"], dtype=np.float64)
            samples = samples[np.isfinite(samples)]
            if samples.size:
                finite_chunks.append(samples)
        if sum(chunk.size for chunk in finite_chunks) < needed:
            raise RuntimeError(f"Simulator returned too few finite samples for {needed} requested at z={z_val}")
        lnmu[idx] = np.concatenate(finite_chunks)[:needed]
    if not np.isfinite(lnmu).all():
        raise ValueError("Mock catalog contains non-finite lnmu values")
    return lnmu


def build_catalog(
    catalog_type: str,
    n: int,
    seed: int,
    cosmology_id: str,
    z: float | None = None,
    output_dir: str | Path = "data/mock_catalogs/phase3",
    mixed_z_bins: int = 3,
) -> Path:
    if cosmology_id not in TRUE_COSMOLOGIES:
        raise ValueError(f"Unknown cosmology_id {cosmology_id}; options={sorted(TRUE_COSMOLOGIES)}")
    cosmo = TRUE_COSMOLOGIES[cosmology_id]
    redshifts, z_distribution = make_redshifts(catalog_type, n, seed, z, mixed_z_bins=mixed_z_bins)
    lnmu = generate_lnmu_for_redshifts(redshifts, cosmo.h, cosmo.OmegaM, cosmo.sigma8, seed)

    catalog_id_payload = {
        "catalog_type": catalog_type,
        "z_distribution": z_distribution,
        "N": int(n),
        "seed": int(seed),
        "cosmology_id": cosmology_id,
        "mixed_z_bins": int(mixed_z_bins),
        "true": {"h": cosmo.h, "OmegaM": cosmo.OmegaM, "sigma8": cosmo.sigma8},
    }
    catalog_id = stable_json_hash(catalog_id_payload)[:16]
    metadata = {
        "catalog_id": catalog_id,
        "catalog_type": catalog_type,
        "z_distribution": z_distribution,
        "N": int(n),
        "seed": int(seed),
        "cosmology_id": cosmology_id,
        "true_h": float(cosmo.h),
        "true_OmegaM": float(cosmo.OmegaM),
        "true_sigma8": float(cosmo.sigma8),
        "simulator_git_commit": simulator_git_commit(),
        "generation_timestamp": utc_timestamp(),
    }
    metadata["catalog_hash"] = catalog_hash(redshifts, lnmu, metadata)

    suffix = f"{catalog_type}_{cosmology_id}_N{n}_seed{seed}"
    if z is not None:
        suffix += f"_z{z:g}"
    path = ensure_dir(output_dir) / f"{suffix}_{catalog_id}.npz"
    save_npz_with_metadata(path, {"z": redshifts, "lnmu": lnmu}, metadata)
    return path


def default_catalog_specs(smoke: bool = False) -> list[dict]:
    sizes = [1000] if smoke else [1000, 10000]
    specs = []
    seed0 = 310000
    for n in sizes:
        for z in [0.5, 1.5, 2.5]:
            specs.append({"catalog_type": "single_z", "z": z, "n": n, "seed": seed0 + len(specs), "cosmology_id": "central"})
        for cosmology_id in ["central", "edge", "low_sigma8", "high_sigma8"]:
            specs.append({"catalog_type": "mixed_uniform", "z": None, "n": n, "seed": seed0 + len(specs), "cosmology_id": cosmology_id})
    return specs


def write_report(paths: list[Path], output_report: str | Path) -> None:
    lines = [
        "# Phase 3 Mock Catalogs",
        "",
        "Mock catalogs are generated with the current C++ weak-lensing simulator and stored as `.npz` files with JSON metadata.",
        "",
        "| Catalog | N | Type | Truth | Hash |",
        "|---|---:|---|---|---|",
    ]
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            md = json.loads(str(npz["metadata"]))
        truth = f"({md['true_h']:.3f}, {md['true_OmegaM']:.3f}, {md['true_sigma8']:.3f})"
        lines.append(f"| `{path.name}` | {md['N']} | {md['catalog_type']} | {truth} | `{md['catalog_hash'][:12]}` |")
    lines.extend(
        [
            "",
            "Required metadata fields are included: catalog id/type, redshift distribution, N, seed, true cosmology, simulator git commit, generation timestamp, and catalog hash.",
        ]
    )
    ensure_dir(Path(output_report).parent)
    Path(output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 weak-lensing mock catalogs.")
    parser.add_argument("--output_dir", default="data/mock_catalogs/phase3")
    parser.add_argument("--output_report", default="docs/phase3/phase3_mock_catalogs.md")
    parser.add_argument("--smoke", action="store_true", help="Generate the minimal N=1000 catalog set.")
    parser.add_argument("--mixed_z_bins", type=int, default=3)
    args = parser.parse_args()

    paths = []
    for spec in default_catalog_specs(smoke=args.smoke):
        path = build_catalog(
            spec["catalog_type"],
            spec["n"],
            spec["seed"],
            spec["cosmology_id"],
            spec.get("z"),
            args.output_dir,
            mixed_z_bins=args.mixed_z_bins,
        )
        print(f"Wrote {path}")
        paths.append(path)
    write_report(paths, args.output_report)
    print(f"Wrote {args.output_report}")


if __name__ == "__main__":
    main()
