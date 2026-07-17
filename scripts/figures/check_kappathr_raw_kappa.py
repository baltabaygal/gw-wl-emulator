#!/usr/bin/env python3
"""Raw-kappa variance diagnostic for the PDF threshold acceptance run."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np


def sigma_se(x: np.ndarray) -> float:
    centered = x - np.mean(x)
    variance = float(np.mean(centered * centered))
    fourth = float(np.mean(centered ** 4))
    return math.sqrt(max(0.0, fourth - variance * variance) / x.size) / (2.0 * math.sqrt(variance))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsamples", type=int, default=200_000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "build"))
    import gwlensing as gw  # pylint: disable=import-error,import-outside-toplevel

    rules = {"adaptive": -1.0, "flat_1e-4": 1.0e-4, "flat_1e-3": 1.0e-3}
    rows = []
    arrays = {}
    for bias in (True, False):
        for z in (0.2, 1.0, 10.0):
            reference_sigma = None
            z_rows = []
            for label, threshold in rules.items():
                start = time.monotonic()
                result = gw.sample_lensing_raw_ml(
                    z=z, h=0.674, OmegaM=0.315, sigma8=0.811,
                    nsamples=args.nsamples, seed=123, filaments=True,
                    bias=bias, ell=True, subhalo=False,
                    kappathr_flat=threshold)
                kappa = np.asarray(result["kappa"], dtype=np.float64)
                centered = kappa - np.mean(kappa)
                arrays[f"bias{int(bias)}_z{str(z).replace('.', 'p')}_{label}"] = kappa
                row = {
                    "bias": bias,
                    "z": z,
                    "rule": label,
                    "kappathr_flat": threshold,
                    "nsamples": kappa.size,
                    "mean_kappa": float(np.mean(kappa)),
                    "sigma_kappa": float(np.std(kappa)),
                    "sigma_kappa_se": sigma_se(kappa),
                    "sigma_kappa_clipped_abs_1": float(np.std(centered[np.abs(centered) < 1.0])),
                    "runtime_seconds": time.monotonic() - start,
                }
                if label == "adaptive":
                    reference_sigma = row["sigma_kappa"]
                row["sigma_relative_to_adaptive"] = row["sigma_kappa"] / reference_sigma - 1.0
                z_rows.append(row)
                print(f"bias={bias} z={z:g} {label}: sigma_k={row['sigma_kappa']:.7g} "
                      f"({row['runtime_seconds']:.1f} s)", flush=True)
            reference_clipped = z_rows[0]["sigma_kappa_clipped_abs_1"]
            for row in z_rows:
                row["clipped_sigma_relative_to_adaptive"] = (
                    row["sigma_kappa_clipped_abs_1"] / reference_clipped - 1.0)
            rows.extend(z_rows)

    out = root / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "kappathr_raw_kappa_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "kappathr_raw_kappa_samples.npz", **arrays)


if __name__ == "__main__":
    main()
