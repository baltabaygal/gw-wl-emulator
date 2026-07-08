import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import h5py
import numpy as np


def wasserstein_1d(u_samples: np.ndarray, v_samples: np.ndarray) -> float:
    u_sorted = np.sort(u_samples)
    v_sorted = np.sort(v_samples)
    if len(u_sorted) == 0 or len(v_sorted) == 0:
        return float("nan")
    n = min(len(u_sorted), len(v_sorted))
    u_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)
    return float(np.mean(np.abs(u_interp - v_interp)))


def split_stats(path: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    with h5py.File(path, "r") as f:
        lnmu = f["samples/lnmu"]
        counts = f["samples/valid_counts"][:]
        z = f["samples/z"][:]
        h = f["samples/h"][:]
        om = f["samples/OmegaM"][:]
        s8 = f["samples/sigma8"][:]

        num_points = int(lnmu.shape[0])
        nsamples = int(lnmu.shape[1])
        total_valid = int(np.sum(counts))
        total_slots = int(num_points * nsamples)
        valid_fraction = float(total_valid / total_slots) if total_slots > 0 else 0.0
        nan_fraction = 1.0 - valid_fraction

        out["num_points"] = num_points
        out["nsamples_per_point"] = nsamples
        out["total_valid_samples"] = total_valid
        out["valid_fraction"] = valid_fraction
        out["nan_fraction"] = nan_fraction
        out["valid_count_min"] = int(np.min(counts)) if len(counts) else 0
        out["valid_count_max"] = int(np.max(counts)) if len(counts) else 0

        out["ranges"] = {
            "z": [float(np.min(z)), float(np.max(z))],
            "h": [float(np.min(h)), float(np.max(h))],
            "OmegaM": [float(np.min(om)), float(np.max(om))],
            "sigma8": [float(np.min(s8)), float(np.max(s8))],
        }

        pre = dict(f["metadata/preprocessing"].attrs.items())
        out["preprocessing"] = {
            "lnmu_mean": float(pre.get("lnmu_mean", 0.0)),
            "lnmu_std": float(pre.get("lnmu_std", 0.0)),
            "lnmu_min": float(pre.get("lnmu_min", 0.0)),
            "lnmu_max": float(pre.get("lnmu_max", 0.0)),
        }

        res = dict(f.get("metadata/resources", {}).attrs.items()) if "metadata/resources" in f else {}
        out["resources"] = {
            "generation_time_sec": float(res.get("generation_time_sec", np.nan)),
            "throughput_samples_per_sec": float(res.get("throughput_samples_per_sec", np.nan)),
            "mem_before_mb": float(res.get("mem_before_mb", np.nan)),
            "mem_after_mb": float(res.get("mem_after_mb", np.nan)),
            "peak_memory_proxy_mb": float(res.get("peak_memory_proxy_mb", np.nan)),
            "cpu_count": int(res.get("cpu_count", -1)) if not np.isnan(res.get("cpu_count", -1)) else -1,
        }

        inv = dict(f.get("metadata/invalid_stats", {}).attrs.items()) if "metadata/invalid_stats" in f else {}
        out["invalid_stats"] = {
            "total_samples": int(inv.get("total_samples", 0)),
            "valid_samples": int(inv.get("valid_samples", 0)),
            "invalid_samples": int(inv.get("invalid_samples", 0)),
            "invalid_fraction": float(inv.get("invalid_fraction", np.nan)),
            "negative_detA": int(inv.get("negative_detA", 0)),
            "nonfinite_mu": int(inv.get("nonfinite_mu", 0)),
            "negative_mu": int(inv.get("negative_mu", 0)),
            "nan_kappa": int(inv.get("nan_kappa", 0)),
            "nan_gamma": int(inv.get("nan_gamma", 0)),
            "overflow_mu": int(inv.get("overflow_mu", 0)),
            "invalid_logmu": int(inv.get("invalid_logmu", 0)),
            "detA_min": float(inv.get("detA_min", np.nan)),
            "detA_max": float(inv.get("detA_max", np.nan)),
            "detA_mean": float(inv.get("detA_mean", np.nan)),
        }

        # Smoothness summary in sigma8-sorted config order
        order = np.argsort(s8)
        dists: List[float] = []
        for i in range(len(order) - 1):
            a = order[i]
            b = order[i + 1]
            na = int(counts[a])
            nb = int(counts[b])
            if na <= 0 or nb <= 0:
                continue
            ua = lnmu[a, :na]
            ub = lnmu[b, :nb]
            dists.append(wasserstein_1d(ua[np.isfinite(ua)], ub[np.isfinite(ub)]))

        out["wasserstein_adjacent_sigma8_mean"] = float(np.nanmean(dists)) if dists else float("nan")
        out["wasserstein_adjacent_sigma8_std"] = float(np.nanstd(dists)) if dists else float("nan")

    return out


def run_regression_summary() -> Tuple[str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_physics_regression.py", "-q"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        status = "PASS" if proc.returncode == 0 else "FAIL"
        tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
        return status, tail
    except Exception as exc:
        return "ERROR", str(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--output", type=str, default="docs/reference/dataset_report.md")
    args = parser.parse_args()

    splits = ["train", "validation", "test"]
    records = {}
    for split in splits:
        path = os.path.join(args.dataset_dir, split, f"dataset_{split}.h5")
        if os.path.exists(path):
            try:
                records[split] = split_stats(path)
            except Exception as exc:
                print(f"[WARN] Skipping unreadable split file {path}: {exc}")

    status, reg_tail = run_regression_summary()

    lines = ["# Dataset Report", "", f"Dataset directory: `{args.dataset_dir}`", ""]

    if not records:
        lines.append("No dataset splits found.")
    else:
        lines.append("## Split Summary")
        lines.append("")
        lines.append("| Split | Points | Samples/Point | Total Valid | Valid Fraction | NaN Fraction |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for split in splits:
            if split not in records:
                continue
            r = records[split]
            lines.append(
                f"| {split} | {r['num_points']} | {r['nsamples_per_point']} | {r['total_valid_samples']} | {r['valid_fraction']:.4f} | {r['nan_fraction']:.4f} |"
            )

        for split in splits:
            if split not in records:
                continue
            r = records[split]
            lines.extend([
                "",
                f"## {split.capitalize()} Details",
                "",
                "### Parameter Ranges",
                f"- z: [{r['ranges']['z'][0]:.4f}, {r['ranges']['z'][1]:.4f}]",
                f"- h: [{r['ranges']['h'][0]:.4f}, {r['ranges']['h'][1]:.4f}]",
                f"- OmegaM: [{r['ranges']['OmegaM'][0]:.4f}, {r['ranges']['OmegaM'][1]:.4f}]",
                f"- sigma8: [{r['ranges']['sigma8'][0]:.4f}, {r['ranges']['sigma8'][1]:.4f}]",
                "",
                "### Valid Count Stats",
                f"- min valid_count: {r['valid_count_min']}",
                f"- max valid_count: {r['valid_count_max']}",
                "",
                "### Normalization Metadata",
                f"- lnmu_mean: {r['preprocessing']['lnmu_mean']:.6f}",
                f"- lnmu_std: {r['preprocessing']['lnmu_std']:.6f}",
                f"- lnmu_min: {r['preprocessing']['lnmu_min']:.6f}",
                f"- lnmu_max: {r['preprocessing']['lnmu_max']:.6f}",
                "",
                "### Resource Usage",
                f"- generation_time_sec: {r['resources']['generation_time_sec']:.3f}",
                f"- throughput_samples_per_sec: {r['resources']['throughput_samples_per_sec']:.3f}",
                f"- mem_before_mb: {r['resources']['mem_before_mb']:.3f}",
                f"- mem_after_mb: {r['resources']['mem_after_mb']:.3f}",
                f"- peak_memory_proxy_mb: {r['resources']['peak_memory_proxy_mb']:.3f}",
                f"- cpu_count: {r['resources']['cpu_count']}",
                "",
                "### Wasserstein Smoothness (Adjacent in sigma8 order)",
                f"- mean distance: {r['wasserstein_adjacent_sigma8_mean']:.6f}",
                f"- std distance: {r['wasserstein_adjacent_sigma8_std']:.6f}",
                "",
                "### Invalid Sample Diagnostics",
                f"- invalid_fraction: {r['invalid_stats']['invalid_fraction']:.6f}",
                f"- negative_detA: {r['invalid_stats']['negative_detA']}",
                f"- nonfinite_mu: {r['invalid_stats']['nonfinite_mu']}",
                f"- negative_mu: {r['invalid_stats']['negative_mu']}",
                f"- overflow_mu: {r['invalid_stats']['overflow_mu']}",
                f"- nan_kappa: {r['invalid_stats']['nan_kappa']}",
                f"- nan_gamma: {r['invalid_stats']['nan_gamma']}",
                f"- invalid_logmu: {r['invalid_stats']['invalid_logmu']}",
                f"- detA_min: {r['invalid_stats']['detA_min']:.6f}",
                f"- detA_max: {r['invalid_stats']['detA_max']:.6f}",
                f"- detA_mean: {r['invalid_stats']['detA_mean']:.6f}",
            ])

    lines.extend([
        "",
        "## Regression Test Summary",
        f"- Physics regression status: **{status}**",
        "",
        "```text",
        reg_tail,
        "```",
    ])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
