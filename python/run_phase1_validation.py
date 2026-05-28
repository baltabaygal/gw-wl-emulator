import os
import subprocess
import sys
import argparse
from typing import List, Tuple


def run_step(name: str, cmd: List[str]) -> Tuple[bool, str]:
    print(f"\n--- {name} ---")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        print(proc.stdout)
    ok = proc.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    return ok, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets_tiny")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase1_validation")
    parser.add_argument("--min_valid_fraction", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    checks = []
    checks.append(run_step(
        "Dataset generation",
        [
            sys.executable,
            "python/generate_dataset.py",
            "--num_points", "10",
            "--nsamples", "1000",
            "--seed", "123",
            "--output_dir", args.dataset_dir,
        ],
    ))

    checks.append(run_step(
        "Dataset validation",
        [
            sys.executable,
            "python/validate_dataset.py",
            "--dataset_dir", args.dataset_dir,
            "--min_valid_fraction", str(args.min_valid_fraction),
        ],
    ))

    checks.append(run_step(
        "Physics regression tests",
        [sys.executable, "-m", "pytest", "tests/test_physics_regression.py"],
    ))

    checks.append(run_step(
        "Smoothness diagnostics",
        [sys.executable, "python/smoothness_diagnostics.py", "--output_dir", args.plots_dir],
    ))

    checks.append(run_step(
        "Monte Carlo diagnostics",
        [sys.executable, "python/diagnostics.py", "--output_dir", args.plots_dir],
    ))

    checks.append(run_step(
        "Parameter coverage plots",
        [
            sys.executable,
            "python/plot_parameter_coverage.py",
            "--dataset_dir", args.dataset_dir,
            "--output_dir", args.plots_dir,
        ],
    ))

    print("\n=== Phase 1 Validation Summary ===")
    all_ok = True
    for (ok, _), name in zip(
        checks,
        [
            "Dataset generation",
            "Dataset validation",
            "Physics regression tests",
            "Smoothness diagnostics",
            "Monte Carlo diagnostics",
            "Parameter coverage plots",
        ],
    ):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
