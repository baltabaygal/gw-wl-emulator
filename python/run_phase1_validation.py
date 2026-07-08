import os
import subprocess
import sys
import argparse
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import DEFAULT_MIN_VALID_FRACTION

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
    parser.add_argument("--dataset_dir", type=str, default="datasets/tiny")
    parser.add_argument("--plots_dir", type=str, default="plots/figures/phase1_validation")
    parser.add_argument("--min_valid_fraction", type=float, default=DEFAULT_MIN_VALID_FRACTION)
    parser.add_argument("--output_report", type=str, default="docs/reference/dataset_validation_report.md")
    parser.add_argument("--json_output", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    checks = []

    # Check if dataset already exists to skip generation step
    train_h5 = os.path.join(args.dataset_dir, "train", f"dataset_train.h5")
    val_h5 = os.path.join(args.dataset_dir, "validation", f"dataset_validation.h5")
    test_h5 = os.path.join(args.dataset_dir, "test", f"dataset_test.h5")
    dataset_exists = os.path.exists(train_h5) and os.path.exists(val_h5) and os.path.exists(test_h5)

    if dataset_exists:
        print(f"Dataset files already exist in {args.dataset_dir}. Skipping generation step to preserve data.")
        checks.append((True, "Skipped (already generated)"))
    else:
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
        "Automated tests suite (pytest)",
        [sys.executable, "-m", "pytest", "tests/"],
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

    cmd_report = [
        sys.executable,
        "python/report_generator.py",
        "--dataset_dir", args.dataset_dir,
        "--output", args.output_report,
    ]
    if args.json_output:
        cmd_report.extend(["--json_output", args.json_output])

    checks.append(run_step(
        "Report generation",
        cmd_report,
    ))

    print("\n=== Phase 1 Validation Summary ===")
    all_ok = True
    for (ok, _), name in zip(
        checks,
        [
            "Dataset generation",
            "Dataset validation",
            "Automated tests suite (pytest)",
            "Smoothness diagnostics",
            "Monte Carlo diagnostics",
            "Parameter coverage plots",
            "Report generation",
        ],
    ):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
