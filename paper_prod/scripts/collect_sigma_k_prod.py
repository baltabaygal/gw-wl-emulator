#!/usr/bin/env python3
"""
Production helper: run the dynamic sigma_k plotting scripts.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def main():
    print("Running dynamic plotter: plot_sigma_k_vs_kappa_threshold.py")
    subprocess.check_call([sys.executable, str(ROOT / "paper_prod" / "scripts" / "plot_sigma_k_vs_kappa_threshold.py")])

    print("Running dynamic plotter: plot_sigma_k_vs_subhalo_factor_paired.py")
    subprocess.check_call([sys.executable, str(ROOT / "paper_prod" / "scripts" / "plot_sigma_k_vs_subhalo_factor_paired.py")])

if __name__ == "__main__":
    main()
