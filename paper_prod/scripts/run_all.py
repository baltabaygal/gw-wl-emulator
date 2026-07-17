#!/usr/bin/env python3
"""
Master script to run all production plotting scripts in the pipeline
and regenerate the manifest.json.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "paper_prod" / "scripts"

PLOTTING_STEPS = [
    {
        "name": "sigma partition vs kappa threshold",
        "script": "plot_sigma_partition_vs_kthr.py",
        "args": []
    },
    {
        "name": "Vaskonen fig3 lin-lin subhalos",
        "script": "plot_vaskonen_fig3_linlin_subhalos.py",
        "args": [
            "--output", str(ROOT / "paper_prod" / "plots" / "figures" / "vaskonen_fig3_linlin_subhal_submitted.png"),
            "--nreal", "400000",
            "--seed", "240706",
            "--strict-weak-lensing",
            "--cache", str(ROOT / "data" / "vaskonen_fig3_linlin_subhalos_strict_n400000.json")
        ]
    }
]

def main():
    import os
    t_start = time.time()
    print("=" * 60)
    print("Starting Unified Plotting Pipeline...")
    print("=" * 60)

    for step in PLOTTING_STEPS:
        name = step["name"]
        script_name = step["script"]
        args = step["args"]
        script_path = SCRIPTS_DIR / script_name

        print(f"\n[Running Step] {name}...")
        print(f"Command: {sys.executable} {script_name} {' '.join(args)}")
        
        t0 = time.time()
        try:
            cmd = [sys.executable, str(script_path)] + args
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
            subprocess.check_call(cmd, cwd=str(ROOT), env=env)
            print(f"[Success] {name} completed in {time.time() - t0:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"[Error] {name} failed with exit code {e.returncode}")
            sys.exit(e.returncode)

    # Step 5: Refresh manifest
    print("\n[Running Step] Refreshing manifest...")
    manifest_script = SCRIPTS_DIR / "generate_manifest.py"
    try:
        subprocess.check_call([sys.executable, str(manifest_script)], cwd=str(ROOT), env=env)
        print("[Success] manifest.json successfully updated")
    except subprocess.CalledProcessError as e:
        print(f"[Error] generate_manifest.py failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    print("\n" + "=" * 60)
    print(f"Pipeline successfully completed in {time.time() - t_start:.1f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()
