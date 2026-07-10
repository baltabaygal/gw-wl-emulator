#!/usr/bin/env python3
"""
Production wrapper: generate Vaskonen Fig.3 and write output + metadata to paper_prod.

Usage: run from repo root or directly. This script calls the original
`scripts/plot_vaskonen_fig3_linlin_subhalos.py` with controlled args and records
provenance (git commit, command, seed, timestamp) alongside the produced image.
"""
import subprocess
import json
from pathlib import Path
import datetime
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper_prod" / "plots" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT = ROOT / "scripts" / "plot_vaskonen_fig3_linlin_subhalos.py"
OUT_PNG = OUT_DIR / "vaskonen_fig3_linlin_subhal_submitted.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

def git_commit_short():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        return out.decode().strip()
    except Exception:
        return "unknown"

def main():
    # deterministic seed and reasonable defaults
    nreal = "20000"
    seed = "240706"
    cmd = [sys.executable, str(SCRIPT), "--output", str(OUT_PNG), "--nreal", nreal, "--seed", seed]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)

    meta = {
        "script": str(SCRIPT.relative_to(ROOT)),
        "command": " ".join(shlex.quote(c) for c in cmd),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "seed": int(seed),
        "nreal": int(nreal),
        "output_png": str(OUT_PNG.relative_to(ROOT)),
        "output_pdf": str(OUT_PDF.relative_to(ROOT)),
    }
    MD = ROOT / "paper_prod" / "metadata"
    MD.mkdir(parents=True, exist_ok=True)
    meta_path = MD / (OUT_PNG.stem + ".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_PNG} and metadata {meta_path}")

if __name__ == "__main__":
    main()
