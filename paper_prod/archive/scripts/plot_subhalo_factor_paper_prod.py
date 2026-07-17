#!/usr/bin/env python3
"""
Production wrapper: generate subhalo-factor paper figure panels and copy panel (b)
into `paper_prod/plots/figures/` with metadata.

This runs `scripts/plot_subhalo_factor_paper.py` and collects the outputs.
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

SCRIPT = ROOT / "scripts" / "plot_subhalo_factor_paper.py"
SRC_BASE = ROOT / "plots" / "figures"
PANEL_B_BASENAME = "subhalo_factor_convergence_paper_b"

def git_commit_short():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        return out.decode().strip()
    except Exception:
        return "unknown"

def main():
    # Run original script (it writes to plots/figures)
    cmd = [sys.executable, str(SCRIPT)]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)

    # Copy panel b (pdf/png) into paper_prod
    outputs = []
    for ext in ("pdf", "png"):
        src = SRC_BASE / f"{PANEL_B_BASENAME}.{ext}"
        dst = OUT_DIR / f"{PANEL_B_BASENAME}.{ext}"
        if src.exists():
            dst.write_bytes(src.read_bytes())
            outputs.append(str(dst.relative_to(ROOT)))
            print(f"Copied {src} -> {dst}")
        else:
            print(f"Warning: expected output {src} not found", file=sys.stderr)

    meta = {
        "script": str(SCRIPT.relative_to(ROOT)),
        "command": " ".join(shlex.quote(c) for c in cmd),
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_short(),
        "outputs": outputs,
    }
    MD = ROOT / "paper_prod" / "metadata"
    MD.mkdir(parents=True, exist_ok=True)
    meta_path = MD / f"{PANEL_B_BASENAME}.metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata {meta_path}")

if __name__ == "__main__":
    main()
