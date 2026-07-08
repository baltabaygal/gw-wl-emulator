#!/usr/bin/env python3
"""Generate a manifest.json summarizing figures and metadata in paper_prod."""
from pathlib import Path
import json
import datetime

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_prod" / "plots" / "figures"
MD_DIR = ROOT / "paper_prod" / "metadata"
OUT = ROOT / "paper_prod" / "manifest.json"

items = []
for p in sorted(FIG_DIR.glob("*")):
    if p.suffix.lower() in [".png", ".pdf", ".svg"]:
        md = None
        md_path = MD_DIR / (p.stem + p.suffix + ".metadata.json")
        if not md_path.exists():
            # try without repeating suffix (some metadata use stem only)
            md_path = MD_DIR / (p.stem + ".metadata.json")
        if md_path.exists():
            try:
                md = json.loads(md_path.read_text())
            except Exception:
                md = {"_error": "failed to parse metadata"}
        items.append({
            "file": str(p.relative_to(ROOT)),
            "size": p.stat().st_size,
            "modified": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            "metadata": md,
        })

OUT.write_text(json.dumps({"generated": datetime.datetime.utcnow().isoformat()+"Z", "items": items}, indent=2))
print(f"Wrote {OUT}")
