# Paper Production — Plots & Plotters

This folder holds the canonical plotters, notebooks, and final figures intended for paper submission. The goal is clear provenance and reproducibility: every figure in the manuscript should be reproducible from the scripts and minimal derived data stored here.
This directory is the self-contained production bundle for paper figures. It contains
only the files you need to reproduce final plots and to publish them with clear
provenance.

Layout

- `scripts/` — production-ready plotters and small helpers. Each script should be
	runnable from the repository root and write its outputs into `paper_prod/plots/figures/`.
- `data/` — cached or derived inputs required by the plotters (small binary files,
	CSVs, JSON). Large raw datasets remain at the repo root `data/` and are referenced.
- `plots/figures/` — final publication-ready images (PDF/PNG/SVG). These are the
	files to include with the manuscript.
- `metadata/` — per-figure JSON provenance files (script, command, git commit, seed,
	timestamp).
- `manifest.json` — auto-generated index of figures + metadata (see `scripts/generate_manifest.py`).

Quick usage

1. Activate the project `test` environment (required for compiled extensions):

```bash
conda activate test
```

2. From the repository root, run a production script. Examples:

```bash
python paper_prod/scripts/plot_vaskonen_fig3_linlin_subhalos.py
python paper_prod/scripts/plot_subhalo_factor_paper.py
python paper_prod/scripts/collect_sigma_k_prod.py
```

3. Generate (or refresh) the manifest:

```bash
python paper_prod/scripts/generate_manifest.py
```

4. Compile the LaTeX layout check from the repository root:

```bash
pdflatex -output-directory paper_prod paper_prod/test_plot_layout.tex
```

Conventions

- Scripts must write outputs into `paper_prod/plots/figures/` and metadata into `paper_prod/metadata/`.
- Data required by production scripts must live under `paper_prod/data/` (small files only).
- `manifest.json` is authoritative for what goes into the submission bundle.
- `test_plot_layout.tex` is a quick RevTeX preview that places the exported plots
  into a two-column manuscript layout so you can verify sizing and rendering.

Files I added or updated

- Production plotters: `paper_prod/scripts/*` (copies/wrappers of the canonical plotters).
- Data caches copied to: `paper_prod/data/`.
- Outputs collected at: `paper_prod/plots/figures/`.
- Per-figure metadata: `paper_prod/metadata/`.

Next recommendations

- Review `paper_prod/plots/figures/` and delete any intermediate images you don't intend
	to publish.
- If you want a single command to rebuild everything, I can add a `Makefile` or a
	`run_all.py` that runs the production scripts and regenerates `manifest.json`.

If you'd like, I can now (A) remove the production script copies and update the
original scripts in-place to write into `paper_prod/` by default, or (B) keep the
copies under `paper_prod/scripts/` and leave the originals unchanged. Tell me which
option you prefer and I'll apply the change.

Regenerating selected figures (already added)

- `paper_prod/scripts/plot_vaskonen_fig3_prod.py` — runs the Vaskonen Fig.3 generator and writes `paper_prod/plots/figures/vaskonen_fig3_linlin_subhal_submitted.png` (+ metadata JSON).
- `paper_prod/scripts/plot_subhalo_factor_paper_prod.py` — runs the subhalo-factor paper plot and copies panel (b) into `paper_prod/plots/figures/` (+ metadata JSON).
- `paper_prod/scripts/collect_sigma_k_prod.py` — copies existing `sigma_k` images into `paper_prod/plots/figures/` and adds metadata; extendable to rerun generators if desired.

Run examples (from repo root):

```bash
python paper_prod/scripts/plot_vaskonen_fig3_prod.py
python paper_prod/scripts/plot_subhalo_factor_paper_prod.py
python paper_prod/scripts/collect_sigma_k_prod.py
```

Make the scripts executable if you prefer to run them directly as shell commands:

```bash
chmod +x paper_prod/scripts/*.py
./paper_prod/scripts/plot_vaskonen_fig3_prod.py
```
