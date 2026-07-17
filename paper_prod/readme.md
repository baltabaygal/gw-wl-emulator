# Paper Production — Plots & Plotters

This folder holds the active production plots intended for paper submission. The goal is clear provenance and reproducibility: the current manuscript figure set should be reproducible from the scripts and minimal derived data stored here.
This directory is the self-contained production bundle for the active paper figures. Retired figures, plotters, and related metadata have been moved to `paper_prod/archive/` so the live folder stays focused on the submission set.

Layout

- `scripts/` — active production plotters and small helpers. Each script should be
	runnable from the repository root and write its outputs into `paper_prod/plots/figures/`.
- `data/` — cached or derived inputs still needed by the active plotters. Large raw datasets remain at the repo root `data/` and are referenced.
- `plots/figures/` — the current publication-ready images (PDF/PNG) that belong in the manuscript.
- `metadata/` — per-figure JSON provenance files for the active figure set.
- `archive/` — retired plots, plotters, metadata, and data kept for reference but excluded from the live production bundle.
- `manifest.json` — auto-generated index of the active figures + metadata (see `scripts/generate_manifest.py`).

Quick usage

1. Activate the project `test` environment (required for compiled extensions):

```bash
conda activate test
```

2. From the repository root, run a production script. Examples:

```bash
python paper_prod/scripts/plot_sigma_partition_vs_kthr.py
python paper_prod/scripts/plot_vaskonen_fig3_linlin_subhalos.py --output paper_prod/plots/figures/vaskonen_fig3_linlin_subhal_submitted.png --nreal 400000 --seed 240706 --strict-weak-lensing --cache data/vaskonen_fig3_linlin_subhalos_strict_n400000.json
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

- Active production plotters: `paper_prod/scripts/plot_sigma_partition_vs_kthr.py` and `paper_prod/scripts/plot_vaskonen_fig3_linlin_subhalos.py`.
- Active outputs: `paper_prod/plots/figures/sigma_partition_vs_kthr.{png,pdf}` and `paper_prod/plots/figures/vaskonen_fig3_linlin_subhal_submitted.{png,pdf}`.
- Active metadata: `paper_prod/metadata/`.
- Archived material: `paper_prod/archive/`.

Current active production figures

- `paper_prod/plots/figures/sigma_partition_vs_kthr.{png,pdf}`
- `paper_prod/plots/figures/vaskonen_fig3_linlin_subhal_submitted.{png,pdf}`

Make the scripts executable if you prefer to run them directly as shell commands:

```bash
chmod +x paper_prod/scripts/*.py
./paper_prod/scripts/plot_vaskonen_fig3_prod.py
```
