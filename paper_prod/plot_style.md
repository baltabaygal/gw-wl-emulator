Plot style and LaTeX rendering guide
==================================

Purpose
-------
Provide a consistent plotting style that matches the visual appearance of figures
embedded in a RevTeX (APS PRD) manuscript. Use the provided style and helper so
all production scripts render with the same fonts, sizes, layout margins, and PDF
embedding.

Files
-----
- `plot_style.mplstyle` — matplotlib style (rc) settings.
- `plot_style.py` — helper `apply_style()` plus shared figure size/layout constants.

TeX context (your document)
----------------------------
The manuscript uses:

\documentclass[aps,prd,10pt,nofootinbib,twocolumn,superscriptaddress,floatfix,notitlepage]{revtex4-1}

Use vector figures for line-art (PDF/SVG) and 300 DPI PNG for rasters.

How to use
----------
In each production script, call the helper at the start:

```python
from paper_prod.plot_style import apply_style
sizes = apply_style()

# choose width/height when creating the figure
fig, ax = plt.subplots(figsize=(sizes['colwidth'], sizes['colheight']))
```

Or load the style directly:

```python
import matplotlib as mpl
mpl.style.use('paper_prod/plot_style.mplstyle')
```

Fonts and math
--------------
- `font.family` is set to `serif` and `mathtext.fontset` to `cm` so math
  uses Computer Modern glyphs that match LaTeX's appearance.
- We do not force `text.usetex=True` by default because it requires a full LaTeX
  toolchain for every plot; the visual match with `mathtext.fontset='cm'` is
  typically sufficient. If you prefer exact LaTeX typesetting, set
  `mpl.rcParams['text.usetex'] = True` in the script (requires TeX).

Embedding fonts
---------------
PDF/PS font types are set to Type 42 to embed fonts as TrueType (better for
arXiv and downstream tools). The style sets `pdf.fonttype = 42` and `ps.fonttype = 42`.

Figure sizing guidelines
------------------------
- Single-column production figure: `3.37 x 2.6 in` — use `FIGURE_SIZES['single']`
  or `sizes['single']`.
- Double-column production figure: `7.1 x 2.8 in` — use `FIGURE_SIZES['double']`
  or `sizes['double']`.
- Shared subplot margins also live in `SUBPLOTS_ADJUST`, so export dimensions stay
  consistent across scripts.
- For multi-panel figures, create panels sized for the column width and
  assemble them in Inkscape/LaTeX if you need complex layouts.

Export recommendations
----------------------
- Save line-art as PDF (vector): `fig.savefig('figure.pdf')`.
- If raster required, export PNG at 300 DPI: `fig.savefig('figure.png', dpi=300)`.
- Avoid rasterizing text; prefer vector outputs for all text and math.

Checklist for a production plotter script
----------------------------------------
1. Start with `apply_style()`.
2. Fix RNG seeds where random sampling is used.
3. Use `FIGURE_SIZES` and `SUBPLOTS_ADJUST` from `plot_style.py` instead of
   repeating literal dimensions in scripts.
4. Save both `figure.pdf` and `figure.png` (PDF for manuscript, PNG for preview).
5. Write a small metadata JSON next to the figure recording the script, command,
   git commit, seed, and timestamp (production scripts already do this).

Examples
--------
See `paper_prod/scripts/plot_vaskonen_fig3_linlin_subhalos.py` and
`paper_prod/scripts/plot_subhalo_factor_paper.py` for production examples that
load data from `paper_prod/data/`, write figures to `paper_prod/plots/figures/`,
and emit metadata to `paper_prod/metadata/`.
