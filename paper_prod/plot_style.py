"""Helper to apply the paper production plotting style.

Usage:
    from paper_prod.plot_style import apply_style
    apply_style()

This ensures consistent fonts, sizes, and PDF font embedding for figures
intended for inclusion in a RevTeX (APS PRD) manuscript.
"""
from pathlib import Path
import matplotlib as mpl
# Ensure the `style` submodule is available even if matplotlib.__init__ does not
import matplotlib.style  # noqa: F401
import matplotlib.ticker as _mticker
import numpy as _np

ROOT = Path(__file__).resolve().parent
STYLE_PATH = ROOT / "plot_style.mplstyle"

FIGURE_SIZES = {
    "single": (3.37, 2.6),
    "double": (7.1, 2.8),
}

SUBPLOTS_ADJUST = {
    "single": {"left": 0.20, "right": 0.95, "bottom": 0.16, "top": 0.92},
    "double": {"left": 0.10, "right": 0.97, "bottom": 0.16, "top": 0.92, "wspace": 0.35},
}

def apply_style():
    """Load the production mplstyle into matplotlib.rcParams."""
    import shutil
    import matplotlib.pyplot as plt
    
    has_latex = shutil.which("latex") is not None
    
    # Try using scienceplots if installed
    try:
        import scienceplots  # noqa: F401
        if has_latex:
            plt.style.use(['science'])
        else:
            plt.style.use(['science', 'no-latex'])
    except Exception:
        # Fall back to custom style sheet
        try:
            mpl.style.use(str(STYLE_PATH))
        except Exception:
            pass
            
    # Apply LaTeX or MathText config
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "cmr10", "DejaVu Serif", "Times New Roman"]
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["axes.unicode_minus"] = False

    if has_latex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
    else:
        mpl.rcParams["text.usetex"] = False

    # Ensure Type-42 (TrueType) fonts are embedded in PDFs (better for arXiv)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    # CRITICAL: Override savefig.bbox that scienceplots sets to "tight".
    # "tight" crops each figure to its content, making canvases different
    # sizes depending on label width. "standard" preserves the exact figsize
    # so all figures are identical when LaTeX scales them to \columnwidth.
    mpl.rcParams["savefig.bbox"] = "standard"
    mpl.rcParams["savefig.pad_inches"] = 0.0

    # Helper sizes (inches) commonly used in scripts.
    sizes = {
        "colwidth": FIGURE_SIZES["single"][0],
        "colheight": FIGURE_SIZES["single"][1],
        "doublewidth": FIGURE_SIZES["double"][0],
        "doubleheight": FIGURE_SIZES["double"][1],
        "single": FIGURE_SIZES["single"],
        "double": FIGURE_SIZES["double"],
    }
    return sizes

# Replace matplotlib's LogFormatter with a small wrapper that prints 0.1/1/10
# instead of $10^{-1}$/$10^0$/$10^1$ for those ticks. This keeps other
# log-formatting behaviour intact but forces the human-readable decimals for
# the common -1,0,1 exponents.
class _PaperLogFormatter(_mticker.LogFormatter):
    def __call__(self, x, pos=None):
        try:
            if _np.isclose(x, 1e-1):
                return "0.1"
            if _np.isclose(x, 1.0):
                return "1"
            if _np.isclose(x, 10.0):
                return "10"
        except Exception:
            pass
        return super().__call__(x, pos)

# Install into the ticker module so that matplotlib will use it when
# constructing log formatters.
_mticker.LogFormatter = _PaperLogFormatter
_mticker.LogFormatterExponent = _PaperLogFormatter
_mticker.LogFormatterMathtext = _PaperLogFormatter

def format_log_axis_decimal(ax, axis='x'):
    """Paper log-axis tick convention: decades n in {-1,0,1} print as the
    plain decimal (0.1, 1, 10); every other decade keeps $10^{n}$ mathtext.
    Not applied automatically by apply_style() -- call this per axis in a
    script wherever its tick range includes 0.1/1/10 and decimal labels read
    better there than $10^{\pm1}$/$10^0$ (e.g. the sigma_partition_vs_* and
    fig_subhalo_sigma_decomposition figures).

    axis: 'x' or 'y' or 'both'
    """
    def _fmt(x, pos=None):
        try:
            if _np.isclose(x, 1e-1):
                return "0.1"
            if _np.isclose(x, 1.0):
                return "1"
            if _np.isclose(x, 10.0):
                return "10"
            # if exact power of ten, use 10^{n}
            exp = int(round(_np.log10(x)))
            if _np.isclose(x, 10.0 ** exp):
                return f"$10^{{{exp}}}$"
            return f"{x:g}"
        except Exception:
            return f"{x:g}"

    fmt = _mticker.FuncFormatter(_fmt)
    if axis in ('x', 'both'):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ('y', 'both'):
        ax.yaxis.set_major_formatter(fmt)

__all__ = ["apply_style", "FIGURE_SIZES", "SUBPLOTS_ADJUST", "format_log_axis_decimal"]
