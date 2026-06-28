"""
Shared publication-quality plotting style for all figures.

Import and call `set_paper_style()` at the top of any plotting script, use
`PALETTE` (colorblind-safe Okabe-Ito) for categorical colors, and `save_fig`
to write vector PDF + raster PNG at publication resolution.
"""

import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe qualitative palette
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def set_paper_style():
    """Clean sans-serif journal style (Elsevier/Springer/Scientometrics-like):
    boxed axes, subtle grid, thin lines, Arial/Helvetica with DejaVu fallback."""
    plt.rcParams.update({
        # Fonts: sans-serif, matching typical journal data figures
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Lines and markers
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        # Axes: full box, thin black spines (journal-traditional, contained look)
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        # Ticks: short, pointing out, on left+bottom
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        # Figure / output
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,   # editable text in PDF (TrueType), not outlines
        "ps.fonttype": 42,
    })


def save_fig(fig, path_stem, formats=("pdf", "png")):
    """Save a figure to multiple formats from a path stem (no extension)."""
    from pathlib import Path
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f".{fmt}"))
    plt.close(fig)
