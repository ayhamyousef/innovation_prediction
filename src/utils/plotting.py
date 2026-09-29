"""
Shared publication-quality plotting style and named-color system.

Design conventions:
  - Sans-serif (Liberation Sans, Arial-metric-compatible) with DejaVu fallback.
  - SEMANTIC colors: a series is colored by what it MEANS (cluster id / model
    name), never by loop index. Use CLUSTER_COLORS / MODEL_STYLE, not PALETTE.
  - Vermillion is reserved for ERROR only. One neutral grey system for all
    non-data reference marks.
  - Greyscale + colorblind safety via non-color cues: cluster line styles,
    model marker shapes + dash patterns, panel position, bold for errors.
"""

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# ----------------------------------------------------------------------
# Named color system (use these everywhere; never hardcode a hex in a script)
# ----------------------------------------------------------------------
# Clusters: semantic, mapped by integer label (NOT loop index).
CLUSTER_COLORS = {0: "#0072B2",  # blue   - moderate-growth majority
                  1: "#E69F00",  # orange - fast-growth
                  2: "#CC79A7"}  # purple - early-plateau minority (pops in scatter)
# Distinct line styles / markers give clusters a greyscale-safe second channel.
CLUSTER_LS = {0: "-", 1: (0, (5, 2)), 2: (0, (1, 1.4))}      # solid / dashed / dotted
CLUSTER_MARKER = {0: "o", 1: "s", 2: "D"}

# Models: mapped by NAME, each with a distinct color, marker shape AND dash
# pattern so identity survives greyscale and the colorblind confusion axes.
MODEL_STYLE = {
    "tabnet":         {"color": "#0072B2", "marker": "o", "ls": "-"},
    "gbdt":           {"color": "#E69F00", "marker": "s", "ls": (0, (5, 2))},
    "extra_trees":    {"color": "#009E73", "marker": "^", "ls": (0, (3, 1, 1, 1))},
    "ft_transformer": {"color": "#CC79A7", "marker": "D", "ls": (0, (1, 1.2))},
    "tabm":           {"color": "#56B4E9", "marker": "v", "ls": "-"},
    "tabmixer":       {"color": "#000000", "marker": "P", "ls": (0, (4, 1, 1, 1))},
    "tabkan":         {"color": "#6E6E6E", "marker": "X", "ls": (0, (1, 1))},
}
MODEL_LABEL = {
    "tabnet": "TabNet", "gbdt": "GBDT", "extra_trees": "ExtraTrees",
    "ft_transformer": "FT-Transformer", "tabm": "TabM",
    "tabmixer": "TabMixer", "tabkan": "TabKAN",
}

ERROR_COLOR = "#D55E00"   # vermillion: misclassification ONLY (never a data category)
REF_COLOR = "#6E6E6E"     # single neutral grey: medians, baselines, annotations
REF_DARK = "#3F3F3F"      # darker grey: ONLY the predicted-cluster median in error fig
GRID_COLOR = "#DDDDDD"

# Encoding-consistent alpha values.
BAND_ALPHA = 0.18         # percentile / std bands
BOX_ALPHA = 0.55          # boxplot fills
SCATTER_ALPHA = 0.28      # dense scatter points

ARROW = "\u2192"     # right arrow; present in Liberation Sans and DejaVu Sans

# Backwards-compatible categorical palette (legacy; prefer the named maps above).
PALETTE = ["#0072B2", "#E69F00", "#CC79A7", "#009E73", "#56B4E9", "#000000"]

_FONTS_REGISTERED = False


def _register_fonts():
    """Register Liberation Sans (all weights) so it is available by name."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    import glob
    for path in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf"):
        try:
            _fm.fontManager.addfont(path)
        except Exception:
            pass
    _FONTS_REGISTERED = True


def set_paper_style():
    """Apply the journal style. Greyscale-safe; Liberation Sans + DejaVu fallback."""
    _register_fonts()
    plt.rcParams.update({
        # Fonts. mathtext stays on dejavusans (a real math font) to avoid the
        # custom+Liberation fallback pitfall; body text is Liberation Sans.
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.0,
        # Lines / markers
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.0,
        # Axes: full box, thin black spines
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.axisbelow": True,
        # Grid: y-only light grey by default (figures override to 'both' as needed)
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        # Ticks: short, outward
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "black",
        "ytick.color": "black",
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.4,
        # Output
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_fig(fig, path_stem, formats=("pdf", "png")):
    """Save to multiple formats from a path stem. Relies on the figure's own
    (constrained/tight) layout; does not force bbox='tight' to avoid clipping
    out-of-axes legends under constrained_layout."""
    from pathlib import Path
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path_stem.with_suffix(f".{fmt}"))
    plt.close(fig)


def model_label(key):
    return MODEL_LABEL.get(key, key)
