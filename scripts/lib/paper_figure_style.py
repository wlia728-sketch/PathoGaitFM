"""Shared publication style for the PathoGaitFM figure-redesign previews.

The cohort palette is semantic and fixed across figures:
TD=green, CP=orange, post-stroke=blue, PD=purple. Quantitative figures always
save with an opaque white background and editable TrueType text in PDF/SVG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

# One bar geometry for every panel in the paper. A group of bars always occupies the same fraction
# of its category slot, whatever the number of series, and a one-series panel borrows the two-series
# width so it does not read as a block beside the grouped panels.
GROUP_SPAN = 0.76
BAR_INSET = 0.10
BAR_ALPHA = 1.0           # the palette is already muted, so bars are drawn opaque


def bar_group(n_series: int, span: float = GROUP_SPAN):
    """Centred x offsets and one bar width for `n_series` grouped bars in a category slot."""
    slot = span / max(int(n_series), 1)
    offsets = (np.arange(int(n_series)) - (int(n_series) - 1) / 2.0) * slot
    return offsets, slot * (1.0 - BAR_INSET)


def single_bar_width(span: float = GROUP_SPAN) -> float:
    """Width for a one-series panel, matched to a two-series bar."""
    return float(bar_group(2, span)[1])


# One semantic colour system for the whole paper.
#   cohort hues        say which cohort, and appear only where cohorts are compared
#   method navy        says which arm the panel puts forward, and appears only in a method panel
#   control greys      say comparator or control, and never carry meaning of their own
# The cohort hues are a desaturated variant of the earlier Okabe-Ito set, same semantics and still
# separable under deuteranopia, but muted so that large filled bars stop competing with the titles.
COLORS = {
    "td": "#2A9D8F",              # typically developing, muted teal
    "cp": "#D69A2D",              # cerebral palsy, muted ochre
    "stroke": "#3E7FAE",          # post-stroke, desaturated blue
    "pd": "#B27A9E",              # Parkinson's disease, dusty mauve
    "macro": "#444A52",           # four-cohort macro, charcoal
    "ours": "#285C7A",            # primary method accent, deep desaturated navy
    "baseline": "#929CA7",        # control, medium grey
    "baseline_light": "#CDD3D9",  # control, light grey
    "control_edge": "#626B75",
    "emphasis_band": "#EAF1F6",   # the pale band that marks a focus group on an axis
    "text": "#22272E",
    "axis": "#343A40",
    "grid": "#E5E9ED",
    "muted": "#8A929B",           # secondary annotations and NA labels
    "subject": "#B7BEC5",
    "white": "#FFFFFF",
}

COHORT_COLORS = {
    "normal": COLORS["td"],
    "cp": COLORS["cp"],
    "vdk_stroke": COLORS["stroke"],
    "bmclab_pd": COLORS["pd"],
    "CP": COLORS["cp"],
    "STROKE": COLORS["stroke"],
}


# One typography scale for every quantitative main figure. The figures are drawn at the width they
# are inserted at, and exported on the full canvas rather than a cropped bounding box, so a point
# declared here is a point on the printed page and no figure is silently rescaled by Word.
FIG_WIDTH_IN = 6.5
FS = {
    "letter": 10.5,      # panel letters, bold
    "title": 9.5,        # panel titles
    "label": 8.5,        # axis titles
    "tick": 7.5,         # tick labels
    "tick_dense": 7.0,   # multi-line category names sharing a narrow panel
    "legend": 7.2,       # legend entries
    "value": 7.0,        # plotted value annotations
    "small": 6.5,        # the floor: nRMSE, sample sizes, NA marks, colourbar ends
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FS["tick"],
            "axes.titlesize": FS["title"],
            "axes.titleweight": "semibold",
            "axes.labelsize": FS["label"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": FS["tick"],
            "ytick.labelsize": FS["tick"],
            "xtick.color": COLORS["axis"],
            "ytick.color": COLORS["axis"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": FS["legend"],
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )


HEADING_PAD_PT = 8.0
LETTER_DX_PT = -13.0


def panel_heading(ax, letter: str, title: str, letter_dx: float = LETTER_DX_PT,
                  heading_dx: float = 0.0) -> None:
    """Panel letter and title at a fixed offset above the axes top edge.

    Both offsets are in points, not axes fractions, so panels of unequal width or height carry the
    letter at the same distance from their own top-left corner and the letters line up across
    figures. set_title already pads in points. Do not put tick labels on an axis top in a row that
    has to align, because set_title then lifts to clear them.

    heading_dx shifts the whole heading sideways, for the one case a fixed offset does not cover: a
    panel whose axes is inset from the figure margin to make room for long row labels, and whose
    letter would otherwise sit well right of the letters below it.
    """
    width_pt = ax.get_position().width * ax.figure.get_size_inches()[0] * 72.0
    ax.set_title(title, loc="left", pad=HEADING_PAD_PT, color=COLORS["text"],
                 x=heading_dx / width_pt if width_pt else 0.0)
    ax.annotate(
        letter,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(letter_dx + heading_dx, HEADING_PAD_PT),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=FS["letter"],
        fontweight="bold",
        color=COLORS["text"],
        annotation_clip=False,
    )


def light_x_grid(ax) -> None:
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def light_y_grid(ax) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def assert_within_canvas(fig, name: str, tolerance_in: float = 0.015) -> None:
    """Abort if anything is drawn outside the declared canvas.

    The exports below write the full canvas rather than a cropped bounding box, which is what keeps
    the declared point sizes identical to the printed ones. That only holds if nothing spills over
    the edge, so a spill has to widen the margins here rather than be cropped away at save time.
    """
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    over = (max(0.0, -tight.x0), max(0.0, -tight.y0),
            max(0.0, tight.x1 - width), max(0.0, tight.y1 - height))
    if max(over) > tolerance_in:
        raise AssertionError(
            "%s draws outside its %.2f x %.2f inch canvas by left %.3f, bottom %.3f, right %.3f, "
            "top %.3f inches. Widen the margins instead of letting the export crop."
            % (name, width, height, *over))


def save_triplet(fig, output_stem: Path) -> None:
    """Publication exports on the full canvas, so the saved width is the declared width."""
    assert_within_canvas(fig, output_stem.name)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            output_stem.with_suffix(f".{extension}"),
            dpi=600 if extension == "png" else None,
            bbox_inches=None,
            facecolor="white",
            transparent=False,
        )


def save_docx_png(fig, output_stem: Path) -> None:
    """Save a 300-dpi PNG for Word embedding without altering publication exports."""
    fig.savefig(
        output_stem.with_name(output_stem.name + "_docx").with_suffix(".png"),
        dpi=300,
        bbox_inches=None,
        facecolor="white",
        transparent=False,
    )


__all__ = [
    "COLORS",
    "COHORT_COLORS",
    "FIG_WIDTH_IN",
    "FS",
    "apply_style",
    "assert_within_canvas",
    "panel_heading",
    "light_x_grid",
    "light_y_grid",
    "save_triplet",
    "save_docx_png",
]
