"""Shared matplotlib styling for every chart in this project.

Centralising the palette and axis treatment means the notebooks produce
charts that look like they belong to the same piece of work, rather than
matplotlib defaults. Import this instead of restyling in each notebook.

Serverless note: matplotlib ships with the runtime. seaborn is deliberately
not used — its default theme is recognisable and adds a dependency for
styling we can do in 40 lines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Palette — Databricks brand neutrals with a single accent, so highlighted
# elements read as deliberate rather than decorative.
INK = "#1B3139"     # primary series, titles
ACCENT = "#FF3621"  # the one thing the reader should look at
MUTED = "#9AA5A9"   # context series, secondary bars
GRID = "#E3E8EA"
POSITIVE = "#C0392B"  # delayed
NEGATIVE = "#2E86AB"  # on time

FIGSIZE_WIDE = (11, 4.5)
FIGSIZE_HALF = (5.5, 4.0)


def new_axes(figsize=FIGSIZE_WIDE):
    """Figure + axes with the house treatment already applied."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, style(ax)


def style(ax, title=None, xlabel=None, ylabel=None, percent_y=False, thousands_y=False):
    """Strip chartjunk, keep a horizontal grid, left-align the title."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9, length=0)

    if title:
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if percent_y:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    if thousands_y:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    return ax


def highlight_colors(values, rule):
    """Accent the bars that satisfy `rule`, mute the rest.

    Encodes the finding in the colour instead of leaving the reader to
    find it: `highlight_colors(rates, lambda v: v > rates.mean())`.
    """
    return [ACCENT if rule(v) else MUTED for v in values]


def annotate_bars(ax, bars, labels, fontsize=8):
    """Value labels above bars — removes the need to read against the axis."""
    for bar, label in zip(bars, labels):
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=INK,
        )
