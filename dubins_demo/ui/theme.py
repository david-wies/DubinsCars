"""Shared UI design system: tokens, ttk style tweaks, and matplotlib styling.

This module is the single source of truth for the app's *look*. It holds the
light-mode color palette, a spacing scale, and two runtime appliers:

- :func:`apply_theme` — sets the ``sv_ttk`` light theme and layers a handful of
  custom ``ttk`` styles on top (card frames, section headings, a flat status
  bar, an accent primary button).
- :func:`style_axes` — restyles a matplotlib ``Figure``/``Axes`` pair to match
  the Tk surface (white canvas, hairline grid, no top/right spines, muted
  ticks) so the plot reads as part of the same modern, light interface.

Import-time is side-effect free: no Tk root is created and no theme is applied
until these functions are called, keeping headless imports safe.
"""

from __future__ import annotations

from tkinter import Menu, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

# -- color palette (light mode, tuned to sv_ttk "light") ---------------------

#: Window/base background used by the sv_ttk light theme.
BG = "#fafafa"
#: Raised surface (entries, cards, the plot canvas).
SURFACE = "#ffffff"
#: Primary text.
TEXT = "#1c1c1c"
#: Secondary/label text and axis ticks.
MUTED = "#5f5f5f"
#: sv_ttk light accent (primary actions, focus).
ACCENT = "#005fb8"
#: Hairline borders and dividers.
BORDER = "#e2e2e2"
#: Plot gridlines.
GRID = "#ececec"
#: Axis sp(left/bottom) color.
SPINE = "#cfcfcf"

# -- spacing scale (px) ------------------------------------------------------

PAD_S = 4
PAD_M = 8
PAD_L = 12
PAD_XL = 16

# -- typography --------------------------------------------------------------

#: Preferred families for matplotlib text, best first. Only the fonts actually
#: installed are handed to matplotlib (see :func:`_resolve_font_family`), so no
#: "font not found" warnings are emitted on systems missing the nicer faces.
PLOT_FONT_STACK = ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]

# ttk style names layered on top of sv_ttk. Views reference these by name.
CARD_FRAME = "Card.TLabelframe"
CARD_LABEL = "Card.TLabelframe.Label"
HEADING_LABEL = "Heading.TLabel"
MUTED_LABEL = "Muted.TLabel"
STATUS_LABEL = "Status.TLabel"
PRIMARY_BUTTON = "Accent.TButton"  # provided by sv_ttk; named here for discoverability


def apply_theme(root: object) -> None:
    """Apply the light theme and register the app's custom ttk styles.

    Call once, after the Tk root and all widgets exist. ``root`` is accepted for
    call-site clarity (and future per-root theming) though sv_ttk themes the
    default root.
    """
    import sv_ttk  # pyright: ignore[reportMissingImports]  # no bundled type stubs

    sv_ttk.set_theme("light")

    style = ttk.Style()

    # Section headings inside panels: slightly larger, semibold, tight.
    base = style.lookup("TLabel", "font") or "TkDefaultFont"
    style.configure(HEADING_LABEL, foreground=TEXT)
    style.configure(MUTED_LABEL, foreground=MUTED)

    # Flat status bar: no sunken 3-D relief (a dated Motif cue); a thin top
    # divider separates it from the content instead.
    style.configure(STATUS_LABEL, foreground=MUTED, background=BG, padding=(PAD_L, PAD_S))

    # Card-like grouped frames with a readable, colored title.
    style.configure(CARD_LABEL, foreground=MUTED)

    _ = base  # font stack is owned by sv_ttk; kept for potential heading sizing


def _resolve_font_family() -> list[str]:
    """Return the installed subset of :data:`PLOT_FONT_STACK`, plus a fallback.

    matplotlib warns once per glyph for every named family it cannot find, so
    only installed families are forwarded; ``"sans-serif"`` always trails as a
    guaranteed fallback.
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    resolved = [name for name in PLOT_FONT_STACK if name in available]
    resolved.append("sans-serif")
    return resolved


def style_axes(figure: Figure, ax: Axes) -> None:
    """Restyle a matplotlib figure/axes to match the light Tk surface.

    Idempotent: safe to call once at construction. Data artists are untouched;
    only the frame, grid, ticks, and background are modernized.
    """
    import matplotlib as mpl

    mpl.rcParams["font.family"] = _resolve_font_family()
    mpl.rcParams["font.size"] = 9

    figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(1.0)

    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(MUTED)


def style_menu(menu: Menu) -> None:
    """Recolor a ``tk.Menu`` to the light palette (sv_ttk leaves menus native).

    Tk draws the menubar and its dropdowns itself on X11, so — unlike ttk
    widgets — their colors must be set directly. Flat chrome, an accent hover,
    and muted disabled text bring the classic gray Motif menu in line with the
    rest of the interface. Apply to the menubar and every submenu.
    """
    menu.configure(
        background=SURFACE,
        foreground=TEXT,
        activebackground=ACCENT,
        activeforeground=SURFACE,
        disabledforeground=MUTED,
        activeborderwidth=0,
        borderwidth=0,
        relief="flat",
    )


def style_legend(legend: Legend | None) -> None:
    """Give a matplotlib legend a light, low-chrome frame."""
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor(SURFACE)
    frame.set_edgecolor(BORDER)
    frame.set_linewidth(0.8)
    title = legend.get_title()
    if title is not None:
        title.set_color(MUTED)
