"""Matplotlib plot canvas: drawing, drag/rotate interaction, and animation.

The canvas is a pure *view*: it never computes paths. On every model
notification it re-reads ``model.solutions`` / ``model.highlighted`` and redraws.
User gestures (drag, rotate, keyboard nudge) only ever call
``model.update(...)`` — the model re-solves and notifies, which drives the
redraw. This keeps the model the single source of truth.

The object-oriented matplotlib API (:class:`~matplotlib.figure.Figure` plus
:class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`) is used instead
of ``pyplot`` so importing this module does not select a global backend or open
a window.
"""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from tkinter import ttk
from typing import TYPE_CHECKING

from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,  # pyright: ignore[reportPrivateImportUsage]  # public API; stub omits it
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow

from dubins_demo.core.angles import normalize
from dubins_demo.core.dubins import Config, DubinsPath, PathType, turning_centers
from dubins_demo.core.model import Scenario
from dubins_demo.ui import theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.backend_bases import MouseEvent

# Colorblind-safe subset of the matplotlib tab10 palette. Green and red are
# deliberately reserved for the start/goal arrows, so no path shares them.
_PATH_COLORS: dict[PathType, str] = {
    PathType.LSL: "#1f77b4",  # blue
    PathType.RSR: "#ff7f0e",  # orange
    PathType.LSR: "#9467bd",  # purple
    PathType.RSL: "#8c564b",  # brown
    PathType.RLR: "#e377c2",  # pink
    PathType.LRL: "#17becf",  # cyan
}

_GRAB_PX = 12.0  # hit-test radius around an arrow base/head, in display pixels
_ARROW_CAP = 5.0  # maximum arrow length in meters (length is otherwise ∝ radius)
_ARROW_FLOOR = 0.3  # minimum arrow length so it stays visible for tiny radii
_ANIM_STEP = 0.05  # sampling step (m) used for the animation marker path
_NUDGE_M = 0.1  # keyboard position nudge (meters)
_NUDGE_RAD = math.radians(1.0)  # keyboard heading nudge (radians)


class PlotCanvas:
    """Interactive matplotlib canvas bound to a :class:`Scenario` model."""

    def __init__(
        self,
        parent: tk.Misc,
        model: Scenario,
        status_sink: Callable[[str], None],
    ) -> None:
        """Build the canvas, toolbar, and animation controls under ``parent``.

        ``status_sink`` receives short strings for the status bar (mouse
        coordinates, warnings). The widget tree is exposed as :attr:`frame` for
        the parent to lay out.
        """
        self.model = model
        self._status = status_sink

        self.frame = ttk.Frame(parent)

        self.figure = Figure(figsize=(6.0, 5.0), layout=None)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        theme.style_axes(self.figure, self.ax)
        # Reserve room on the right for the legend placed outside the axes.
        self.figure.subplots_adjust(left=0.10, right=0.78, top=0.96, bottom=0.08)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        canvas_widget = self.canvas.get_tk_widget()

        controls = ttk.Frame(self.frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame, pack_toolbar=False)

        controls.pack(side="bottom", fill="x")
        self.toolbar.pack(side="bottom", fill="x")
        canvas_widget.pack(side="top", fill="both", expand=True)

        self._build_controls(controls)

        # --- artist bookkeeping (managed so redraws never disturb zoom) -----
        self._path_lines: list[Line2D] = []
        self._patches: list[FancyArrow | Circle] = []
        self._legend = None
        self._marker: Line2D | None = None
        # Data-space arrow geometry, refreshed each redraw for hit-testing:
        # name -> (base_x, base_y, head_x, head_y).
        self._config_points: dict[str, tuple[float, float, float, float]] = {}

        # --- view state -----------------------------------------------------
        self._auto_fit = True
        self._programmatic_limits = False
        self.ax.callbacks.connect("xlim_changed", self._on_limits_changed)
        self.ax.callbacks.connect("ylim_changed", self._on_limits_changed)

        # --- interaction state ---------------------------------------------
        self._selected: str | None = None  # "start" | "goal" | None
        self._drag: tuple[str, str] | None = None  # (config_name, "move"|"rotate")

        # --- animation state -----------------------------------------------
        self._anim: FuncAnimation | None = None
        self._anim_points = None
        self._playing = False
        # Current marker frame; lives on the instance so it survives pause →
        # resume. FuncAnimation just ticks; the position is read here, not from
        # the ``frame`` it passes in. Reset to 0 only on a full stop.
        self._frame = 0

        self._connect_events(canvas_widget)
        self.model.add_listener(self._on_model_changed)
        self._redraw()

    # -- widget construction -------------------------------------------------

    def _build_controls(self, controls: ttk.Frame) -> None:
        self._play_button = ttk.Button(
            controls,
            text="▶",
            width=3,
            style=theme.PRIMARY_BUTTON,
            command=self._toggle_animation,
        )
        self._play_button.pack(side="left", padx=(0, theme.PAD_M), pady=theme.PAD_S)

        ttk.Label(controls, text="Speed (m/s):").pack(side="left", padx=(0, theme.PAD_S))
        self._speed_var = tk.StringVar(value=f"{self.model.animation_speed:g}")
        speed_entry = ttk.Entry(controls, textvariable=self._speed_var, width=7)
        speed_entry.pack(side="left", padx=(0, theme.PAD_M))
        speed_entry.bind("<FocusOut>", self._commit_speed)
        speed_entry.bind("<Return>", self._commit_speed)

        self._circles_var = tk.BooleanVar(value=self.model.show_circles)
        ttk.Checkbutton(
            controls,
            text="Turning circles",
            variable=self._circles_var,
            command=self._toggle_circles,
        ).pack(side="left", padx=(0, theme.PAD_S))

    def _connect_events(self, canvas_widget: tk.Widget) -> None:
        # matplotlib types the callback as (Event) -> Any; our handlers narrow to
        # MouseEvent (a subclass), which the stub rejects on contravariance.
        self.canvas.mpl_connect("button_press_event", self._on_press)  # pyright: ignore[reportArgumentType]
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)  # pyright: ignore[reportArgumentType]
        self.canvas.mpl_connect("button_release_event", self._on_release)  # pyright: ignore[reportArgumentType]

        canvas_widget["takefocus"] = True
        canvas_widget.bind("<Left>", lambda _e: self._nudge(-_NUDGE_M, 0.0, 0.0))
        canvas_widget.bind("<Right>", lambda _e: self._nudge(_NUDGE_M, 0.0, 0.0))
        canvas_widget.bind("<Up>", lambda _e: self._nudge(0.0, _NUDGE_M, 0.0))
        canvas_widget.bind("<Down>", lambda _e: self._nudge(0.0, -_NUDGE_M, 0.0))
        canvas_widget.bind("<Shift-Left>", lambda _e: self._nudge(0.0, 0.0, _NUDGE_RAD))
        canvas_widget.bind("<Shift-Right>", lambda _e: self._nudge(0.0, 0.0, -_NUDGE_RAD))
        canvas_widget.bind("<Shift-Up>", lambda _e: self._nudge(0.0, 0.0, _NUDGE_RAD))
        canvas_widget.bind("<Shift-Down>", lambda _e: self._nudge(0.0, 0.0, -_NUDGE_RAD))

    # -- model notification --------------------------------------------------

    def _on_model_changed(self) -> None:
        # Any scenario change invalidates a running animation (FR-15 reset).
        self._stop_animation()
        self._redraw()

    # -- drawing -------------------------------------------------------------

    def _radius(self) -> float:
        return self.model.radius_policy.min_radius()

    def _arrow_length(self, radius: float) -> float:
        """Arrow length in meters: proportional to ``radius`` but clamped so it
        stays visible for tiny radii and never dominates for large ones."""
        return min(max(radius, _ARROW_FLOOR), _ARROW_CAP)

    def _clear_artists(self) -> None:
        for line in self._path_lines:
            line.remove()
        self._path_lines.clear()
        for patch in self._patches:
            patch.remove()
        self._patches.clear()
        if self._legend is not None:
            self._legend.remove()
            self._legend = None

    def _redraw(self) -> None:
        self._clear_artists()
        radius = self._radius()

        all_xy: list[tuple[float, float]] = []
        for path_type, solution in self.model.solutions.items():
            if not isinstance(solution, DubinsPath):
                continue
            samples = solution.sample()
            highlighted = path_type is self.model.highlighted
            line = Line2D(
                samples[:, 0],
                samples[:, 1],
                color=_PATH_COLORS[path_type],
                linewidth=3.0 if highlighted else 1.5,
                alpha=1.0 if highlighted else 0.35,
                label=path_type.value,
                zorder=4 if highlighted else 3,
            )
            self.ax.add_line(line)
            self._path_lines.append(line)
            all_xy.extend((float(x), float(y)) for x, y in samples[:, :2])

        self._draw_arrow(self.model.start, radius, "#2ca02c", "start", all_xy)
        self._draw_arrow(self.model.goal, radius, "#d62728", "goal", all_xy)

        if self.model.show_circles:
            self._draw_circles(radius)

        if self._selected is not None:
            self._draw_selection(self._selected, radius)

        if self._path_lines:
            self._legend = self.ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize="small",
                title="Path type",
            )
            theme.style_legend(self._legend)

        self._fit_view(all_xy)
        self.canvas.draw_idle()

    def _draw_arrow(
        self,
        cfg: Config,
        radius: float,
        color: str,
        name: str,
        all_xy: list[tuple[float, float]],
    ) -> None:
        length = self._arrow_length(radius)
        dx = length * math.cos(cfg.theta)
        dy = length * math.sin(cfg.theta)
        arrow = FancyArrow(
            cfg.x,
            cfg.y,
            dx,
            dy,
            width=length * 0.05,
            head_width=length * 0.24,
            head_length=length * 0.32,
            length_includes_head=True,
            color=color,
            zorder=5,
        )
        self.ax.add_patch(arrow)
        self._patches.append(arrow)
        self._config_points[name] = (cfg.x, cfg.y, cfg.x + dx, cfg.y + dy)
        all_xy.append((cfg.x, cfg.y))
        all_xy.append((cfg.x + dx, cfg.y + dy))

    def _draw_circles(self, radius: float) -> None:
        for cfg in (self.model.start, self.model.goal):
            for center in turning_centers(cfg, radius):
                circle = Circle(
                    center,
                    radius,
                    fill=False,
                    linestyle="--",
                    edgecolor=theme.SPINE,
                    linewidth=1.0,
                    zorder=2,
                )
                self.ax.add_patch(circle)
                self._patches.append(circle)

    def _draw_selection(self, name: str, radius: float) -> None:
        cfg = getattr(self.model, name)
        length = self._arrow_length(radius)
        outline = Circle(
            (cfg.x, cfg.y),
            length * 0.30,
            fill=False,
            linestyle=":",
            edgecolor="#333333",
            linewidth=1.2,
            zorder=6,
        )
        self.ax.add_patch(outline)
        self._patches.append(outline)

    # -- auto-fit view -------------------------------------------------------

    def _on_limits_changed(self, _ax: object) -> None:
        # A limit change we did not initiate means the user zoomed/panned.
        if not self._programmatic_limits:
            self._auto_fit = False

    def _fit_view(self, all_xy: list[tuple[float, float]]) -> None:
        if self._auto_fit:
            self._apply_fit(all_xy)
            return
        # Respect manual zoom until a configuration leaves the visible area.
        if self._configs_out_of_view():
            self._auto_fit = True
            self._apply_fit(all_xy)

    def _apply_fit(self, all_xy: list[tuple[float, float]]) -> None:
        if not all_xy:
            return
        xs = [x for x, _ in all_xy]
        ys = [y for _, y in all_xy]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        span = max(xmax - xmin, ymax - ymin, 2.0 * self._radius())
        pad = 0.10 * span
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * span + pad
        self._programmatic_limits = True
        try:
            self.ax.set_xlim(cx - half, cx + half)
            self.ax.set_ylim(cy - half, cy + half)
        finally:
            self._programmatic_limits = False

    def _configs_out_of_view(self) -> bool:
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        for cfg in (self.model.start, self.model.goal):
            if not (xmin <= cfg.x <= xmax and ymin <= cfg.y <= ymax):
                return True
        return False

    # -- mouse interaction ---------------------------------------------------

    def _hit_test(self, event: MouseEvent) -> tuple[str, str] | None:
        """Return ``(config_name, mode)`` for the closest grabbed handle."""
        if event.x is None or event.y is None:
            return None
        best: tuple[str, str] | None = None
        best_dist = _GRAB_PX
        for name, (bx, by, hx, hy) in self._config_points.items():
            for mode, (px, py) in (("move", (bx, by)), ("rotate", (hx, hy))):
                sx, sy = self.ax.transData.transform((px, py))
                dist = math.hypot(sx - event.x, sy - event.y)
                if dist <= best_dist:
                    best_dist = dist
                    best = (name, mode)
        return best

    def _on_press(self, event: MouseEvent) -> None:
        self.canvas.get_tk_widget().focus_set()
        if event.inaxes is not self.ax or event.button != 1:
            return
        hit = self._hit_test(event)
        if hit is None:
            return
        name, _mode = hit  # mode is resolved again on drag; only the name is needed here
        self._drag = hit
        self._selected = name
        self._stop_animation()
        self._redraw()  # show the selection cue immediately

    def _on_motion(self, event: MouseEvent) -> None:
        if event.inaxes is self.ax and event.xdata is not None:
            self._status(f"x = {event.xdata:.3f} m   y = {event.ydata:.3f} m")
        if self._drag is None or event.x is None:
            return
        name, mode = self._drag
        inv = self.ax.transData.inverted()
        data_x, data_y = inv.transform((event.x, event.y))
        # A degenerate axes state can make the inverse transform yield nan/inf;
        # Config would reject those, so ignore the gesture rather than let the
        # ValueError escape this matplotlib callback (it would only hit stderr).
        if not (math.isfinite(data_x) and math.isfinite(data_y)):
            return
        cfg = getattr(self.model, name)
        if mode == "move":
            new_cfg = Config(data_x, data_y, cfg.theta)
        else:  # rotate: heading points from the arrow base toward the cursor
            heading = math.atan2(data_y - cfg.y, data_x - cfg.x)
            new_cfg = Config(cfg.x, cfg.y, normalize(heading))
        self.model.update(**{name: new_cfg})
        self.canvas.draw_idle()

    def _on_release(self, _event: MouseEvent) -> None:
        self._drag = None

    def _nudge(self, dx: float, dy: float, dtheta: float) -> None:
        if self._selected is None:
            self._status("Click a start/goal arrow first to nudge it.")
            return
        cfg = getattr(self.model, self._selected)
        new_cfg = Config(cfg.x + dx, cfg.y + dy, normalize(cfg.theta + dtheta))
        self.model.update(**{self._selected: new_cfg})

    # -- overlay toggles -----------------------------------------------------

    def _toggle_circles(self) -> None:
        self.model.update(show_circles=self._circles_var.get())

    # -- animation -----------------------------------------------------------

    def _commit_speed(self, _event: object = None) -> bool:
        """Parse and store the playback speed; return whether it was valid."""
        try:
            speed = float(self._speed_var.get())
        except ValueError:
            self._status("Speed must be a number (m/s).")
            return False
        # ``nan <= 0`` is False, so reject non-finite values explicitly.
        if not math.isfinite(speed) or speed <= 0.0:
            self._status("Speed must be a positive, finite number (m/s).")
            return False
        # Speed is playback-only state: use the dedicated setter so it does not
        # re-solve or fire the notify that would stop a running animation.
        self.model.set_animation_speed(speed)
        self._speed_var.set(f"{speed:g}")
        # Apply the new speed live: rebuild the running animation so its
        # interval is recomputed. _start_animation preserves _frame, so the
        # marker keeps its position rather than jumping back to the start.
        # Clear _playing first so the re-entrant _commit_speed inside
        # _start_animation does not recurse; _start_animation re-sets it.
        if self._playing:
            self._playing = False
            self._start_animation()
        return True

    def _toggle_animation(self) -> None:
        if self._playing:
            self._pause_animation()
        else:
            self._start_animation()

    def _start_animation(self) -> None:
        highlighted = self.model.highlighted
        if highlighted is None:
            self._status("No feasible path to animate.")
            return
        solution = self.model.solutions[highlighted]
        if not isinstance(solution, DubinsPath):
            return
        if not self._commit_speed():
            return
        speed = max(self.model.animation_speed, 1e-6)
        points = solution.sample(_ANIM_STEP)
        interval = max(10.0, (_ANIM_STEP / speed) * 1000.0)
        self._anim_points = points
        # Resume from the paused frame; a genuine model change routes through
        # _stop_animation, which has already reset _frame to 0. Guard the index
        # in case the (re-sampled) path is shorter than the preserved frame.
        self._frame %= len(points)
        # Discard any prior animation (a paused instance, or the one being
        # replaced by a live speed change) so a stale event source can never
        # keep ticking alongside the new one.
        if self._anim is not None:
            if self._anim.event_source is not None:
                self._anim.event_source.stop()
            self._anim = None
        self._anim = FuncAnimation(
            self.figure,
            self._animate,
            frames=len(points),
            interval=interval,
            blit=False,
            repeat=True,
            cache_frame_data=False,
        )
        self._playing = True
        self._play_button.configure(text="⏸")
        self.canvas.draw_idle()

    def _pause_animation(self) -> None:
        if self._anim is not None and self._anim.event_source is not None:
            self._anim.event_source.stop()
        self._playing = False
        self._play_button.configure(text="▶")

    def _stop_animation(self) -> None:
        if self._anim is not None:
            if self._anim.event_source is not None:
                self._anim.event_source.stop()
            self._anim = None
        self._anim_points = None
        self._playing = False
        self._frame = 0  # full reset (FR-15): next play starts from the beginning
        if self._marker is not None:
            self._marker.remove()
            self._marker = None
        self._play_button.configure(text="▶")

    def _animate(self, _frame: int) -> tuple[Line2D, ...]:
        # The frame index lives on the instance (see _frame) so it survives
        # pause → resume; FuncAnimation's own counter is ignored deliberately.
        points = self._anim_points
        if points is None:
            return ()
        x, y, theta = points[self._frame]
        # A (3, 0, angle) marker is a triangle; angle 0 points up (+Y), so
        # subtract 90° to align the tip with the math-convention heading.
        marker = (3, 0, math.degrees(theta) - 90.0)
        if self._marker is None:
            (self._marker,) = self.ax.plot(
                [x], [y], marker=marker, markersize=13, color="#111111", zorder=7
            )
        else:
            self._marker.set_data([x], [y])
            # (numsides, style, angle) tuple is a valid marker but absent from MarkerType.
            self._marker.set_marker(marker)  # pyright: ignore[reportArgumentType]
        self._frame = (self._frame + 1) % len(points)
        return (self._marker,)
