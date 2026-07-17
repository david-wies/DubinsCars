"""Input panel: coordinate/heading entries, display toggles, radius sub-frame.

The panel formats the model's canonical values (radians, math convention,
meters) for display, and parses edits back into canonical form before calling
``model.update(...)``. The heading convention (angle/azimuth) and unit
(deg/rad) toggles affect *only* this formatting — the stored :class:`Config` is
never touched by a toggle.

As a model listener the panel rewrites its entry text whenever a change
originates elsewhere (a drag on the plot, a keyboard nudge, a file load),
skipping whichever widget currently holds focus so it never clobbers an
in-progress edit.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from dataclasses import replace
from tkinter import ttk

from dubins_demo.core.angles import (
    angle_to_azimuth,
    azimuth_to_angle,
    deg_to_rad,
    normalize,
    rad_to_deg,
)
from dubins_demo.core.dubins import Config
from dubins_demo.core.model import Convention, FixedRadius, Scenario, Unit

_ERROR_BG = "#ffcccc"  # light red so the (dark) entry text stays readable
_RADIUS_MIN = 0.1
_RADIUS_MAX = 50.0


def _fmt(value: float) -> str:
    """Format a float for an entry field with trimmed, human-friendly digits."""
    return f"{value:.6g}"


class InputPanel:
    """Start/goal entry fields, convention/unit toggles, and the radius frame."""

    #: entry key -> (config attribute name "start"/"goal", field "x"/"y"/"h")
    _FIELDS = {
        "start_x": ("start", "x"),
        "start_y": ("start", "y"),
        "start_h": ("start", "h"),
        "goal_x": ("goal", "x"),
        "goal_y": ("goal", "y"),
        "goal_h": ("goal", "h"),
    }

    def __init__(
        self,
        parent: tk.Misc,
        model: Scenario,
        status_sink: Callable[[str], None],
    ) -> None:
        """Build the panel under ``parent`` and subscribe it to ``model``."""
        self.model = model
        self._status = status_sink
        self.frame = ttk.Frame(parent, padding=6)

        self._refreshing = False
        self._entries: dict[str, tk.Entry] = {}

        self._build_config_group("Start", "start")
        self._build_config_group("Goal", "goal")
        self._build_display_options()

        self._radius = _FixedRadiusFrame(self.frame, model, status_sink)
        self._radius.frame.pack(fill="x", pady=(8, 0))

        self._normal_bg = self._entries["start_x"].cget("background")

        self.model.add_listener(self._on_model_changed)
        self._on_model_changed()

    # -- widget construction -------------------------------------------------

    def _build_config_group(self, title: str, which: str) -> None:
        group = ttk.LabelFrame(self.frame, text=title, padding=6)
        group.pack(fill="x", pady=(0, 6))
        labels = (("X [m]", "x"), ("Y [m]", "y"), ("Heading", "h"))
        for row, (label, field) in enumerate(labels):
            ttk.Label(group, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            key = f"{which}_{field}"
            entry = tk.Entry(group, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=1)
            entry.bind("<Return>", lambda _e, k=key: self._commit(k))
            entry.bind("<FocusOut>", lambda _e, k=key: self._commit(k))
            self._entries[key] = entry
        group.columnconfigure(1, weight=1)

    def _build_display_options(self) -> None:
        options = ttk.LabelFrame(self.frame, text="Heading display", padding=6)
        options.pack(fill="x", pady=(0, 6))

        self._conv_var = tk.StringVar(value=self.model.heading_convention.value)
        ttk.Label(options, text="Convention:").grid(row=0, column=0, sticky="w")
        for col, conv in enumerate(Convention, start=1):
            ttk.Radiobutton(
                options,
                text=conv.value,
                value=conv.value,
                variable=self._conv_var,
                command=self._commit_convention,
            ).grid(row=0, column=col, sticky="w", padx=2)

        self._unit_var = tk.StringVar(value=self.model.angle_unit.value)
        ttk.Label(options, text="Unit:").grid(row=1, column=0, sticky="w")
        for col, unit in enumerate(Unit, start=1):
            ttk.Radiobutton(
                options,
                text=unit.value,
                value=unit.value,
                variable=self._unit_var,
                command=self._commit_unit,
            ).grid(row=1, column=col, sticky="w", padx=2)

    # -- display <-> canonical conversion ------------------------------------

    def _heading_to_display(self, theta: float) -> float:
        if self._convention() is Convention.AZIMUTH:
            value = angle_to_azimuth(theta)
        else:
            value = normalize(theta)
        return rad_to_deg(value) if self._unit() is Unit.DEG else value

    def _heading_from_display(self, raw: float) -> float:
        radians = deg_to_rad(raw) if self._unit() is Unit.DEG else raw
        if self._convention() is Convention.AZIMUTH:
            return azimuth_to_angle(radians)
        return normalize(radians)

    def _display_value(self, key: str) -> str:
        _which, field = self._FIELDS[key]
        cfg: Config = getattr(self.model, _which)
        if field == "x":
            return _fmt(cfg.x)
        if field == "y":
            return _fmt(cfg.y)
        return _fmt(self._heading_to_display(cfg.theta))

    def _convention(self) -> Convention:
        return Convention(self._conv_var.get())

    def _unit(self) -> Unit:
        return Unit(self._unit_var.get())

    # -- edit commits --------------------------------------------------------

    def _commit(self, key: str) -> None:
        if self._refreshing:
            return
        entry = self._entries[key]
        try:
            raw = float(entry.get())
        except ValueError:
            entry.configure(background=_ERROR_BG)
            self._status(f"{key.replace('_', ' ')}: not a number — value unchanged.")
            return
        entry.configure(background=self._normal_bg)

        which, field = self._FIELDS[key]
        cfg: Config = getattr(self.model, which)
        if field == "x":
            new_cfg = replace(cfg, x=raw)
        elif field == "y":
            new_cfg = replace(cfg, y=raw)
        else:
            new_cfg = replace(cfg, theta=self._heading_from_display(raw))
        self.model.update(**{which: new_cfg})

    def _commit_convention(self) -> None:
        # A display change: re-solve is harmless, and the notification triggers
        # a reformat of every heading entry from the canonical model value.
        self.model.update(heading_convention=self._convention())

    def _commit_unit(self) -> None:
        self.model.update(angle_unit=self._unit())

    # -- model notification --------------------------------------------------

    def _on_model_changed(self) -> None:
        self._refreshing = True
        try:
            focused = self.frame.focus_get()
            self._conv_var.set(self.model.heading_convention.value)
            self._unit_var.set(self.model.angle_unit.value)
            for key, entry in self._entries.items():
                if entry is focused:
                    continue
                entry.configure(background=self._normal_bg)
                entry.delete(0, "end")
                entry.insert(0, self._display_value(key))
            self._radius.refresh(focused)
        finally:
            self._refreshing = False


class _FixedRadiusFrame:
    """Swappable radius sub-frame (EXT-2): a slider linked to an entry.

    Both widgets drive ``FixedRadius.value`` through ``model.update`` and are
    refreshed from the model on notification. Values outside ``[0.1, 50]`` are
    clamped (FR-3). This class is intentionally self-contained so a future
    speed/vehicle-parameter frame can replace it without touching the rest of
    the input panel.
    """

    def __init__(
        self,
        parent: tk.Misc,
        model: Scenario,
        status_sink: Callable[[str], None],
    ) -> None:
        self.model = model
        self._status = status_sink
        self._refreshing = False

        self.frame = ttk.LabelFrame(parent, text="Turn radius [m]", padding=6)

        self._scale = tk.Scale(
            self.frame,
            from_=_RADIUS_MIN,
            to=_RADIUS_MAX,
            resolution=0.1,
            orient="horizontal",
            showvalue=False,
            command=self._on_slide,
        )
        self._scale.grid(row=0, column=0, sticky="ew")

        self._entry = tk.Entry(self.frame, width=8)
        self._entry.grid(row=0, column=1, padx=(6, 0))
        self._entry.bind("<Return>", self._on_entry)
        self._entry.bind("<FocusOut>", self._on_entry)

        self.frame.columnconfigure(0, weight=1)
        self._normal_bg = self._entry.cget("background")

    def _current(self) -> float:
        return self.model.radius_policy.min_radius()

    def _apply(self, value: float) -> None:
        clamped = min(_RADIUS_MAX, max(_RADIUS_MIN, value))
        self.model.update(radius_policy=FixedRadius(clamped))

    def _on_slide(self, value: str) -> None:
        if self._refreshing:
            return
        self._apply(float(value))

    def _on_entry(self, _event: object = None) -> None:
        if self._refreshing:
            return
        try:
            value = float(self._entry.get())
        except ValueError:
            self._entry.configure(background=_ERROR_BG)
            self._status("Turn radius: not a number — value unchanged.")
            return
        self._entry.configure(background=self._normal_bg)
        self._apply(value)

    def refresh(self, focused: tk.Misc | None) -> None:
        """Rewrite the slider and entry from the model, skipping focused entry."""
        self._refreshing = True
        try:
            value = self._current()
            self._scale.set(value)
            if self._entry is not focused:
                self._entry.configure(background=self._normal_bg)
                self._entry.delete(0, "end")
                self._entry.insert(0, _fmt(value))
        finally:
            self._refreshing = False
