"""Main application window: layout, menus, status bar, and file I/O wiring.

``App`` owns the Tk root, builds the model-backed views (input panel, details
panel, plot canvas), the menu bar, and the status bar. Panels never import one
another; they communicate only through the shared :class:`Scenario` model and a
``StatusSink`` callable this class hands them.

A Tk root is created only when ``App`` is constructed, never at import time, so
importing this module is side-effect free and headless-safe.
"""

from __future__ import annotations

import importlib.resources
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import sv_ttk

from dubins_demo.core.dubins import DubinsPath
from dubins_demo.core.model import Scenario
from dubins_demo.persistence.scenario_io import (
    ScenarioError,
    export_waypoints_csv,
    load_scenario,
    save_scenario,
)
from dubins_demo.ui.details_panel import DetailsPanel
from dubins_demo.ui.input_panel import InputPanel
from dubins_demo.ui.plot_canvas import PlotCanvas

_EXPORT_LABEL = "Export CSV…"
_DEFAULT_STEP = 0.05


class App:
    """The Dubins Path Demonstrator main window."""

    def __init__(self, model: Scenario) -> None:
        """Create the root window and wire the model to all three panels."""
        self.model = model
        self.root = tk.Tk()
        self.root.title("Dubins Path Demonstrator")
        self.root.geometry("1100x720")

        self._status_var = tk.StringVar(value="Ready.")

        self._build_menu()
        self._build_layout()

        self.model.add_listener(self._on_model_changed)
        self._on_model_changed()

        sv_ttk.set_theme("light")

    # -- layout --------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        self._file_menu = tk.Menu(menubar, tearoff=False)
        self._file_menu.add_command(label="Save…", command=self._on_save)
        self._file_menu.add_command(label="Load…", command=self._on_load)
        self._file_menu.add_command(label=_EXPORT_LABEL, command=self._on_export_csv)
        self._file_menu.add_separator()
        self._file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=self._file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Open help page", command=self._on_help)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="ns")
        left.rowconfigure(1, weight=1)

        self.input_panel = InputPanel(left, self.model, self._set_status)
        self.input_panel.frame.grid(row=0, column=0, sticky="new")
        self.details_panel = DetailsPanel(left, self.model, self._set_status)
        self.details_panel.frame.grid(row=1, column=0, sticky="nsew")

        self.plot_canvas = PlotCanvas(self.root, self.model, self._set_status)
        self.plot_canvas.frame.grid(row=0, column=1, sticky="nsew")

        status = ttk.Label(
            self.root, textvariable=self._status_var, relief="sunken", anchor="w", padding=(6, 2)
        )
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    # -- status sink ---------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self._status_var.set(message)

    def _on_model_changed(self) -> None:
        has_feasible = self.model.highlighted is not None
        self._file_menu.entryconfig(_EXPORT_LABEL, state="normal" if has_feasible else "disabled")
        if not has_feasible:
            self._set_status("No feasible Dubins path for this scenario.")

    # -- file menu handlers --------------------------------------------------

    def _on_save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save scenario",
            defaultextension=".json",
            filetypes=[("Scenario JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            save_scenario(self.model, path)
        except (ScenarioError, OSError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self._set_status(f"Saved scenario to {path}")

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load scenario",
            filetypes=[("Scenario JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            loaded = load_scenario(path)
        except (ScenarioError, OSError) as exc:
            # The model is never touched on a failed load (FR-23-style safety).
            messagebox.showerror("Load failed", str(exc))
            return
        self.model.update(**loaded.to_update_kwargs())
        self._set_status(f"Loaded scenario from {path}")

    def _on_export_csv(self) -> None:
        highlighted = self.model.highlighted
        if highlighted is None:
            return
        solution = self.model.solutions[highlighted]
        if not isinstance(solution, DubinsPath):
            return
        step = simpledialog.askfloat(
            "Export waypoints",
            "Sample step (m):",
            initialvalue=_DEFAULT_STEP,
            minvalue=1e-4,
            parent=self.root,
        )
        if step is None:
            return
        path = filedialog.asksaveasfilename(
            title="Export waypoints CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export_waypoints_csv(path, solution, step)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._set_status(f"Exported {highlighted.value} waypoints to {path}")

    # -- help menu -----------------------------------------------------------

    def _help_url(self) -> str:
        """Return a ``file://`` URL for the bundled help page, cross-platform."""
        try:
            resource = importlib.resources.files("dubins_demo") / "help" / "index.html"
            candidate = Path(str(resource))
            if not candidate.is_file():
                raise FileNotFoundError
        except (FileNotFoundError, ModuleNotFoundError, TypeError):
            candidate = Path(__file__).resolve().parent.parent / "help" / "index.html"
        return candidate.resolve().as_uri()

    def _on_help(self) -> None:
        try:
            webbrowser.open(self._help_url())
        except OSError as exc:
            messagebox.showerror("Help unavailable", str(exc))

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> None:
        """Enter the Tk main loop (blocks until the window is closed)."""
        self.root.mainloop()
