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

from dubins_demo.core.dubins import DubinsPath
from dubins_demo.core.model import Scenario
from dubins_demo.persistence.scenario_io import (
    ScenarioError,
    export_waypoints_csv,
    load_scenario,
    save_scenario,
)
from dubins_demo.ui import theme
from dubins_demo.ui.details_panel import DetailsPanel
from dubins_demo.ui.input_panel import InputPanel
from dubins_demo.ui.plot_canvas import PlotCanvas

_EXPORT_LABEL = "Export CSV…"
_DEFAULT_STEP = 0.05
_PANEL_REFRESH_FAILED_STATUS = "A panel failed to refresh — see the error dialog."


class App:
    """The Dubins Path Demonstrator main window."""

    def __init__(self, model: Scenario) -> None:
        """Create the root window and wire the model to all three panels."""
        self.model = model
        self.root = tk.Tk()
        self.root.title("Dubins Path Demonstrator")
        self.root.geometry("1100x720")
        self.root.minsize(920, 600)

        self._status_var = tk.StringVar(value="Ready.")
        self._no_feasible_shown = False

        self._build_menu()
        self._build_layout()

        # _on_model_changed is registered after the panel listeners (added in
        # _build_layout) so it runs last in each notify pass: panels refresh
        # first, then App re-reads the model. Because it runs last it can also
        # overwrite the status set by _on_listener_error, which is why _on_load
        # re-asserts the honest failure status when update() reports a failed
        # refresh.
        self.model.add_listener(self._on_model_changed)
        self.model.set_error_handler(self._on_listener_error)
        self._on_model_changed()

        # Apply the shared design system once, after every widget exists.
        theme.apply_theme(self.root)
        # sv_ttk.set_theme() recolors native menus from an idle callback, so a
        # synchronous restyle here would be overwritten. Queue ours after_idle
        # (FIFO) so it runs *after* sv_ttk's and wins the final word.
        self.root.after_idle(self._style_menus)

    # -- layout --------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        self._file_menu = tk.Menu(menubar, tearoff=False)
        self._file_menu.add_command(
            label="Save…", underline=0, accelerator="Ctrl+S", command=self._on_save
        )
        self._file_menu.add_command(
            label="Load…", underline=0, accelerator="Ctrl+O", command=self._on_load
        )
        self._file_menu.add_command(
            label=_EXPORT_LABEL, underline=0, accelerator="Ctrl+E", command=self._on_export_csv
        )
        self._file_menu.add_separator()
        self._file_menu.add_command(
            label="Exit", underline=1, accelerator="Ctrl+Q", command=self.root.destroy
        )
        menubar.add_cascade(label="File", underline=0, menu=self._file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(
            label="Open help page", underline=0, accelerator="F1", command=self._on_help
        )
        menubar.add_cascade(label="Help", underline=0, menu=help_menu)

        # Recolored after apply_theme(): sv_ttk.set_theme() rewrites menu colors,
        # so styling them here would be clobbered. Held for _style_menus().
        self._menus = (menubar, self._file_menu, help_menu)

        self._bind_accelerators()
        self.root.config(menu=menubar)

    def _style_menus(self) -> None:
        """Apply the light menu palette; call *after* ``theme.apply_theme``."""
        for menu in self._menus:
            theme.style_menu(menu)

    def _bind_accelerators(self) -> None:
        """Wire the menu accelerators to root key bindings (both keypad cases)."""
        bindings = {
            "<Control-s>": self._on_save,
            "<Control-o>": self._on_load,
            "<Control-e>": self._on_export_csv,
            "<Control-q>": lambda: self.root.destroy(),
            "<F1>": self._on_help,
        }
        for sequence, handler in bindings.items():
            self.root.bind_all(sequence, lambda _e, fn=handler: fn())

    def _build_layout(self) -> None:
        # Content stretches in the plot column (1); the status bar sits below a
        # thin divider (row 1), with the status label on the final row (row 2).
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, width=320)
        left.grid(row=0, column=0, sticky="ns", padx=(theme.PAD_L, theme.PAD_M), pady=theme.PAD_L)
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        self.input_panel = InputPanel(left, self.model, self._set_status)
        self.input_panel.frame.grid(row=0, column=0, sticky="new")
        self.details_panel = DetailsPanel(left, self.model, self._set_status)
        self.details_panel.frame.grid(row=1, column=0, sticky="nsew", pady=(theme.PAD_M, 0))

        self.plot_canvas = PlotCanvas(self.root, self.model, self._set_status)
        self.plot_canvas.frame.grid(
            row=0, column=1, sticky="nsew", padx=(0, theme.PAD_L), pady=theme.PAD_L
        )

        separator = ttk.Separator(self.root, orient="horizontal")
        separator.grid(row=1, column=0, columnspan=2, sticky="ew")

        status = ttk.Label(
            self.root, textvariable=self._status_var, style=theme.STATUS_LABEL, anchor="w"
        )
        status.grid(row=2, column=0, columnspan=2, sticky="ew")

    # -- status sink ---------------------------------------------------------

    def _set_status(self, message: str) -> None:
        self._status_var.set(message)

    def _on_listener_error(self, exc: BaseException) -> None:
        """Surface a view-refresh failure the model would otherwise swallow.

        Registered with :meth:`Scenario.set_error_handler`, so a panel that
        raises while re-reading the model no longer fails invisibly: the status
        bar flags it and a dialog carries the detail, matching how load/save
        errors are shown. Other panels are still refreshed (the model isolates
        each listener); this only reports the one that broke.

        :meth:`Scenario.update` returns ``False`` when a listener raises, which
        lets a batched operation like :meth:`_on_load` keep this honest status
        instead of overwriting it with a success line.
        """
        self._set_status(_PANEL_REFRESH_FAILED_STATUS)
        messagebox.showerror("Panel refresh failed", str(exc))

    def _on_model_changed(self) -> None:
        has_feasible = self.model.highlighted is not None
        self._file_menu.entryconfig(_EXPORT_LABEL, state="normal" if has_feasible else "disabled")
        if not has_feasible:
            self._set_status("No feasible Dubins path for this scenario.")
            self._no_feasible_shown = True
        elif self._no_feasible_shown:
            # Clear only the stale infeasibility notice; leave any other status
            # (mouse coords, "Loaded…") untouched.
            self._set_status("Ready.")
            self._no_feasible_shown = False

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
        if not self.model.update(**loaded.to_update_kwargs()):
            # A panel raised while refreshing: _on_listener_error already showed
            # the dialog, but _on_model_changed (the last listener) may have
            # stamped "Ready." or the infeasibility notice over the honest status
            # during update(). Re-assert it rather than falsely reporting a clean
            # load.
            self._set_status(_PANEL_REFRESH_FAILED_STATUS)
            return
        # update() has already fired _on_model_changed, which sets the
        # infeasibility notice for an unsolvable scenario. Overwriting it with a
        # bare success line would hide that the loaded file has no path, so
        # qualify the status when nothing is feasible.
        if self.model.highlighted is None:
            self._set_status(f"Loaded {path} — no feasible Dubins path for this scenario.")
        else:
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
            # export_waypoints_csv samples the path before opening the file, so
            # a compute error (ValueError) is guarded here alongside I/O errors.
            export_waypoints_csv(path, solution, step)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        self._set_status(f"Exported {highlighted.value} waypoints to {path}")

    # -- help menu -----------------------------------------------------------

    def _help_url(self) -> str:
        """Return a ``file://`` URL for the bundled help page, cross-platform.

        Both the packaged-resource location and the source-tree fallback are
        checked for an existing file before their URL is returned; if neither
        exists, :class:`FileNotFoundError` is raised rather than handing back a
        URL that points at a missing page.
        """
        try:
            resource = importlib.resources.files("dubins_demo") / "help" / "index.html"
            candidate = Path(str(resource))
            if candidate.is_file():
                return candidate.resolve().as_uri()
        except (ModuleNotFoundError, TypeError):
            pass
        fallback = Path(__file__).resolve().parent.parent / "help" / "index.html"
        if fallback.is_file():
            return fallback.resolve().as_uri()
        raise FileNotFoundError("bundled help page (help/index.html) not found")

    def _on_help(self) -> None:
        try:
            opened = webbrowser.open(self._help_url())
        except OSError as exc:  # FileNotFoundError (missing page) is an OSError
            messagebox.showerror("Help unavailable", str(exc))
            return
        if not opened:
            messagebox.showerror(
                "Help unavailable", "No web browser could be opened for the help page."
            )

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> None:
        """Enter the Tk main loop (blocks until the window is closed)."""
        self.root.mainloop()
