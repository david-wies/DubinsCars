"""Details panel: a sortable table of the six Dubins solutions.

Feasible paths are listed with their total length and per-segment breakdown and
can be re-sorted by clicking a column header (default: length ascending).
Infeasible words are always pinned to the bottom, grayed out, with their reason
in the Segments column (FR-13). A row click selects that word as the highlight
(``model.update(selected_type=...)``); the highlighted row is kept in sync when
the model falls back to the shortest path after a re-solve (FR-12).
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

from dubins_demo.core.dubins import DubinsPath, Infeasible, PathType
from dubins_demo.core.model import Scenario


def _segments_text(path: DubinsPath) -> str:
    """Render a path's segments, e.g. ``L 2.31 → S 5.10 → L 0.87``."""
    return " → ".join(f"{seg.kind.value} {seg.length:.2f}" for seg in path.segments)


class DetailsPanel:
    """A ``ttk.Treeview`` view of the model's cached solutions."""

    _COLUMNS = (("type", "Type", 60), ("length", "Length [m]", 90), ("segments", "Segments", 200))

    def __init__(
        self,
        parent: tk.Misc,
        model: Scenario,
        status_sink: Callable[[str], None],
    ) -> None:
        """Build the table under ``parent`` and subscribe it to ``model``."""
        self.model = model
        self._status = status_sink
        self.frame = ttk.Frame(parent, padding=6)

        self._refreshing = False
        self._sort_col = "length"
        self._sort_reverse = False

        columns = [name for name, _label, _w in self._COLUMNS]
        self._tree = ttk.Treeview(self.frame, columns=columns, show="headings", height=6)
        for name, label, width in self._COLUMNS:
            self._tree.heading(name, text=label, command=lambda c=name: self._sort_by(c))
            anchor = "w" if name == "segments" else "center"
            self._tree.column(name, width=width, anchor=anchor, stretch=(name == "segments"))
        self._tree.tag_configure("infeasible", foreground="#909090")

        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        self.model.add_listener(self._on_model_changed)
        self._render()

    # -- sorting -------------------------------------------------------------

    def _sort_by(self, column: str) -> None:
        if self._sort_col == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = column
            self._sort_reverse = False
        self._render()

    def _sort_key(self, item: tuple[PathType, DubinsPath]):
        path_type, path = item
        if self._sort_col == "type":
            return path_type.value
        if self._sort_col == "segments":
            return _segments_text(path)
        return path.length

    # -- rendering -----------------------------------------------------------

    def _on_model_changed(self) -> None:
        self._render()

    def _render(self) -> None:
        self._refreshing = True
        try:
            self._tree.delete(*self._tree.get_children())
            feasible: list[tuple[PathType, DubinsPath]] = []
            infeasible: list[tuple[PathType, Infeasible]] = []
            for path_type, solution in self.model.solutions.items():
                if isinstance(solution, DubinsPath):
                    feasible.append((path_type, solution))
                else:
                    infeasible.append((path_type, solution))

            feasible.sort(key=self._sort_key, reverse=self._sort_reverse)
            for path_type, path in feasible:
                self._tree.insert(
                    "",
                    "end",
                    iid=path_type.name,
                    values=(path_type.value, f"{path.length:.2f}", _segments_text(path)),
                )
            for path_type, reason in infeasible:
                self._tree.insert(
                    "",
                    "end",
                    iid=path_type.name,
                    values=(path_type.value, "—", reason.reason),
                    tags=("infeasible",),
                )

            highlighted = self.model.highlighted
            if highlighted is not None:
                self._tree.selection_set(highlighted.name)
                self._tree.see(highlighted.name)
            else:
                self._tree.selection_set()
        finally:
            self._refreshing = False

    # -- selection -----------------------------------------------------------

    def _on_select(self, _event: object) -> None:
        if self._refreshing:
            return
        selection = self._tree.selection()
        if not selection:
            return
        try:
            path_type = PathType[selection[0]]
        except KeyError:
            return
        # Tk delivers <<TreeviewSelect>> asynchronously, so the selection_set()
        # in _render() echoes back here after _refreshing has cleared. Bail when
        # the row already matches the model to avoid an infinite update/notify
        # loop (update() always notifies, even on a no-op change).
        if path_type is self.model.highlighted:
            return
        self.model.update(selected_type=path_type)
