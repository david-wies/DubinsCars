"""Observable scenario model: the single source of truth for the UI.

The model owns the current start/goal configurations, the turn-radius policy,
display preferences, and the cached solver results. Views subscribe via
:meth:`Scenario.add_listener` and re-read the model on notification; they never
run the solvers themselves. All solving happens in :meth:`Scenario.update`,
which batches field changes into a single ``solve_all`` call and a single
notification.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from dubins_demo.core.dubins import (
    Config,
    DubinsPath,
    Infeasible,
    PathType,
    shortest,
    solve_all,
)


@runtime_checkable
class RadiusPolicy(Protocol):
    """Supplies the minimum turn radius in meters."""

    def min_radius(self) -> float: ...


@dataclass
class FixedRadius:
    """A constant turn radius driven directly by the UI (EXT-1)."""

    value: float

    def min_radius(self) -> float:
        """Return the fixed radius value."""
        return self.value


class Convention(Enum):
    """Heading display convention (affects UI formatting only)."""

    ANGLE = "angle"
    AZIMUTH = "azimuth"


class Unit(Enum):
    """Angle display unit (affects UI formatting only)."""

    DEG = "deg"
    RAD = "rad"


class Scenario:
    """Mutable, observable scenario state with cached Dubins solutions."""

    def __init__(
        self,
        start: Config,
        goal: Config,
        radius_policy: RadiusPolicy,
        *,
        heading_convention: Convention = Convention.ANGLE,
        angle_unit: Unit = Unit.DEG,
        selected_type: PathType | None = None,
        show_circles: bool = False,
        animation_speed: float = 1.0,
    ) -> None:
        self.start = start
        self.goal = goal
        self.radius_policy = radius_policy
        self.heading_convention = heading_convention
        self.angle_unit = angle_unit
        self.selected_type = selected_type
        self.show_circles = show_circles
        self.animation_speed = animation_speed

        self._listeners: list[Callable[[], None]] = []
        self.solutions: dict[PathType, DubinsPath | Infeasible] = {}
        self.highlighted: PathType | None = None
        self._resolve()  # valid solutions/highlighted immediately after construction

    def add_listener(self, cb: Callable[[], None]) -> None:
        """Register a zero-argument callback fired after each :meth:`update`."""
        self._listeners.append(cb)

    def update(self, **changes: object) -> None:
        """Apply field changes, re-solve, then notify all listeners exactly once.

        Only existing public attributes may be set; an unknown field name is a
        programming error and raises :class:`AttributeError`.
        """
        for name, value in changes.items():
            if not hasattr(self, name) or name.startswith("_"):
                raise AttributeError(f"unknown scenario field: {name!r}")
            setattr(self, name, value)
        self._resolve()
        self._notify()

    def _resolve(self) -> None:
        """Recompute cached ``solutions`` and the ``highlighted`` selection."""
        self.solutions = solve_all(self.start, self.goal, self.radius_policy.min_radius())
        selected = self.selected_type
        if selected is not None and isinstance(self.solutions.get(selected), DubinsPath):
            self.highlighted = selected
        else:
            self.highlighted = shortest(self.solutions)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb()
