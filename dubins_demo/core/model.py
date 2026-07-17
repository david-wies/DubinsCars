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


@dataclass(frozen=True)
class FixedRadius:
    """A constant turn radius driven directly by the UI (EXT-1).

    Frozen so that a value object holding it (e.g. ``LoadedScenario``) is
    genuinely immutable; change the radius by constructing a new instance.
    """

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
    """Observable scenario state with cached Dubins solutions.

    All input/display state is held in private fields exposed through read-only
    :func:`property` accessors. :meth:`update` is the *sole* mutator: it is the
    only way to change a field, and it always re-solves and notifies, so the
    cached ``solutions`` / ``highlighted`` can never go stale behind the views'
    backs.
    """

    #: Fields that :meth:`update` may set. The derived caches (``solutions``,
    #: ``highlighted``) are deliberately absent -- they are recomputed, never set.
    _SETTABLE = frozenset(
        {
            "start",
            "goal",
            "radius_policy",
            "heading_convention",
            "angle_unit",
            "selected_type",
            "show_circles",
        }
    )

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
        self._start = start
        self._goal = goal
        self._radius_policy = radius_policy
        self._heading_convention = heading_convention
        self._angle_unit = angle_unit
        self._selected_type = selected_type
        self._show_circles = show_circles
        self._animation_speed = animation_speed

        self._listeners: list[Callable[[], None]] = []
        self._solutions: dict[PathType, DubinsPath | Infeasible] = {}
        self._highlighted: PathType | None = None
        self._resolve()  # valid solutions/highlighted immediately after construction

    # -- read-only accessors -------------------------------------------------

    @property
    def start(self) -> Config:
        """The start configuration (set via :meth:`update`)."""
        return self._start

    @property
    def goal(self) -> Config:
        """The goal configuration (set via :meth:`update`)."""
        return self._goal

    @property
    def radius_policy(self) -> RadiusPolicy:
        """The turn-radius policy (set via :meth:`update`)."""
        return self._radius_policy

    @property
    def heading_convention(self) -> Convention:
        """The heading display convention (set via :meth:`update`)."""
        return self._heading_convention

    @property
    def angle_unit(self) -> Unit:
        """The angle display unit (set via :meth:`update`)."""
        return self._angle_unit

    @property
    def selected_type(self) -> PathType | None:
        """The user-selected word to highlight, if any (set via :meth:`update`)."""
        return self._selected_type

    @property
    def show_circles(self) -> bool:
        """Whether turning circles are shown (set via :meth:`update`)."""
        return self._show_circles

    @property
    def animation_speed(self) -> float:
        """The animation speed in m/s (set via :meth:`update`)."""
        return self._animation_speed

    @property
    def solutions(self) -> dict[PathType, DubinsPath | Infeasible]:
        """The cached per-word solver results (recomputed by :meth:`update`)."""
        return self._solutions

    @property
    def highlighted(self) -> PathType | None:
        """The word currently highlighted, or ``None`` if none is feasible."""
        return self._highlighted

    # -- listeners / mutation ------------------------------------------------

    def add_listener(self, cb: Callable[[], None]) -> None:
        """Register a zero-argument callback fired after each :meth:`update`."""
        self._listeners.append(cb)

    def update(self, **changes: object) -> None:
        """Apply field changes, re-solve, then notify all listeners exactly once.

        Only the settable input/display fields in :attr:`_SETTABLE` may be
        changed; the derived caches (``solutions``, ``highlighted``) and any
        unknown name are rejected with :class:`AttributeError`.
        """
        for name, value in changes.items():
            if name not in self._SETTABLE:
                raise AttributeError(f"unknown or read-only scenario field: {name!r}")
            setattr(self, f"_{name}", value)
        self._resolve()
        self._notify()

    def set_animation_speed(self, speed: float) -> None:
        """Set playback speed *without* re-solving or notifying.

        Animation speed is pure playback state; it does not affect the geometry.
        Routing it through :meth:`update` would waste a re-solve and, worse,
        trigger the FR-15 "any scenario change resets the animation" teardown in
        the views -- editing the speed while a path is animating would stop it.
        This dedicated setter keeps the field encapsulated (no public attribute
        write) while leaving a running animation untouched.
        """
        self._animation_speed = speed

    def _resolve(self) -> None:
        """Recompute cached ``solutions`` and the ``highlighted`` selection."""
        self._solutions = solve_all(self._start, self._goal, self._radius_policy.min_radius())
        selected = self._selected_type
        if selected is not None and isinstance(self._solutions.get(selected), DubinsPath):
            self._highlighted = selected
        else:
            self._highlighted = shortest(self._solutions)

    def _notify(self) -> None:
        for cb in self._listeners:
            cb()
