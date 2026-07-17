"""Tests for :mod:`dubins_demo.core.model`."""

from __future__ import annotations

import math

import pytest

from dubins_demo.core.dubins import Config, DubinsPath, PathType, shortest
from dubins_demo.core.model import (
    Convention,
    FixedRadius,
    RadiusPolicy,
    Scenario,
    Unit,
)


def _make_scenario(**kwargs: object) -> Scenario:
    return Scenario(
        start=Config(0.0, 0.0, 0.0),
        goal=Config(10.0, 5.0, math.pi / 2),
        radius_policy=FixedRadius(2.0),
        **kwargs,  # type: ignore[arg-type]
    )


def test_fixed_radius_is_a_policy() -> None:
    policy = FixedRadius(3.5)
    assert isinstance(policy, RadiusPolicy)
    assert policy.min_radius() == 3.5


def test_constructor_solves_immediately() -> None:
    scenario = _make_scenario()
    assert set(scenario.solutions) == set(PathType)
    assert scenario.highlighted == shortest(scenario.solutions)


def test_update_notifies_listeners_exactly_once() -> None:
    scenario = _make_scenario()
    calls = {"a": 0, "b": 0}

    scenario.add_listener(lambda: calls.__setitem__("a", calls["a"] + 1))
    scenario.add_listener(lambda: calls.__setitem__("b", calls["b"] + 1))

    scenario.update(goal=Config(8.0, -3.0, 0.0), show_circles=True)
    assert calls == {"a": 1, "b": 1}

    scenario.update(show_circles=False)
    assert calls == {"a": 2, "b": 2}


def test_set_animation_speed_does_not_notify_or_resolve() -> None:
    scenario = _make_scenario()
    solutions_before = scenario.solutions
    calls = {"n": 0}
    scenario.add_listener(lambda: calls.__setitem__("n", calls["n"] + 1))

    scenario.set_animation_speed(2.0)

    assert scenario.animation_speed == 2.0
    assert calls["n"] == 0  # playback state must not fire the FR-15 reset
    assert scenario.solutions is solutions_before  # no re-solve


def test_animation_speed_is_not_settable_via_update() -> None:
    scenario = _make_scenario()
    with pytest.raises(AttributeError):
        scenario.update(animation_speed=2.0)


def test_update_applies_changes_and_resolves() -> None:
    scenario = _make_scenario()
    scenario.update(radius_policy=FixedRadius(1.0), show_circles=True)
    assert scenario.show_circles is True
    assert scenario.radius_policy.min_radius() == 1.0
    for sol in scenario.solutions.values():
        if isinstance(sol, DubinsPath):
            assert sol.radius == 1.0


def test_selected_type_kept_when_feasible() -> None:
    scenario = _make_scenario()
    # Pick a feasible, non-shortest word to prove selection overrides the default.
    feasible = [pt for pt, s in scenario.solutions.items() if isinstance(s, DubinsPath)]
    non_shortest = next(pt for pt in feasible if pt != scenario.highlighted)
    scenario.update(selected_type=non_shortest)
    assert scenario.highlighted == non_shortest


def test_selection_falls_back_to_shortest_when_infeasible() -> None:
    scenario = _make_scenario()
    # RLR is infeasible for a far-apart goal; selecting it should fall back.
    scenario.update(goal=Config(50.0, 0.0, 0.0), selected_type=PathType.RLR)
    assert scenario.highlighted == shortest(scenario.solutions)
    assert scenario.highlighted != PathType.RLR


def test_unknown_field_rejected() -> None:
    scenario = _make_scenario()
    with pytest.raises(AttributeError):
        scenario.update(nonexistent_field=123)
    with pytest.raises(AttributeError):
        scenario.update(_listeners=[])


def test_highlighted_none_when_all_infeasible() -> None:
    scenario = _make_scenario()
    # A non-positive radius makes every word infeasible, so nothing is highlighted.
    scenario.update(radius_policy=FixedRadius(0.0))
    assert scenario.highlighted is None
    assert all(not isinstance(s, DubinsPath) for s in scenario.solutions.values())


def test_derived_caches_are_read_only() -> None:
    scenario = _make_scenario()
    # ``solutions`` / ``highlighted`` are recomputed, never set through update().
    with pytest.raises(AttributeError):
        scenario.update(solutions={})
    with pytest.raises(AttributeError):
        scenario.update(highlighted=PathType.LSL)
    # And the properties themselves reject direct assignment.
    with pytest.raises(AttributeError):
        scenario.start = Config(1.0, 1.0, 0.0)  # type: ignore[misc]


def test_display_prefs_do_not_affect_solutions() -> None:
    scenario = _make_scenario()
    before = {pt: getattr(s, "length", None) for pt, s in scenario.solutions.items()}
    scenario.update(heading_convention=Convention.AZIMUTH, angle_unit=Unit.RAD)
    after = {pt: getattr(s, "length", None) for pt, s in scenario.solutions.items()}
    assert before == after
