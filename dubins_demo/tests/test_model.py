"""Tests for :mod:`dubins_demo.core.model`."""

from __future__ import annotations

import math

import pytest

from dubins_demo.core.dubins import Config, DubinsPath, PathType, shortest, solve_all
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


def test_fixed_radius_rejects_non_positive_and_non_finite() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="radius"):
            FixedRadius(bad)


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


def test_update_with_invalid_field_mutates_nothing() -> None:
    # A mix of one valid and one rejected key must leave the model untouched:
    # no partial mutation, no re-solve, no notification (keys validated first).
    scenario = _make_scenario()
    solutions_before = scenario.solutions
    circles_before = scenario.show_circles
    calls = {"n": 0}
    scenario.add_listener(lambda: calls.__setitem__("n", calls["n"] + 1))

    with pytest.raises(AttributeError):
        scenario.update(show_circles=not circles_before, animation_speed=2.0)

    assert scenario.show_circles is circles_before
    assert scenario.solutions is solutions_before  # no re-solve
    assert calls["n"] == 0  # no notification


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
    # A non-positive radius makes every word infeasible, so nothing is
    # highlighted. FixedRadius now rejects non-positive values, so drive the
    # degenerate case through a bare RadiusPolicy stub that returns 0.
    class _ZeroRadius:
        def min_radius(self) -> float:
            return 0.0

    scenario = Scenario(
        start=Config(0.0, 0.0, 0.0),
        goal=Config(10.0, 5.0, math.pi / 2),
        radius_policy=_ZeroRadius(),
    )
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


def test_solutions_is_a_read_only_view() -> None:
    scenario = _make_scenario()
    solutions = scenario.solutions
    # The cache cannot be mutated through the returned view.
    with pytest.raises(TypeError):
        solutions[PathType.LSL] = None  # type: ignore[index]
    # And it reflects the current solve, updating on the next re-solve.
    assert set(solutions) == set(PathType)
    scenario.update(goal=Config(20.0, 0.0, 0.0))
    assert set(scenario.solutions) == set(PathType)


def test_animation_speed_rejects_negative_and_non_finite() -> None:
    scenario = _make_scenario()
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="speed"):
            scenario.set_animation_speed(bad)
        with pytest.raises(ValueError, match="speed"):
            _make_scenario(animation_speed=bad)


def test_animation_speed_rejects_zero() -> None:
    scenario = _make_scenario()
    with pytest.raises(ValueError, match="speed"):
        scenario.set_animation_speed(0.0)
    with pytest.raises(ValueError, match="speed"):
        _make_scenario(animation_speed=0.0)


def test_update_with_bad_value_leaves_model_unmutated() -> None:
    scenario = _make_scenario()
    original_goal = scenario.goal
    solutions_before = scenario.solutions
    calls = {"n": 0}
    scenario.add_listener(lambda: calls.__setitem__("n", calls["n"] + 1))

    # An ill-typed value is rejected before any field is written, even when a
    # valid field is changed in the same call.
    with pytest.raises(TypeError):
        scenario.update(goal=Config(1.0, 1.0, 0.0), start="oops")
    assert scenario.goal is original_goal  # valid field not applied either
    assert scenario.solutions is solutions_before  # no re-solve
    assert calls["n"] == 0  # no notification

    with pytest.raises(TypeError):
        scenario.update(radius_policy=None)
    assert calls["n"] == 0


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("start", "oops"),
        ("goal", "oops"),
        ("radius_policy", None),
        ("heading_convention", "angle"),  # the string, not the Convention enum
        ("angle_unit", "deg"),  # the string, not the Unit enum
        ("selected_type", "LSL"),  # not a PathType and not None
        ("show_circles", "yes"),  # not a bool
    ],
)
def test_constructor_rejects_ill_typed_field(field: str, bad_value: object) -> None:
    # Each of the seven _validate_settable guards in __init__ must reject a
    # single ill-typed field. Start from an all-valid kwargs baseline and
    # corrupt exactly one field so the failure is unambiguous.
    kwargs: dict[str, object] = {
        "start": Config(0.0, 0.0, 0.0),
        "goal": Config(10.0, 5.0, math.pi / 2),
        "radius_policy": FixedRadius(2.0),
        "heading_convention": Convention.ANGLE,
        "angle_unit": Unit.DEG,
        "selected_type": None,
        "show_circles": False,
    }
    kwargs[field] = bad_value
    with pytest.raises(TypeError, match=field):
        Scenario(**kwargs)  # type: ignore[arg-type]


def test_notify_isolates_failing_listener(capsys: pytest.CaptureFixture[str]) -> None:
    # A listener that raises must not strand later listeners: _notify isolates
    # each callback, so the second still fires and the model stays consistent.
    scenario = _make_scenario()
    fired = {"second": 0}

    def boom() -> None:
        raise RuntimeError("view blew up")

    scenario.add_listener(boom)
    scenario.add_listener(lambda: fired.__setitem__("second", fired["second"] + 1))

    new_goal = Config(8.0, -3.0, 0.0)
    # update() must not propagate the listener's exception.
    scenario.update(goal=new_goal)

    assert fired["second"] == 1  # later listener still ran
    assert scenario.goal is new_goal  # model state applied and consistent
    assert scenario.highlighted == shortest(scenario.solutions)
    # The swallowed error leaves a stderr trace rather than vanishing silently.
    assert "RuntimeError" in capsys.readouterr().err


def test_notify_error_handler_receives_failure_and_others_still_run() -> None:
    # A registered error handler surfaces a swallowed listener crash while the
    # per-listener isolation is preserved: the later listener still fires and
    # the handler is called with the raised exception.
    scenario = _make_scenario()
    fired = {"second": 0}
    seen: list[BaseException] = []

    boom_error = RuntimeError("view blew up")

    def boom() -> None:
        raise boom_error

    scenario.set_error_handler(seen.append)
    scenario.add_listener(boom)
    scenario.add_listener(lambda: fired.__setitem__("second", fired["second"] + 1))

    scenario.update(goal=Config(8.0, -3.0, 0.0))

    assert fired["second"] == 1  # later listener still ran
    assert seen == [boom_error]  # handler received the exact exception


def test_notify_falls_back_to_stderr_without_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # With no handler registered, a raising listener falls back to a stderr
    # traceback and does not propagate out of update().
    scenario = _make_scenario()

    scenario.add_listener(lambda: (_ for _ in ()).throw(RuntimeError("view blew up")))
    scenario.update(goal=Config(8.0, -3.0, 0.0))  # must not raise

    assert "RuntimeError" in capsys.readouterr().err


def test_notify_raising_error_handler_does_not_propagate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A handler that itself raises must not defeat the isolation it serves:
    # its failure is swallowed to stderr and the remaining listeners still run.
    scenario = _make_scenario()
    fired = {"second": 0}

    def bad_handler(_exc: BaseException) -> None:
        raise ValueError("handler blew up")

    scenario.set_error_handler(bad_handler)
    scenario.add_listener(lambda: (_ for _ in ()).throw(RuntimeError("view blew up")))
    scenario.add_listener(lambda: fired.__setitem__("second", fired["second"] + 1))

    scenario.update(goal=Config(8.0, -3.0, 0.0))  # must not raise

    assert fired["second"] == 1  # isolation preserved despite the broken handler
    assert "ValueError" in capsys.readouterr().err


def test_update_solves_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _make_scenario()
    count = {"n": 0}
    real = solve_all

    def counting_solve_all(*args: object, **kwargs: object) -> object:
        count["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("dubins_demo.core.model.solve_all", counting_solve_all)
    scenario.update(goal=Config(8.0, -3.0, 0.0), show_circles=True)
    assert count["n"] == 1  # one update -> exactly one solve_all, not one per field
