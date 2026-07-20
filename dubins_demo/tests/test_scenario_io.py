"""Tests for :mod:`dubins_demo.persistence.scenario_io`."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from dubins_demo.core.dubins import Config, DubinsPath, solve_all
from dubins_demo.core.model import Convention, FixedRadius, Scenario, Unit
from dubins_demo.persistence.scenario_io import (
    ScenarioError,
    dict_to_scenario,
    export_waypoints_csv,
    load_scenario,
    save_scenario,
    scenario_to_dict,
)


def _make_scenario() -> Scenario:
    """A representative scenario with non-default display preferences."""
    return Scenario(
        start=Config(x=0.0, y=0.0, theta=0.0),
        goal=Config(x=10.0, y=5.0, theta=1.57),
        radius_policy=FixedRadius(2.0),
        heading_convention=Convention.AZIMUTH,
        angle_unit=Unit.RAD,
    )


def _feasible_path() -> DubinsPath:
    """A known feasible path for CSV-export tests."""
    start = Config(0.0, 0.0, 0.0)
    goal = Config(10.0, 0.0, 0.0)
    solutions = solve_all(start, goal, radius=2.0)
    for sol in solutions.values():
        if isinstance(sol, DubinsPath):
            return sol
    raise AssertionError("expected at least one feasible path for the test fixture")


def _zero_length_path() -> DubinsPath:
    """A degenerate zero-length path (start == goal) that samples to one row."""
    config = Config(3.0, 4.0, 1.0)
    solutions = solve_all(config, config, radius=2.0)
    for sol in solutions.values():
        if isinstance(sol, DubinsPath) and sol.length == 0.0:
            return sol
    raise AssertionError("expected a zero-length feasible path for the test fixture")


# --- JSON round-trip --------------------------------------------------------


def test_json_round_trip_restores_all_fields(tmp_path: Path) -> None:
    scenario = _make_scenario()
    target = tmp_path / "scenario.json"

    save_scenario(scenario, target)
    loaded = load_scenario(target)

    assert loaded.start.x == scenario.start.x
    assert loaded.start.y == scenario.start.y
    assert loaded.start.theta == pytest.approx(scenario.start.theta, abs=1e-12)
    assert loaded.goal.x == scenario.goal.x
    assert loaded.goal.y == scenario.goal.y
    assert loaded.goal.theta == pytest.approx(scenario.goal.theta, abs=1e-12)

    assert isinstance(loaded.radius_policy, FixedRadius)
    assert loaded.radius_policy.value == pytest.approx(2.0, abs=1e-12)
    assert loaded.radius_policy.min_radius() == pytest.approx(2.0, abs=1e-12)

    assert loaded.heading_convention is Convention.AZIMUTH
    assert loaded.angle_unit is Unit.RAD


def test_saved_file_matches_ext4_schema(tmp_path: Path) -> None:
    scenario = _make_scenario()
    target = tmp_path / "scenario.json"
    save_scenario(scenario, target)

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {
        "version": 1,
        "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
        "goal": {"x": 10.0, "y": 5.0, "theta": 1.57},
        "radius_policy": {"type": "fixed", "value": 2.0},
        "display": {"heading_convention": "azimuth", "angle_unit": "rad"},
    }


def test_partial_write_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-write I/O failure must leave the prior good file intact, no .tmp behind."""
    scenario = _make_scenario()
    target = tmp_path / "scenario.json"

    # Establish a known-good file that must survive a failed re-save.
    good = '{"kept": true}\n'
    target.write_text(good, encoding="utf-8")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated disk full")

    monkeypatch.setattr(json, "dump", boom)

    with pytest.raises(ScenarioError):
        save_scenario(scenario, target)

    # Original content untouched (atomic replace never happened) ...
    assert target.read_text(encoding="utf-8") == good
    # ... and no stray temp file was left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_scenario_replace_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An os.replace failure must leave the prior good file intact, no .tmp behind.

    Mirrors the CSV path's os.replace-failure test but for ``save_scenario``,
    whose contract wraps the raw :class:`OSError` into a :class:`ScenarioError`.
    Patching ``os.replace`` (not ``json.dump``) exercises the branch after the
    temp file is fully written, where the atomic rename itself fails.
    """
    scenario = _make_scenario()
    target = tmp_path / "scenario.json"

    # Establish a known-good file that must survive a failed re-save.
    good = '{"kept": true}\n'
    target.write_text(good, encoding="utf-8")

    def boom(src: object, dst: object, *args: object, **kwargs: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(ScenarioError):
        save_scenario(scenario, target)

    # Original content untouched (atomic replace never landed) ...
    assert target.read_text(encoding="utf-8") == good
    # ... and the fully-written temp file was cleaned up.
    assert list(tmp_path.glob("*.tmp")) == []


def test_to_update_kwargs_matches_load(tmp_path: Path) -> None:
    scenario = _make_scenario()
    target = tmp_path / "scenario.json"
    save_scenario(scenario, target)

    kwargs = load_scenario(target).to_update_kwargs()
    assert set(kwargs) == {
        "start",
        "goal",
        "radius_policy",
        "heading_convention",
        "angle_unit",
    }


# --- Validation errors ------------------------------------------------------


def test_unknown_radius_policy_type_raises(tmp_path: Path) -> None:
    document = scenario_to_dict(_make_scenario())
    document["radius_policy"] = {"type": "speed_based", "value": 2.0}

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "speed_based" in str(excinfo.value)


def test_malformed_json_raises_and_leaves_model_untouched(tmp_path: Path) -> None:
    scenario = _make_scenario()
    before = scenario_to_dict(scenario)

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ScenarioError):
        load_scenario(bad)

    assert scenario_to_dict(scenario) == before


def test_missing_key_raises(tmp_path: Path) -> None:
    document = scenario_to_dict(_make_scenario())
    del document["goal"]

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "goal" in str(excinfo.value)


def test_wrong_version_raises(tmp_path: Path) -> None:
    document = scenario_to_dict(_make_scenario())
    document["version"] = 2

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "2" in str(excinfo.value)


def test_bool_version_rejected() -> None:
    # ``True == 1 == SCHEMA_VERSION`` must not sneak past the version gate.
    document = scenario_to_dict(_make_scenario())
    document["version"] = True

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "version" in str(excinfo.value)


def test_non_numeric_coordinate_raises() -> None:
    document = scenario_to_dict(_make_scenario())
    document["start"]["x"] = "not a number"

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "start.x" in str(excinfo.value)


def test_integer_json_coordinates_load_as_floats() -> None:
    # JSON integers (no decimal point) are valid coordinates: ``_require_number``
    # accepts int and coerces via ``float()``. ``scenario_to_dict`` only ever
    # emits floats, so build the document by hand to exercise the int path.
    document = scenario_to_dict(_make_scenario())
    document["start"] = {"x": 0, "y": 0, "theta": 0}
    document["goal"] = {"x": 10, "y": 5, "theta": 1}

    loaded = dict_to_scenario(document)

    assert isinstance(loaded.start.x, float)
    assert loaded.start.x == 0.0
    assert loaded.goal.x == 10.0
    assert loaded.goal.theta == 1.0


@pytest.mark.parametrize("section", ["start", "goal", "display", "radius_policy"])
def test_scalar_where_mapping_required_raises(section: str) -> None:
    # A scalar where a nested object is expected must surface as a ScenarioError
    # naming the offending section, not a bare TypeError from ``_require_mapping``.
    document = scenario_to_dict(_make_scenario())
    document[section] = 5

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert section in str(excinfo.value)


@pytest.mark.parametrize("dropped_key", ["type", "value"])
def test_radius_policy_missing_key_raises(dropped_key: str) -> None:
    # Distinct from a present-but-wrong ``type``: an absent ``type``/``value``
    # must be reported by ``_require_key`` as a missing-key ScenarioError.
    document = scenario_to_dict(_make_scenario())
    del document["radius_policy"][dropped_key]

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    message = str(excinfo.value)
    assert dropped_key in message
    assert "radius_policy" in message


def test_bool_coordinate_rejected() -> None:
    # bool is an int subclass; ``true``/``false`` must not pass as a coordinate.
    document = scenario_to_dict(_make_scenario())
    document["start"]["x"] = True

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "start.x" in str(excinfo.value)


def test_non_finite_coordinate_rejected() -> None:
    # json.loads accepts NaN/Infinity by default; they must never reach the core.
    for bad in (float("nan"), float("inf"), float("-inf")):
        document = scenario_to_dict(_make_scenario())
        document["goal"]["y"] = bad
        with pytest.raises(ScenarioError) as excinfo:
            dict_to_scenario(document)
        assert "goal.y" in str(excinfo.value)


def test_non_finite_or_non_positive_radius_rejected() -> None:
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        document = scenario_to_dict(_make_scenario())
        document["radius_policy"]["value"] = bad
        with pytest.raises(ScenarioError):
            dict_to_scenario(document)


def test_scenario_to_dict_rejects_non_fixed_radius_policy() -> None:
    class _StubPolicy:
        def min_radius(self) -> float:
            return 2.0

    scenario = Scenario(
        start=Config(0.0, 0.0, 0.0),
        goal=Config(5.0, 0.0, 0.0),
        radius_policy=_StubPolicy(),
    )
    with pytest.raises(ScenarioError):
        scenario_to_dict(scenario)


def test_load_missing_file_raises_scenario_error(tmp_path: Path) -> None:
    # Every load failure -- filesystem included -- surfaces as one ScenarioError
    # so callers handle a single exception type (CLAUDE.md persistence contract).
    missing = tmp_path / "nope.json"
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(missing)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_load_undecodable_bytes_raises_scenario_error(tmp_path: Path) -> None:
    # Invalid UTF-8 must not escape as a bare UnicodeDecodeError.
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(bad)
    assert isinstance(excinfo.value.__cause__, UnicodeError)


def test_top_level_non_object_json_raises() -> None:
    with pytest.raises(ScenarioError):
        dict_to_scenario([1, 2, 3])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("heading_convention", "sideways"),
        ("angle_unit", "gradians"),
    ],
    ids=["heading_convention", "angle_unit"],
)
def test_invalid_display_enum_raises(field: str, bad_value: str) -> None:
    # ``_display_from_dict`` has two symmetric try/except blocks -- one per enum.
    # Corrupting each field in turn exercises both the ``Convention`` and the
    # ``Unit`` failure branch, and asserts each reports a sensible message that
    # echoes the bad value and enumerates the valid options.
    document = scenario_to_dict(_make_scenario())
    document["display"][field] = bad_value

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    message = str(excinfo.value)
    assert bad_value in message
    assert "expected one of" in message


def test_load_bad_document_does_not_mutate_live_scenario(tmp_path: Path) -> None:
    """A rejected load must not partially update a caller-held model."""
    scenario = _make_scenario()
    snapshot = scenario_to_dict(scenario)

    document = scenario_to_dict(scenario)
    document["radius_policy"] = {"type": "bogus", "value": 9.0}
    bad = tmp_path / "bogus.json"
    bad.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ScenarioError):
        loaded = load_scenario(bad)
        scenario.update(**loaded.to_update_kwargs())  # never reached

    assert scenario_to_dict(scenario) == snapshot


# --- CSV export -------------------------------------------------------------


def test_csv_export_header_and_row_count(tmp_path: Path) -> None:
    path = _feasible_path()
    step = 0.05
    samples = path.sample(step)

    target = tmp_path / "waypoints.csv"
    export_waypoints_csv(target, path, step=step)

    with target.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["x", "y", "theta_rad"]
    assert len(rows) == len(samples) + 1  # header + one row per sample
    assert all(row for row in rows)  # no blank interleaved rows (Windows newline check)


def test_csv_export_endpoints_match_samples(tmp_path: Path) -> None:
    path = _feasible_path()
    step = 0.05
    samples = path.sample(step)

    target = tmp_path / "waypoints.csv"
    export_waypoints_csv(target, path, step=step)

    with target.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    first = [float(v) for v in rows[1]]
    last = [float(v) for v in rows[-1]]
    assert first == pytest.approx(list(samples[0]), abs=1e-9)
    assert last == pytest.approx(list(samples[-1]), abs=1e-9)


def test_csv_export_zero_length_path_single_row(tmp_path: Path) -> None:
    path = _zero_length_path()
    samples = path.sample(0.05)
    assert len(samples) == 1  # sanity: the fixture really samples to one row

    target = tmp_path / "waypoints.csv"
    export_waypoints_csv(target, path)

    with target.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["x", "y", "theta_rad"]
    assert len(rows) == 2  # header + single data row
    data = [float(v) for v in rows[1]]
    assert data == pytest.approx(list(samples[0]), abs=1e-9)


def test_csv_export_accepts_str_path(tmp_path: Path) -> None:
    path = _feasible_path()
    target = tmp_path / "as_str.csv"

    export_waypoints_csv(str(target), path)

    assert target.exists()


@pytest.mark.parametrize("bad_step", [0.0, float("nan")], ids=["zero", "nan"])
def test_csv_export_invalid_step_raises_and_preserves_existing_file(
    tmp_path: Path, bad_step: float
) -> None:
    """An invalid sampling ``step`` raises ValueError without touching any file.

    Sampling happens before the temp file is opened, so a non-positive or
    non-finite ``step`` must surface the raw :class:`ValueError` from
    ``DubinsPath.sample`` and leave a pre-existing target -- and its directory --
    completely undisturbed (no truncation, no stray ``.tmp``).
    """
    path = _feasible_path()
    target = tmp_path / "waypoints.csv"

    # Establish a known-good file that must survive a rejected export.
    good = "x,y,theta_rad\n0.0,0.0,0.0\n"
    target.write_text(good, encoding="utf-8")

    with pytest.raises(ValueError):
        export_waypoints_csv(target, path, step=bad_step)

    # Existing file untouched, and no temp file created.
    assert target.read_text(encoding="utf-8") == good
    assert list(tmp_path.glob("*.tmp")) == []


def test_csv_export_partial_write_failure_preserves_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-write I/O failure must leave the prior good file intact, no .tmp behind."""
    path = _feasible_path()
    target = tmp_path / "waypoints.csv"

    # Establish a known-good file that must survive a failed re-export.
    good = "x,y,theta_rad\n0.0,0.0,0.0\n"
    target.write_text(good, encoding="utf-8")

    real_replace = os.replace

    def boom(src: object, dst: object, *args: object, **kwargs: object) -> None:
        raise OSError("simulated disk full")

    monkeypatch.setattr(os, "replace", boom)

    # The raw OSError propagates -- the caller (ui/app.py) handles OSError directly.
    with pytest.raises(OSError):
        export_waypoints_csv(target, path)

    # Original content untouched (atomic replace never landed) ...
    assert target.read_text(encoding="utf-8") == good
    # ... and no stray temp file was left behind.
    assert list(tmp_path.glob("*.tmp")) == []

    # Sanity: with os.replace restored, the export succeeds and overwrites.
    monkeypatch.setattr(os, "replace", real_replace)
    export_waypoints_csv(target, path)
    assert target.read_text(encoding="utf-8") != good
