"""Tests for :mod:`dubins_demo.persistence.scenario_io`."""

from __future__ import annotations

import csv
import json
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


def test_non_numeric_coordinate_raises() -> None:
    document = scenario_to_dict(_make_scenario())
    document["start"]["x"] = "not a number"

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "start.x" in str(excinfo.value)


def test_invalid_display_enum_raises() -> None:
    document = scenario_to_dict(_make_scenario())
    document["display"]["angle_unit"] = "gradians"

    with pytest.raises(ScenarioError) as excinfo:
        dict_to_scenario(document)
    assert "gradians" in str(excinfo.value)


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


def test_csv_export_accepts_str_path(tmp_path: Path) -> None:
    path = _feasible_path()
    target = tmp_path / "as_str.csv"

    export_waypoints_csv(str(target), path)

    assert target.exists()
