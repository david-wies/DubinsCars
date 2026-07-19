"""Scenario JSON save/load and CSV waypoint export.

This layer is pure I/O plus (de)serialization. It is UI-free: it imports no
Tkinter and no matplotlib, working only with the core types (:class:`Config`,
:class:`FixedRadius`, :class:`Convention`, :class:`Unit`, :class:`DubinsPath`)
and plain dicts / strings / paths.

The JSON schema (spec EXT-4, version 1)::

    {
      "version": 1,
      "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
      "goal":  {"x": 10.0, "y": 5.0, "theta": 1.57},
      "radius_policy": {"type": "fixed", "value": 2.0},
      "display": {"heading_convention": "angle", "angle_unit": "deg"}
    }

``theta`` is always stored canonically: radians in the math convention
(0 = +X, counter-clockwise positive), exactly as held by :class:`Config`.

Loading validates the whole document before returning anything. Any malformed
input -- bad JSON, a missing or wrong-typed key, an unknown radius-policy type,
or a wrong ``version`` -- is reported as a single :class:`ScenarioError` with a
helpful message; a bare ``KeyError`` / ``ValueError`` never reaches the caller.
Because :func:`load_scenario` builds a fresh value object and mutates nothing,
the caller (the UI) can turn a :class:`ScenarioError` into an error dialog and
leave its live model untouched.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dubins_demo.core.dubins import Config, DubinsPath
from dubins_demo.core.model import Convention, FixedRadius, RadiusPolicy, Scenario, Unit

SCHEMA_VERSION = 1
"""The only scenario-file schema version this module reads or writes."""


class ScenarioError(Exception):
    """A scenario file is malformed, unsupported, or otherwise unloadable.

    Raised by :func:`load_scenario` / :func:`dict_to_scenario` for every kind
    of bad input so callers only have to catch one exception type.
    """


@dataclass(frozen=True)
class LoadedScenario:
    """The scenario fields recovered from a JSON document.

    This is a plain value object, deliberately *not* a live :class:`Scenario`:
    :func:`load_scenario` never touches a running model, so a failed load
    cannot partially mutate application state. The caller applies it in one
    shot via :meth:`Scenario.update`, e.g.::

        loaded = load_scenario(path)
        scenario.update(**loaded.to_update_kwargs())
    """

    start: Config
    goal: Config
    radius_policy: RadiusPolicy
    heading_convention: Convention
    angle_unit: Unit

    def to_update_kwargs(self) -> dict[str, object]:
        """Return kwargs suitable for a single :meth:`Scenario.update` call."""
        return {
            "start": self.start,
            "goal": self.goal,
            "radius_policy": self.radius_policy,
            "heading_convention": self.heading_convention,
            "angle_unit": self.angle_unit,
        }


# --- Serialization (in-memory) ---------------------------------------------


def _config_to_dict(cfg: Config) -> dict[str, float]:
    return {"x": cfg.x, "y": cfg.y, "theta": cfg.theta}


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Serialize a scenario's persistent fields to a JSON-ready dict.

    Reads ``start``, ``goal``, ``radius_policy``, ``heading_convention`` and
    ``angle_unit`` from ``scenario``. Only :class:`FixedRadius` policies are
    supported; any other policy raises :class:`ScenarioError` rather than
    silently dropping data.
    """
    policy = scenario.radius_policy
    if not isinstance(policy, FixedRadius):
        raise ScenarioError(
            f"cannot serialize radius policy of type {type(policy).__name__!r}; "
            "only 'fixed' policies are supported"
        )
    return {
        "version": SCHEMA_VERSION,
        "start": _config_to_dict(scenario.start),
        "goal": _config_to_dict(scenario.goal),
        "radius_policy": {"type": "fixed", "value": policy.value},
        "display": {
            "heading_convention": scenario.heading_convention.value,
            "angle_unit": scenario.angle_unit.value,
        },
    }


def _require_mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioError(f"{where} must be a JSON object, got {type(value).__name__}")
    return value


def _require_key(mapping: dict[str, Any], key: str, where: str) -> Any:
    try:
        return mapping[key]
    except KeyError:
        raise ScenarioError(f"missing required key {key!r} in {where}") from None


def _require_number(value: object, where: str) -> float:
    # bool is an int subclass; reject it so ``true``/``false`` are not coordinates.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{where} must be a number, got {type(value).__name__}")
    number = float(value)
    # ``json.loads`` accepts NaN/Infinity by default; those are never valid here.
    if not math.isfinite(number):
        raise ScenarioError(f"{where} must be a finite number, got {number!r}")
    return number


def _config_from_dict(value: object, where: str) -> Config:
    mapping = _require_mapping(value, where)
    return Config(
        x=_require_number(_require_key(mapping, "x", where), f"{where}.x"),
        y=_require_number(_require_key(mapping, "y", where), f"{where}.y"),
        theta=_require_number(_require_key(mapping, "theta", where), f"{where}.theta"),
    )


def _radius_policy_from_dict(value: object) -> RadiusPolicy:
    mapping = _require_mapping(value, "radius_policy")
    policy_type = _require_key(mapping, "type", "radius_policy")
    if policy_type != "fixed":
        raise ScenarioError(f"unknown radius_policy type {policy_type!r} (expected 'fixed')")
    value_num = _require_number(
        _require_key(mapping, "value", "radius_policy"), "radius_policy.value"
    )
    try:
        return FixedRadius(value=value_num)
    except ValueError as exc:
        raise ScenarioError(f"invalid radius_policy.value: {exc}") from exc


def _display_from_dict(value: object) -> tuple[Convention, Unit]:
    mapping = _require_mapping(value, "display")
    conv_raw = _require_key(mapping, "heading_convention", "display")
    unit_raw = _require_key(mapping, "angle_unit", "display")
    try:
        convention = Convention(conv_raw)
    except ValueError:
        valid = ", ".join(repr(c.value) for c in Convention)
        raise ScenarioError(
            f"invalid heading_convention {conv_raw!r} (expected one of {valid})"
        ) from None
    try:
        unit = Unit(unit_raw)
    except ValueError:
        valid = ", ".join(repr(u.value) for u in Unit)
        raise ScenarioError(f"invalid angle_unit {unit_raw!r} (expected one of {valid})") from None
    return convention, unit


def dict_to_scenario(data: object) -> LoadedScenario:
    """Validate a parsed JSON document and return a :class:`LoadedScenario`.

    Every failure mode -- non-object root, missing key, wrong-typed field,
    unknown radius-policy type, or unsupported ``version`` -- raises
    :class:`ScenarioError`. Nothing is mutated, so a rejected document leaves
    any caller-held model untouched.
    """
    mapping = _require_mapping(data, "scenario")

    version = _require_key(mapping, "version", "scenario")
    # ``type(...) is int`` rejects ``True``/``False`` (bool == int subclass, and
    # ``True == 1`` would otherwise sneak past the version gate).
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ScenarioError(
            f"unsupported scenario version {version!r} (this build reads version {SCHEMA_VERSION})"
        )

    start = _config_from_dict(_require_key(mapping, "start", "scenario"), "start")
    goal = _config_from_dict(_require_key(mapping, "goal", "scenario"), "goal")
    radius_policy = _radius_policy_from_dict(_require_key(mapping, "radius_policy", "scenario"))
    convention, unit = _display_from_dict(_require_key(mapping, "display", "scenario"))

    return LoadedScenario(
        start=start,
        goal=goal,
        radius_policy=radius_policy,
        heading_convention=convention,
        angle_unit=unit,
    )


# --- File I/O ---------------------------------------------------------------


def save_scenario(scenario: Scenario, path: str | os.PathLike[str]) -> None:
    """Write ``scenario`` to ``path`` as pretty-printed (indent=2) UTF-8 JSON.

    The write is atomic: the document is streamed to a sibling ``.tmp`` file in
    the same directory and then :func:`os.replace`-d onto ``path`` (atomic on
    POSIX and Windows). A mid-write failure therefore never truncates a
    previously-good file, and no stray temp file is left behind -- both the
    serialization step (which happens first) and any I/O failure surface as a
    single :class:`ScenarioError`.
    """
    document = scenario_to_dict(scenario)
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2)
            fh.write("\n")  # POSIX-friendly trailing newline
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ScenarioError(f"could not write scenario file {target}: {exc}") from exc


def load_scenario(path: str | os.PathLike[str]) -> LoadedScenario:
    """Read and validate a scenario file, returning a :class:`LoadedScenario`.

    Every failure mode is wrapped into a single :class:`ScenarioError`: an
    unreadable file (:class:`OSError`), an undecodable byte stream
    (:class:`UnicodeError`), malformed JSON, and every schema violation. Callers
    handle exactly one exception type and can leave their live model untouched.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ScenarioError(f"could not read scenario file {source}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{source} is not valid JSON: {exc}") from exc
    return dict_to_scenario(data)


def export_waypoints_csv(
    path: str | os.PathLike[str], path_obj: DubinsPath, step: float = 0.05
) -> None:
    """Export a Dubins path's sampled waypoints to CSV.

    Writes a header row ``x,y,theta_rad`` followed by one row per sample from
    ``path_obj.sample(step)`` (``theta`` in radians, math convention). The file
    is opened with ``newline=""`` -- the standard-library csv idiom that keeps
    Windows from inserting blank rows between records.

    Sampling happens before any file is touched, so a :class:`ValueError` from
    an invalid ``step`` surfaces without disturbing an existing file. The write
    itself is atomic (mirroring :func:`save_scenario`): rows are written to a
    sibling ``.tmp`` file and then :func:`os.replace`-d onto ``path`` (atomic on
    POSIX and Windows), so a mid-write failure never truncates a previously-good
    file and leaves no stray temp file behind. Unlike :func:`save_scenario`, an
    I/O failure propagates as the raw :class:`OSError` -- the caller
    (``ui/app.py``) already handles ``OSError`` directly.
    """
    samples = path_obj.sample(step)
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "theta_rad"])
            for x, y, theta in samples:
                writer.writerow([float(x), float(y), float(theta)])
        os.replace(tmp, target)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
