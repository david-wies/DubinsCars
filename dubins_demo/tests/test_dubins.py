"""Tests for :mod:`dubins_demo.core.dubins`."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from dubins_demo.core.angles import normalize
from dubins_demo.core.dubins import (
    Config,
    DubinsPath,
    Infeasible,
    PathType,
    Segment,
    SegmentKind,
    shortest,
    solve_all,
    turning_centers,
)

_TAU = 2.0 * math.pi
_MIRROR = {
    PathType.LSL: PathType.RSR,
    PathType.RSR: PathType.LSL,
    PathType.LSR: PathType.RSL,
    PathType.RSL: PathType.LSR,
    PathType.RLR: PathType.LRL,
    PathType.LRL: PathType.RLR,
}


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (radians)."""
    return abs((a - b + math.pi) % _TAU - math.pi)


def _random_scenarios(seed: int, n: int) -> list[tuple[Config, Config, float]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        start = Config(rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(0, _TAU))
        goal = Config(rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(0, _TAU))
        radius = rng.uniform(0.3, 6.0)
        out.append((start, goal, radius))
    return out


def test_path_type_kinds() -> None:
    assert PathType.LSL.kinds == (SegmentKind.L, SegmentKind.S, SegmentKind.L)
    assert PathType.RLR.kinds == (SegmentKind.R, SegmentKind.L, SegmentKind.R)
    assert PathType.LSR.kinds == (SegmentKind.L, SegmentKind.S, SegmentKind.R)


def test_solve_all_returns_all_six() -> None:
    sols = solve_all(Config(0, 0, 0), Config(10, 5, 1.0), 2.0)
    assert set(sols) == set(PathType)


def test_non_positive_radius_all_infeasible() -> None:
    for radius in (0.0, -1.0):
        sols = solve_all(Config(0, 0, 0), Config(3, 4, 1.0), radius)
        assert all(isinstance(s, Infeasible) for s in sols.values())


# --- Endpoint property: the correctness oracle ------------------------------


@pytest.mark.parametrize("scenario", _random_scenarios(seed=1234, n=400))
def test_sample_endpoint_matches_goal(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    sols = solve_all(start, goal, radius)
    feasible = [s for s in sols.values() if isinstance(s, DubinsPath)]
    assert feasible, "expected at least one feasible path for a generic scenario"
    for path in feasible:
        end = path.sample(0.05)[-1]
        assert end[0] == pytest.approx(goal.x, abs=1e-6)
        assert end[1] == pytest.approx(goal.y, abs=1e-6)
        assert _angle_diff(end[2], normalize(goal.theta)) < 1e-6


def test_sample_start_matches_start() -> None:
    start = Config(1.0, -2.0, 0.7)
    goal = Config(6.0, 3.0, 2.1)
    for path in solve_all(start, goal, 1.5).values():
        if isinstance(path, DubinsPath):
            first = path.sample(0.05)[0]
            assert first[0] == pytest.approx(start.x, abs=1e-9)
            assert first[1] == pytest.approx(start.y, abs=1e-9)
            assert _angle_diff(first[2], normalize(start.theta)) < 1e-9


# --- Length / segment consistency -------------------------------------------


@pytest.mark.parametrize("scenario", _random_scenarios(seed=99, n=200))
def test_length_equals_segment_sum(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    for path in solve_all(start, goal, radius).values():
        if isinstance(path, DubinsPath):
            assert len(path.segments) == 3
            assert all(seg.length >= -1e-12 for seg in path.segments)
            assert path.length == pytest.approx(sum(s.length for s in path.segments))


# --- Sampling continuity -----------------------------------------------------


@pytest.mark.parametrize("scenario", _random_scenarios(seed=7, n=100))
def test_sample_continuity(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    step = 0.05
    for path in solve_all(start, goal, radius).values():
        if isinstance(path, DubinsPath) and path.length > 0:
            pts = path.sample(step)
            deltas = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
            # Chord length between samples never exceeds the arc step.
            assert float(deltas.max(initial=0.0)) <= step + 1e-9


def test_sample_step_density() -> None:
    path = next(
        p
        for p in solve_all(Config(0, 0, 0), Config(20, 0, 0), 2.0).values()
        if isinstance(p, DubinsPath)
    )
    pts = path.sample(0.05)
    assert pts.shape[1] == 3
    assert pts.shape[0] >= math.ceil(path.length / 0.05)


def test_zero_length_path_samples_single_point() -> None:
    start = Config(2.0, 2.0, 1.0)
    # Start == goal, same heading: the RSR/LSL word collapses to zero length.
    sols = solve_all(start, Config(2.0, 2.0, 1.0), 1.5)
    zero = [
        p for p in sols.values() if isinstance(p, DubinsPath) and p.length == pytest.approx(0.0)
    ]
    assert zero
    pts = zero[0].sample(0.05)
    assert pts.shape == (1, 3)
    assert pts[-1, 0] == pytest.approx(start.x)
    assert pts[-1, 1] == pytest.approx(start.y)


# --- Rigid-transform invariance ---------------------------------------------


def _rigid(cfg: Config, ang: float, tx: float, ty: float) -> Config:
    c, s = math.cos(ang), math.sin(ang)
    return Config(c * cfg.x - s * cfg.y + tx, s * cfg.x + c * cfg.y + ty, cfg.theta + ang)


def _mirror(cfg: Config) -> Config:
    """Reflect across the x-axis: swaps left and right turns."""
    return Config(cfg.x, -cfg.y, -cfg.theta)


@pytest.mark.parametrize("scenario", _random_scenarios(seed=55, n=150))
def test_rigid_transform_preserves_lengths(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    rng = random.Random(hash((start.x, goal.y, radius)) & 0xFFFF)
    ang, tx, ty = rng.uniform(0, _TAU), rng.uniform(-8, 8), rng.uniform(-8, 8)
    base = solve_all(start, goal, radius)
    moved = solve_all(_rigid(start, ang, tx, ty), _rigid(goal, ang, tx, ty), radius)
    for pt in PathType:
        a, b = base[pt], moved[pt]
        assert isinstance(a, DubinsPath) == isinstance(b, DubinsPath)
        if isinstance(a, DubinsPath) and isinstance(b, DubinsPath):
            assert a.length == pytest.approx(b.length, abs=1e-9)


@pytest.mark.parametrize("scenario", _random_scenarios(seed=56, n=150))
def test_mirror_swaps_left_right_types(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    base = solve_all(start, goal, radius)
    mirrored = solve_all(_mirror(start), _mirror(goal), radius)
    for pt in PathType:
        a, b = base[pt], mirrored[_MIRROR[pt]]
        assert isinstance(a, DubinsPath) == isinstance(b, DubinsPath)
        if isinstance(a, DubinsPath) and isinstance(b, DubinsPath):
            assert a.length == pytest.approx(b.length, abs=1e-9)


# --- CCC existence boundary --------------------------------------------------


def test_ccc_existence_boundary() -> None:
    # Two configurations with parallel headings a distance apart on the x-axis.
    # The RLR/LRL words exist only when the turning-circle centers are within 4r.
    radius = 1.0
    theta = 0.0

    def ccc_feasible(sep: float) -> bool:
        start = Config(0.0, 0.0, theta)
        goal = Config(sep, 0.0, theta)
        sols = solve_all(start, goal, radius)
        return isinstance(sols[PathType.RLR], DubinsPath) or isinstance(
            sols[PathType.LRL], DubinsPath
        )

    # Centers for same-heading configs on the x-axis are separated by exactly sep,
    # so the CCC limit is sep = 4r.
    assert ccc_feasible(3.5 * radius)  # comfortably below 4r
    assert not ccc_feasible(4.5 * radius)  # beyond 4r
    # Just below vs just above the 4r threshold behaves consistently.
    assert ccc_feasible(4.0 * radius - 1e-6)
    assert not ccc_feasible(4.0 * radius + 1e-3)


# --- shortest() --------------------------------------------------------------


@pytest.mark.parametrize("scenario", _random_scenarios(seed=321, n=200))
def test_shortest_returns_min_length_feasible(scenario: tuple[Config, Config, float]) -> None:
    start, goal, radius = scenario
    sols = solve_all(start, goal, radius)
    best = shortest(sols)
    feasible = {pt: s for pt, s in sols.items() if isinstance(s, DubinsPath)}
    if not feasible:
        assert best is None
        return
    assert best in feasible
    min_len = min(s.length for s in feasible.values())
    assert feasible[best].length == pytest.approx(min_len)


def test_shortest_none_when_all_infeasible() -> None:
    sols = solve_all(Config(0, 0, 0), Config(1, 1, 1), 0.0)
    assert shortest(sols) is None


# --- turning_centers ---------------------------------------------------------


def test_turning_centers_geometry() -> None:
    radius = 2.0
    cfg = Config(0.0, 0.0, 0.0)  # heading East
    left, right = turning_centers(cfg, radius)
    # Left of East is North (+y), right is South (-y).
    assert left == pytest.approx((0.0, radius))
    assert right == pytest.approx((0.0, -radius))
    for center in (left, right):
        assert math.hypot(center[0] - cfg.x, center[1] - cfg.y) == pytest.approx(radius)


def test_known_straight_line_scenario() -> None:
    # Goal directly ahead: the LSL/RSR straight-through path has length == distance,
    # with zero-length arcs.
    start = Config(0.0, 0.0, 0.0)
    goal = Config(10.0, 0.0, 0.0)
    sols = solve_all(start, goal, 2.0)
    lsl = sols[PathType.LSL]
    assert isinstance(lsl, DubinsPath)
    assert lsl.length == pytest.approx(10.0)
    assert lsl.segments[1].kind is SegmentKind.S
    assert lsl.segments[1].length == pytest.approx(10.0)
    assert lsl.segments[0].length == pytest.approx(0.0)
    assert lsl.segments[2].length == pytest.approx(0.0)


def test_segment_immutability() -> None:
    seg = Segment(SegmentKind.S, 3.0)
    with pytest.raises((AttributeError, TypeError)):
        seg.length = 4.0  # type: ignore[misc]
