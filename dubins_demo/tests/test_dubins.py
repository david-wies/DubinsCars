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


@pytest.mark.parametrize(
    "theta,expected_left,expected_right",
    [
        # Heading East: left is North (+y), right is South (-y).
        (0.0, (0.0, 1.0), (0.0, -1.0)),
        # Heading North: left is West (-x), right is East (+x).
        (math.pi / 2, (-1.0, 0.0), (1.0, 0.0)),
    ],
)
def test_turning_centers_geometry(
    theta: float,
    expected_left: tuple[float, float],
    expected_right: tuple[float, float],
) -> None:
    radius = 2.0
    cfg = Config(0.0, 0.0, theta)
    left, right = turning_centers(cfg, radius)
    assert left == pytest.approx((radius * expected_left[0], radius * expected_left[1]))
    assert right == pytest.approx((radius * expected_right[0], radius * expected_right[1]))
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


# --- Value-object validation -------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_x(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Config(bad, 0.0, 0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_y(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Config(0.0, bad, 0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_config_rejects_non_finite_theta(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Config(0.0, 0.0, bad)


@pytest.mark.parametrize("theta", [7.0, -1.0, 10.0 * math.pi])
def test_config_accepts_unnormalized_theta(theta: float) -> None:
    # theta is intentionally not normalized to [0, 2*pi); values outside that
    # range are stored unchanged (see Config docstring / _advance).
    cfg = Config(1.0, 2.0, theta)
    assert cfg.theta == theta


def test_segment_rejects_negative_length() -> None:
    with pytest.raises(ValueError, match="length"):
        Segment(SegmentKind.S, -0.1)


def test_segment_rejects_nan_length() -> None:
    with pytest.raises(ValueError, match="length"):
        Segment(SegmentKind.S, math.nan)


def test_dubins_path_rejects_non_positive_radius() -> None:
    left = Segment(SegmentKind.L, 1.0)
    straight = Segment(SegmentKind.S, 1.0)
    with pytest.raises(ValueError, match="radius"):
        DubinsPath(PathType.LSL, (left, straight, left), 0.0, Config(0.0, 0.0, 0.0))


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_dubins_path_rejects_non_finite_radius(bad: float) -> None:
    left = Segment(SegmentKind.L, 1.0)
    straight = Segment(SegmentKind.S, 1.0)
    with pytest.raises(ValueError, match="radius"):
        DubinsPath(PathType.LSL, (left, straight, left), bad, Config(0.0, 0.0, 0.0))


def test_dubins_path_rejects_segments_inconsistent_with_word() -> None:
    # An LSL path must spell L, S, L; three straights must be rejected.
    straight = Segment(SegmentKind.S, 1.0)
    with pytest.raises(ValueError, match="do not match"):
        DubinsPath(PathType.LSL, (straight, straight, straight), 1.0, Config(0.0, 0.0, 0.0))


def test_sample_rejects_non_positive_and_non_finite_step() -> None:
    start = Config(0.0, 0.0, 0.0)
    goal = Config(4.0, 0.0, 0.0)
    path = solve_all(start, goal, 1.0)[PathType.LSL]
    assert isinstance(path, DubinsPath)
    for bad in (0.0, -0.05, math.nan, math.inf):
        with pytest.raises(ValueError, match="step"):
            path.sample(bad)


# --- In-place U-turn (coincident position, reversed heading) -----------------


def test_in_place_uturn_is_a_feasible_ccc_maneuver() -> None:
    # Same position, opposite heading: distance d == 0 but headings differ, so
    # the turn must be made with a CCC (RLR/LRL) word.
    start = Config(2.0, 2.0, 0.0)
    goal = Config(2.0, 2.0, math.pi)
    sols = solve_all(start, goal, 1.5)
    ccc = [sols[PathType.RLR], sols[PathType.LRL]]
    feasible = [p for p in ccc if isinstance(p, DubinsPath)]
    assert feasible, "an in-place U-turn must be reachable by at least one CCC word"
    for path in feasible:
        end = path.sample(0.01)[-1]
        # The maneuver returns to the start point at the reversed heading.
        assert end[0] == pytest.approx(start.x, abs=1e-6)
        assert end[1] == pytest.approx(start.y, abs=1e-6)
        assert _angle_diff(end[2], normalize(goal.theta)) < 1e-6
        # A genuine three-arc maneuver, not a degenerate zero-length path.
        assert path.length > 0.0


# --- Independent geometric length oracles ------------------------------------
# The endpoint tests above prove each path *reaches* the goal, but an arc length
# wrong by a multiple of 2*pi*r would land on the same endpoint. These cross-
# check the solver's segment lengths against paths built from tangent-circle
# geometry, wholly independent of the closed-form canonical-frame solver.


def _fwd(start: Config, segs: list[tuple[str, float]], radius: float) -> tuple[float, float, float]:
    """Forward-integrate a ``(kind, length)`` list with plain trig."""
    x, y, th = start.x, start.y, start.theta
    for kind, length in segs:
        if kind == "S":
            x += length * math.cos(th)
            y += length * math.sin(th)
        elif kind == "L":
            cx, cy = x - radius * math.sin(th), y + radius * math.cos(th)
            th += length / radius
            x, y = cx + radius * math.sin(th), cy - radius * math.cos(th)
        else:  # "R"
            cx, cy = x + radius * math.sin(th), y - radius * math.cos(th)
            th -= length / radius
            x, y = cx - radius * math.sin(th), cy + radius * math.cos(th)
    return x, y, th


def _left_center(cfg: Config, r: float) -> tuple[float, float]:
    return (cfg.x - r * math.sin(cfg.theta), cfg.y + r * math.cos(cfg.theta))


def _right_center(cfg: Config, r: float) -> tuple[float, float]:
    return (cfg.x + r * math.sin(cfg.theta), cfg.y - r * math.cos(cfg.theta))


def _csc_oracle(word: PathType, start: Config, goal: Config, r: float) -> list[tuple[str, float]]:
    """Build a CSC (LSL/RSR/LSR/RSL) path from tangent geometry, solver-free."""
    if word is PathType.LSL:
        c1, c2 = _left_center(start, r), _left_center(goal, r)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        psi = math.atan2(dy, dx)  # outer-tangent travel direction
        return [
            ("L", r * normalize(psi - start.theta)),
            ("S", math.hypot(dx, dy)),
            ("L", r * normalize(goal.theta - psi)),
        ]
    if word is PathType.RSR:
        c1, c2 = _right_center(start, r), _right_center(goal, r)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        psi = math.atan2(dy, dx)
        return [
            ("R", r * normalize(start.theta - psi)),
            ("S", math.hypot(dx, dy)),
            ("R", r * normalize(psi - goal.theta)),
        ]
    # Inner tangent between opposite-sense circles (LSR / RSL).
    if word is PathType.LSR:
        c1, c2 = _left_center(start, r), _right_center(goal, r)
        sign, k1, k3 = +1.0, "L", "R"
    elif word is PathType.RSL:
        c1, c2 = _right_center(start, r), _left_center(goal, r)
        sign, k1, k3 = -1.0, "R", "L"
    else:
        raise ValueError(word)
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    dist = math.hypot(dx, dy)
    theta_c = math.atan2(dy, dx)
    psi = theta_c + sign * math.asin(2.0 * r / dist)
    straight = math.sqrt(dist * dist - 4.0 * r * r)
    # Arc sweep sign depends on turn direction: +delta for L (CCW), -delta for R.
    l1 = r * normalize(psi - start.theta) if k1 == "L" else r * normalize(start.theta - psi)
    l3 = r * normalize(goal.theta - psi) if k3 == "L" else r * normalize(psi - goal.theta)
    return [(k1, l1), ("S", straight), (k3, l3)]


_CSC_FIXED = [
    (Config(0.0, 0.0, 0.5), Config(9.0, 4.0, 2.0), 1.3),
    (Config(-3.0, 2.0, 1.2), Config(6.0, -5.0, -0.4), 2.1),
    (Config(1.0, 1.0, 0.0), Config(12.0, 3.0, 0.7), 1.0),
]


@pytest.mark.parametrize("start,goal,radius", _CSC_FIXED)
@pytest.mark.parametrize("word", [PathType.LSL, PathType.RSR, PathType.LSR, PathType.RSL])
def test_csc_segment_lengths_match_tangent_oracle(
    word: PathType, start: Config, goal: Config, radius: float
) -> None:
    sol = solve_all(start, goal, radius)[word]
    if not isinstance(sol, DubinsPath):
        pytest.skip(f"{word.value} infeasible for this fixture")
    segs = _csc_oracle(word, start, goal, radius)
    # Self-validate the oracle: its own segments must reconstruct the goal.
    ex, ey, eth = _fwd(start, segs, radius)
    assert ex == pytest.approx(goal.x, abs=1e-9)
    assert ey == pytest.approx(goal.y, abs=1e-9)
    assert _angle_diff(eth, goal.theta) < 1e-9
    # Cross-check the solver's per-segment lengths against the oracle.
    for seg, (_kind, length) in zip(sol.segments, segs, strict=True):
        assert seg.length == pytest.approx(length, abs=1e-9)


_FAR_FIXED = [
    (Config(0.0, 0.0, 0.3), Config(40.0, 12.0, 2.4), 1.5),
    (Config(-5.0, 3.0, -1.0), Config(35.0, -20.0, 0.9), 2.0),
]


@pytest.mark.parametrize("start,goal,radius", _FAR_FIXED)
def test_shortest_matches_min_csc_for_far_goals(start: Config, goal: Config, radius: float) -> None:
    # For widely separated configurations the optimum is always a CSC word, so
    # shortest() must agree with a brute-force minimum over the independently
    # constructed CSC candidate lengths.
    sols = solve_all(start, goal, radius)
    best = shortest(sols)
    assert best in {PathType.LSL, PathType.RSR, PathType.LSR, PathType.RSL}

    oracle_min = math.inf
    for word in (PathType.LSL, PathType.RSR, PathType.LSR, PathType.RSL):
        if not isinstance(sols[word], DubinsPath):
            continue
        segs = _csc_oracle(word, start, goal, radius)
        ex, ey, eth = _fwd(start, segs, radius)
        assert ex == pytest.approx(goal.x, abs=1e-9)  # oracle self-check
        assert ey == pytest.approx(goal.y, abs=1e-9)
        assert _angle_diff(eth, goal.theta) < 1e-9
        oracle_min = min(oracle_min, sum(length for _kind, length in segs))

    best_sol = sols[best]
    assert isinstance(best_sol, DubinsPath)
    assert best_sol.length == pytest.approx(oracle_min, abs=1e-9)
