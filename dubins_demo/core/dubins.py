"""Pure Dubins-path math: the six word solvers, sampling, and helpers.

This module is UI-free (no Tkinter, no matplotlib, no model imports). It uses
the canonical-frame method of Shkel & Lumelsky (2001): the start/goal pair is
transformed so the problem becomes ``(0, 0, alpha) -> (d, 0, beta)`` at unit
turn radius, each of the six words is solved in closed form, and the resulting
segment parameters are scaled back by the real turn radius.

Conventions (see ``core/angles``): positions in meters, headings in radians in
the math convention (0 = +X, counter-clockwise positive). An ``L`` arc turns
left (center 90 deg to the left of the heading, angular velocity positive); an
``R`` arc turns right; ``S`` is a straight segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dubins_demo.core.angles import normalize


class SegmentKind(Enum):
    """The three primitive segment kinds of a Dubins path."""

    L = "L"  # left arc
    S = "S"  # straight
    R = "R"  # right arc


class PathType(Enum):
    """The six Dubins words. Declaration order fixes iteration/display order."""

    LSL = "LSL"
    RSR = "RSR"
    LSR = "LSR"
    RSL = "RSL"
    RLR = "RLR"
    LRL = "LRL"

    @property
    def kinds(self) -> tuple[SegmentKind, SegmentKind, SegmentKind]:
        """The three :class:`SegmentKind` values spelled by this word."""
        a, b, c = self.value
        return (SegmentKind[a], SegmentKind[b], SegmentKind[c])


@dataclass(frozen=True)
class Config:
    """An oriented planar configuration: position (m) and heading (rad)."""

    x: float
    y: float
    theta: float


@dataclass(frozen=True)
class Segment:
    """A single path primitive with its arc length in meters (``>= 0``)."""

    kind: SegmentKind
    length: float

    def __post_init__(self) -> None:
        if self.length < 0.0:
            raise ValueError(f"segment length must be >= 0, got {self.length!r}")


@dataclass(frozen=True)
class Infeasible:
    """Marker for a word that does not exist for the current scenario."""

    path_type: PathType
    reason: str


def _advance(
    x: float, y: float, theta: float, kind: SegmentKind, s: float, radius: float
) -> tuple[float, float, float]:
    """Advance a configuration by arc length ``s`` along a single segment.

    Arcs are integrated as an exact rotation about the segment's circle center;
    straights are linear. Returns the new ``(x, y, theta)`` (theta not yet
    normalized so that turn accumulation stays continuous).
    """
    if kind is SegmentKind.S:
        return x + s * math.cos(theta), y + s * math.sin(theta), theta

    phi = s / radius  # swept angle magnitude
    if kind is SegmentKind.L:
        # center is 90 deg to the left of the heading; rotate by +phi
        cx = x - radius * math.sin(theta)
        cy = y + radius * math.cos(theta)
        new_theta = theta + phi
        nx = cx + radius * math.sin(new_theta)
        ny = cy - radius * math.cos(new_theta)
        return nx, ny, new_theta

    # SegmentKind.R: center 90 deg to the right; rotate by -phi
    cx = x + radius * math.sin(theta)
    cy = y - radius * math.cos(theta)
    new_theta = theta - phi
    nx = cx - radius * math.sin(new_theta)
    ny = cy + radius * math.cos(new_theta)
    return nx, ny, new_theta


@dataclass(frozen=True)
class DubinsPath:
    """A feasible Dubins path: exactly three segments at a fixed turn radius."""

    path_type: PathType
    segments: tuple[Segment, Segment, Segment]
    radius: float
    start: Config

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"turn radius must be positive, got {self.radius!r}")
        kinds = tuple(seg.kind for seg in self.segments)
        if kinds != self.path_type.kinds:
            raise ValueError(
                f"segment kinds {tuple(k.value for k in kinds)} do not match "
                f"the {self.path_type.value} word {tuple(k.value for k in self.path_type.kinds)}"
            )

    @property
    def length(self) -> float:
        """Total path length: the sum of the segment lengths."""
        return sum(seg.length for seg in self.segments)

    def sample(self, step: float = 0.05) -> np.ndarray:
        """Sample the path into an ``(N, 3)`` array of ``(x, y, theta)``.

        Points are spaced at most ``step`` meters apart along the arc length,
        and the final row is exactly the path's endpoint -- equal to the goal
        configuration up to solver tolerance -- since it is placed at the exact
        total arc length (headings are normalized to ``[0, 2*pi)``). Tests rely
        on both properties.
        """
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"sample step must be a positive finite value, got {step!r}")

        # Precompute the entry configuration of each segment.
        starts: list[tuple[float, float, float]] = []
        cx, cy, ct = self.start.x, self.start.y, self.start.theta
        for seg in self.segments:
            starts.append((cx, cy, ct))
            cx, cy, ct = _advance(cx, cy, ct, seg.kind, seg.length, self.radius)

        total = self.length
        if total <= 0.0:
            return np.array([[self.start.x, self.start.y, normalize(self.start.theta)]])

        # Global sample distances: 0, step, 2*step, ..., total (endpoint exact).
        n_intervals = max(1, math.ceil(total / step))
        distances = np.linspace(0.0, total, n_intervals + 1)

        cum = 0.0
        bounds: list[float] = []
        for seg in self.segments:
            cum += seg.length
            bounds.append(cum)

        out = np.empty((distances.size, 3))
        for i, g in enumerate(distances):
            seg_idx = 0
            while seg_idx < len(self.segments) - 1 and g > bounds[seg_idx]:
                seg_idx += 1
            offset = g - (bounds[seg_idx] - self.segments[seg_idx].length)
            sx, sy, st = starts[seg_idx]
            px, py, pt = _advance(sx, sy, st, self.segments[seg_idx].kind, offset, self.radius)
            out[i, 0] = px
            out[i, 1] = py
            out[i, 2] = normalize(pt)
        return out


# --- Closed-form word solvers in the canonical frame ------------------------
# Each returns (t, p, q) normalized segment parameters at unit radius, or None
# with a reason if the word is infeasible for the given (alpha, beta, d).


def _lsl(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0:
        return None
    tmp1 = math.atan2(math.cos(beta) - math.cos(alpha), tmp0)
    t = normalize(-alpha + tmp1)
    p = math.sqrt(p_sq)
    q = normalize(beta - tmp1)
    return t, p, q


def _rsr(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(beta) - math.sin(alpha))
    if p_sq < 0:
        return None
    tmp1 = math.atan2(math.cos(alpha) - math.cos(beta), tmp0)
    t = normalize(alpha - tmp1)
    p = math.sqrt(p_sq)
    q = normalize(-beta + tmp1)
    return t, p, q


def _lsr(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    p_sq = -2 + d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta))
    tmp -= math.atan2(-2.0, p)
    t = normalize(-alpha + tmp)
    q = normalize(-normalize(beta) + tmp)
    return t, p, q


def _rsl(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    p_sq = -2 + d * d + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0:
        return None
    p = math.sqrt(p_sq)
    tmp = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta))
    tmp -= math.atan2(2.0, p)
    t = normalize(alpha - tmp)
    q = normalize(beta - tmp)
    return t, p, q


def _rlr(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    tmp = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8
    if abs(tmp) > 1:
        return None
    p = normalize(-math.acos(tmp))  # middle arc, >= pi
    t = normalize(
        alpha
        - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
        + p / 2
    )
    q = normalize(alpha - beta - t + p)
    return t, p, q


def _lrl(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    tmp = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-math.sin(alpha) + math.sin(beta))) / 8
    if abs(tmp) > 1:
        return None
    p = normalize(-math.acos(tmp))  # middle arc, >= pi
    t = normalize(
        -alpha
        + math.atan2(-math.cos(alpha) + math.cos(beta), d + math.sin(alpha) - math.sin(beta))
        + p / 2
    )
    q = normalize(normalize(beta) - alpha - t + p)
    return t, p, q


_SOLVERS = {
    PathType.LSL: (_lsl, "outer tangent does not exist"),
    PathType.RSR: (_rsr, "outer tangent does not exist"),
    PathType.LSR: (_lsr, "inner tangent does not exist (turning circles overlap)"),
    PathType.RSL: (_rsl, "inner tangent does not exist (turning circles overlap)"),
    PathType.RLR: (_rlr, "turning-circle centers are more than 4r apart"),
    PathType.LRL: (_lrl, "turning-circle centers are more than 4r apart"),
}


def _canonical_frame(start: Config, goal: Config, radius: float) -> tuple[float, float, float]:
    """Return ``(alpha, beta, d)`` for the canonical unit-radius problem."""
    dx = goal.x - start.x
    dy = goal.y - start.y
    dist = math.hypot(dx, dy)
    d = dist / radius
    theta0 = math.atan2(dy, dx)
    alpha = normalize(start.theta - theta0)
    beta = normalize(goal.theta - theta0)
    return alpha, beta, d


def _solve_one(
    path_type: PathType, start: Config, goal: Config, radius: float
) -> DubinsPath | Infeasible:
    """Solve a single word, returning a path or an :class:`Infeasible`."""
    solver, default_reason = _SOLVERS[path_type]
    try:
        alpha, beta, d = _canonical_frame(start, goal, radius)
        result = solver(alpha, beta, d)
    except ValueError as exc:
        # A ValueError here is not legitimate geometric infeasibility (that is
        # signalled by a ``None`` return); it means an unexpected internal error
        # slipped past the closed-form guards. Surface it distinctly rather than
        # letting it raise, preserving FR-8/FR-24 (solvers must not raise).
        # ZeroDivisionError is intentionally not caught: solve_all guards
        # radius > 0, so it cannot occur; let it surface if it ever does.
        return Infeasible(path_type, f"INTERNAL: unexpected {type(exc).__name__}: {exc}")
    if result is None:
        return Infeasible(path_type, default_reason)
    t, p, q = result
    k0, k1, k2 = path_type.kinds
    segments = (
        Segment(k0, t * radius),
        Segment(k1, p * radius),
        Segment(k2, q * radius),
    )
    return DubinsPath(path_type=path_type, segments=segments, radius=radius, start=start)


def solve_all(
    start: Config, goal: Config, radius: float
) -> dict[PathType, DubinsPath | Infeasible]:
    """Solve all six Dubins words for the given scenario.

    Every word is guarded so a degenerate scenario yields an
    :class:`Infeasible` entry rather than raising (FR-8, FR-24). A non-positive
    radius makes every word infeasible.
    """
    if radius <= 0.0:
        reason = "turn radius must be positive"
        return {pt: Infeasible(pt, reason) for pt in PathType}
    return {pt: _solve_one(pt, start, goal, radius) for pt in PathType}


def shortest(solutions: dict[PathType, DubinsPath | Infeasible]) -> PathType | None:
    """Return the shortest feasible path type, or ``None`` if none exist."""
    best: PathType | None = None
    best_len = math.inf
    for pt, sol in solutions.items():
        if isinstance(sol, DubinsPath) and sol.length < best_len:
            best_len = sol.length
            best = pt
    return best


def turning_centers(cfg: Config, radius: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the ``(left, right)`` turning-circle centers for a configuration.

    The left circle center is 90 deg to the left of the heading, the right one
    90 deg to the right, each at distance ``radius``.
    """
    left = (cfg.x - radius * math.sin(cfg.theta), cfg.y + radius * math.cos(cfg.theta))
    right = (cfg.x + radius * math.sin(cfg.theta), cfg.y - radius * math.cos(cfg.theta))
    return left, right
