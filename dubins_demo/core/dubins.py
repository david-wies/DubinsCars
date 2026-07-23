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
from collections.abc import Mapping
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


# Default absolute tolerance (meters / radians) for :meth:`Config.approx`.
_CONFIG_EPS = 1e-9


@dataclass(frozen=True)
class Config:
    """An oriented planar configuration: position (m) and heading (rad).

    ``theta`` is normalized to ``[0, 2*pi)`` on construction, so two configs
    built from headings that differ by a multiple of ``2*pi`` are identical.

    ``==`` is exact structural equality (and ``Config`` is hashable): it means
    "the same stored values", is transitive, and is safe as a dict key or set
    member. It does *not* absorb floating-point noise -- for a tolerant
    "same pose" test use :meth:`approx`, which compares within a tolerance and
    treats the heading on the circle so the ``0``/``2*pi`` seam is not a cliff.
    Tolerance lives there, opt-in, rather than in ``==`` where it would break
    transitivity.

    Normalization does not affect continuous angle accumulation across arcs --
    see :func:`_advance`, which never constructs a ``Config``.

    Equality/hashing rely on ``theta`` staying strictly below ``2*pi`` after
    normalization; see the edge-case guard in :func:`angles.normalize`.
    """

    x: float
    y: float
    theta: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x) and math.isfinite(self.y) and math.isfinite(self.theta)):
            raise ValueError(
                f"config components must be finite, got "
                f"x={self.x!r}, y={self.y!r}, theta={self.theta!r}"
            )
        # Frozen dataclass: bypass the immutability guard to canonicalize theta.
        object.__setattr__(self, "theta", normalize(self.theta))

    def approx(self, other: Config, *, tol: float = _CONFIG_EPS) -> bool:
        """Return whether *other* is the same pose within *tol*.

        Positions and headings must each differ by strictly less than *tol*
        (a difference of exactly *tol* is not approx-equal); the heading is
        compared as the shortest arc on the circle, so headings straddling
        the ``0``/``2*pi`` seam are not spuriously far apart. *tol* applies
        uniformly to both position (meters) and heading (radians) -- a value
        meaningful for one is not automatically meaningful for the other.

        Not transitive: ``a.approx(b)`` and ``b.approx(c)`` do not imply
        ``a.approx(c)``. Do not chain calls to build equivalence classes
        (e.g. deduplicating a list of near-identical poses).
        """
        if tol <= 0:
            raise ValueError(f"tol must be positive, got tol={tol!r}")
        dtheta = abs(self.theta - other.theta)
        dtheta = min(dtheta, math.tau - dtheta)  # shortest arc across the seam
        return abs(self.x - other.x) < tol and abs(self.y - other.y) < tol and dtheta < tol


@dataclass(frozen=True)
class Segment:
    """A single path primitive with its arc length in meters (``>= 0``)."""

    kind: SegmentKind
    length: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length < 0.0:
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
        if not math.isfinite(self.radius) or self.radius <= 0.0:
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
# Each returns (t, p, q) at unit radius, or bare None if the word is
# infeasible for the given (alpha, beta, d); the human-readable reason is
# attached separately in _solve_one from the _SOLVERS table below, not
# carried by the return value. t and q are normalized to [0, 2*pi); for the
# CSC words (LSL/RSR/LSR/RSL) p is an unbounded straight-segment length
# (p = sqrt(p_sq)), not a normalized angle -- only the CCC words (RLR/LRL)
# use p as a normalized middle-arc angle. LSL/RSR are always feasible and
# never return None (see the p_sq comments in _lsl/_rsr below); only
# LSR/RSL/RLR/LRL can.


def _lsl(alpha: float, beta: float, d: float) -> tuple[float, float, float]:
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    # p_sq is algebraically (d + sin a - sin b)^2 + (cos b - cos a)^2, a sum of
    # squares, so the outer tangent always exists (LSL is feasible for every
    # scenario). Computed in expanded form it can round to a tiny negative when
    # the start/goal left circles nearly coincide (p ~ 0); clamp to 0 rather
    # than reporting a false infeasibility.
    p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))
    tmp1 = math.atan2(math.cos(beta) - math.cos(alpha), tmp0)
    t = normalize(-alpha + tmp1)
    p = math.sqrt(max(0.0, p_sq))
    q = normalize(beta - tmp1)
    return t, p, q


def _rsr(alpha: float, beta: float, d: float) -> tuple[float, float, float]:
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    # See _lsl: p_sq is a sum of squares, so RSR is always feasible; clamp a
    # rounding-induced tiny negative to 0 instead of returning None.
    p_sq = 2 + d * d - 2 * math.cos(alpha - beta) + 2 * d * (math.sin(beta) - math.sin(alpha))
    tmp1 = math.atan2(math.cos(alpha) - math.cos(beta), tmp0)
    t = normalize(alpha - tmp1)
    p = math.sqrt(max(0.0, p_sq))
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
    q = normalize(-beta + tmp)
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
    p = normalize(-math.acos(tmp))  # CCC middle arc angle, a major arc in [pi, 2*pi)
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
    p = normalize(-math.acos(tmp))  # CCC middle arc angle, a major arc in [pi, 2*pi)
    t = normalize(
        -alpha
        + math.atan2(-math.cos(alpha) + math.cos(beta), d + math.sin(alpha) - math.sin(beta))
        + p / 2
    )
    q = normalize(beta - alpha - t + p)
    return t, p, q


_SOLVERS = {
    # LSL/RSR solvers never return None (see the block comment above the
    # solvers), so these two reason strings are unreachable dead data --
    # kept only for tuple-shape symmetry with the other four entries.
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
        # A ``None`` result is legitimate geometric infeasibility, not an error;
        # return the normal reason before any construction that could raise.
        if result is None:
            return Infeasible(path_type, default_reason)
        t, p, q = result
        k0, k1, k2 = path_type.kinds
        # Segment/DubinsPath __post_init__ enforce finiteness/non-negativity and
        # word-kind agreement; their construction stays inside this try so a
        # ValueError from those guards is caught by the INTERNAL net below rather
        # than escaping (it would otherwise propagate through solve_all into a
        # Tk callback).
        segments = (
            Segment(k0, t * radius),
            Segment(k1, p * radius),
            Segment(k2, q * radius),
        )
        return DubinsPath(path_type=path_type, segments=segments, radius=radius, start=start)
    except ValueError as exc:
        # A ValueError here is not legitimate geometric infeasibility (that is
        # signalled by the ``None`` return above); it means an unexpected
        # internal error slipped past the closed-form guards. Surface it
        # distinctly rather than letting it raise, preserving FR-8/FR-24
        # (solvers must not raise). ZeroDivisionError is intentionally not
        # caught: solve_all guards radius > 0, so it cannot occur; let it
        # surface if it ever does.
        return Infeasible(path_type, f"INTERNAL: unexpected {type(exc).__name__}: {exc}")


def solve_all(
    start: Config, goal: Config, radius: float
) -> dict[PathType, DubinsPath | Infeasible]:
    """Solve all six Dubins words for the given scenario.

    Every word is guarded so a degenerate scenario yields an
    :class:`Infeasible` entry rather than raising (FR-8, FR-24). A radius that
    is not a positive, finite number makes every word infeasible.
    """
    # ``not (radius > 0.0)`` traps zero, negatives, and NaN (all comparisons
    # with NaN are False); the explicit ``isfinite`` check also traps +inf,
    # which would otherwise pass the ``> 0`` test and later raise when a
    # non-finite segment length is constructed (violating FR-8/FR-24).
    if not (radius > 0.0) or not math.isfinite(radius):
        reason = "turn radius must be a positive, finite number"
        return {pt: Infeasible(pt, reason) for pt in PathType}
    return {pt: _solve_one(pt, start, goal, radius) for pt in PathType}


def shortest(solutions: Mapping[PathType, DubinsPath | Infeasible]) -> PathType | None:
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
