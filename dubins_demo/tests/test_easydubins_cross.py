"""Cross-validation of the solver against the independent ``easydubins`` package.

``easydubins`` (https://pypi.org/project/easydubins/) is a third-party, wholly
separate implementation of the same Shkel & Lumelsky (2001) closed forms. It is
an *optional* dependency in the ``crosscheck`` extra
(``pip install -e ".[crosscheck]"``); if it is missing the whole module skips
rather than fails.

Two levels of agreement are asserted:

* **Full pipeline** -- our ``solve_all`` + ``shortest`` against
  ``easydubins.dubin_path.dubins_path``. Each library performs its *own*
  world-to-canonical-frame transform, so this cross-checks the frame math and
  the word selection end to end, black box.
* **Per-word closed form** -- our six word solvers against
  ``easydubins.dubin_path.general_planner`` on the *same* canonical
  ``(alpha, beta, d)``. This pins the individual closed forms, including which
  words are geometrically infeasible for a given scenario. The per-word level
  feeds our canonical ``(alpha, beta, d)`` directly to easydubins, so it relies
  on easydubins sharing that exact convention, guarded by the
  ``easydubins>=1.3,<2`` version pin in ``pyproject.toml``.

Agreement between the two independent implementations is exact to floating
point (observed max diff ~1e-13 over thousands of cases), so ``1e-9`` is a
comfortably tight tolerance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import pytest

from dubins_demo.core import dubins
from dubins_demo.core.dubins import Config, DubinsPath, PathType, shortest, solve_all

ed = pytest.importorskip(
    "easydubins.dubin_path",
    reason="easydubins is an optional dependency (pip install -e '.[crosscheck]')",
)

_ABS_TOL = 1e-9

# A handful of hand-picked scenarios plus a deterministic random spread. Fixed
# cases cover coincident start/goal, pure translation, and antipodal headings;
# the random cases sweep positions, headings, and radius. Seeded locally so the
# list is stable and does not perturb the global RNG.
_FIXED_CASES: list[tuple[Config, Config, float]] = [
    (Config(0.0, 0.0, 0.0), Config(4.0, 0.0, 0.0), 1.0),
    (Config(0.0, 0.0, 0.0), Config(0.0, 0.0, math.pi / 2), 1.0),
    (Config(0.0, 0.0, 0.0), Config(4.0, 0.0, math.pi), 1.0),
    (Config(-2.0, 1.0, 1.3), Config(3.0, -2.5, -0.7), 0.8),
    (Config(0.0, 0.0, 0.0), Config(1.0, 1.0, math.pi / 4), 2.5),
    (Config(5.0, 5.0, math.pi), Config(-5.0, -5.0, 0.0), 1.7),
]


def test_easydubins_general_planner_convention_is_stable() -> None:
    """Pin easydubins's ``general_planner`` calling convention (guards issue #3).

    The per-word cross-check feeds our canonical ``(alpha, beta, d)`` straight
    into ``general_planner``, so it silently depends on easydubins sharing that
    exact convention. The ``easydubins>=1.3,<2`` pin allows in-range minors that
    could in principle re-interpret the arguments (arg order, degrees vs radians,
    a sign flip). These golden values -- read from the pinned version and
    independent of our own solver -- freeze that convention, so any such drift
    fails loudly here instead of silently corrupting the per-word comparison.

    * A symmetric straight anchor ``(0, 0, d) -> (0, d, 0)`` pins the ``d``
      argument position and confirms lengths are unit-radius radians, not degrees.
    * An asymmetric anchor additionally pins the ``alpha`` vs ``beta`` order: an
      arg transpose changes ``LSL`` and ``RSR`` here (the symmetric anchor alone
      would not catch it).
    """
    for word in ("LSL", "RSR"):
        straight = ed.general_planner(word, 0.0, 0.0, 2.0)
        assert straight is not None
        assert straight[0] == pytest.approx([0.0, 2.0, 0.0], abs=_ABS_TOL)

    golden = {
        "LSL": [5.640035023465776, 1.491253824037104, 1.4431502837138102],
        "RSR": [0.10905718280959825, 2.6437350841009115, 5.374128124369988],
        "LSR": [6.178286521505711, 2.853125248843, 5.378286521505711],
    }
    for word, expected in golden.items():
        result = ed.general_planner(word, 0.3, 1.1, 2.0)
        assert result is not None, f"{word} unexpectedly infeasible for the convention anchor"
        assert result[0] == pytest.approx(expected, abs=_ABS_TOL), (
            f"{word} convention drift: easydubins changed its canonical frame or arg order"
        )


def _random_cases(n: int, seed: int) -> list[tuple[Config, Config, float]]:
    rng = random.Random(seed)
    cases: list[tuple[Config, Config, float]] = []
    for _ in range(n):
        start = Config(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 2 * math.pi))
        goal = Config(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0, 2 * math.pi))
        radius = rng.uniform(0.3, 3.0)
        cases.append((start, goal, radius))
    return cases


_ALL_CASES = _FIXED_CASES + _random_cases(400, seed=20260720)

# The full-pipeline test already sweeps all 400 random cases; the per-word test
# only needs a smaller slice to catch a closed-form regression per word without
# multiplying CI runtime by six (40 random + 6 fixed) x 6 words. A *distinct*
# seed (not a prefix of the 400) gives the per-word level its own geometries
# rather than re-checking the first 40 full-pipeline cases.
_PER_WORD_CASES = _FIXED_CASES + _random_cases(40, seed=20260721)


def _case_id(case: tuple[Config, Config, float]) -> str:
    start, goal, radius = case
    return (
        f"s({start.x:.2f},{start.y:.2f},{start.theta:.2f})"
        f"-g({goal.x:.2f},{goal.y:.2f},{goal.theta:.2f})-r{radius:.2f}"
    )


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id)
def test_shortest_path_matches_easydubins(case: tuple[Config, Config, float]) -> None:
    """Our shortest path agrees with easydubins on word and total length."""
    start, goal, radius = case

    best_solutions = solve_all(start, goal, radius)
    best = shortest(best_solutions)
    assert best is not None, "Dubins paths always exist between finite configs"
    our_len = _length(best_solutions[best])

    ed_mode, ed_lengths, _ = ed.dubins_path(
        (start.x, start.y, start.theta), (goal.x, goal.y, goal.theta), radius
    )
    ed_word = "".join(ed_mode)
    ed_len = sum(abs(seg) for seg in ed_lengths)

    assert our_len == pytest.approx(ed_len, abs=_ABS_TOL)
    # The word can legitimately differ only on an exact length tie; the length
    # assertion above already pins our_len to ed_len, so the word must match
    # unless two distinct words share the (near-)minimum length.
    assert best.value == ed_word or _is_length_tie(best_solutions, ed_len)


@pytest.mark.parametrize("case", _PER_WORD_CASES, ids=_case_id)
@pytest.mark.parametrize("path_type", list(PathType), ids=lambda pt: pt.value)
def test_per_word_matches_easydubins(
    path_type: PathType, case: tuple[Config, Config, float]
) -> None:
    """Each word solver agrees with easydubins on feasibility and segment lengths.

    Both are fed the *same* canonical ``(alpha, beta, d)`` so this isolates the
    closed-form word formulas from the frame transform.
    """
    start, goal, radius = case
    alpha, beta, d = dubins._canonical_frame(start, goal, radius)

    solver = dubins._SOLVERS[path_type][0]
    ours = solver(alpha, beta, d)
    theirs = ed.general_planner(path_type.value, alpha, beta, d)

    # Feasibility must agree: both ``None`` or both a solution. A tolerance-free
    # compare is right in the common case, but a scenario landing within
    # float-noise of an existence boundary (p_sq ~ 0 for LSR/RSL, |tmp| ~ 1 for
    # CCC) is the one spot where the two algebraically-identical impls can round
    # opposite ways. If exactly one side is feasible AND that feasible solution
    # sits on such a boundary -- detected via its middle segment -- the
    # disagreement is benign boundary noise, so skip; a disagreement away from
    # any boundary is a real bug and must still hard-fail. Boundary signatures of
    # the middle segment: LSR/RSL straight collapses to ~0; CCC ``|tmp|=1`` lands
    # the reflex arc ``p = normalize(-acos(tmp))`` at ~0 / ~2*pi (tmp=+1) or ~pi
    # (tmp=-1), the two edges of its feasible ``{0} u [pi, 2*pi)`` range.
    if (ours is None) != (theirs is None):
        feasible = list(ours) if ours is not None else theirs[0]
        middle = feasible[1]
        is_ccc = path_type.value in ("RLR", "LRL")
        on_boundary = abs(middle) < 1e-6 or (
            is_ccc and (abs(middle - math.pi) < 1e-6 or abs(middle - 2 * math.pi) < 1e-6)
        )
        if on_boundary:
            pytest.skip(f"{path_type.value} feasibility on existence boundary (benign noise)")
    assert (ours is None) == (theirs is None), (
        f"{path_type.value} feasibility disagreement: ours={ours!r} theirs={theirs!r}"
    )
    if ours is None:
        return

    ed_path = theirs[0]  # theirs = (path=[t, p, q], mode, cost); [0] is the [t, p, q] list
    for ours_seg, ed_seg, name in zip(ours, ed_path, ("t", "p", "q"), strict=True):
        assert ours_seg == pytest.approx(ed_seg, abs=_ABS_TOL), (
            f"{path_type.value} segment {name} differs"
        )


def _length(solution: object) -> float:
    assert isinstance(solution, DubinsPath)
    return solution.length


def _is_length_tie(solutions: Mapping[PathType, object], ed_len: float) -> bool:
    """True if two distinct words share the (near-)minimum length for this case.

    A word-name mismatch is only acceptable when the total lengths tie, in which
    case either library may pick either winner. The tie window
    ``abs(length - ed_len) < 1e-6`` is intentionally looser than ``_ABS_TOL``
    (1e-9): a genuine length tie can sit anywhere in float noise, so the tie
    test uses a wider window than the exact-match assertion.
    """
    lengths = [sol.length for sol in solutions.values() if isinstance(sol, DubinsPath)]
    near_min = [length for length in lengths if abs(length - ed_len) < 1e-6]
    return len(near_min) >= 2
