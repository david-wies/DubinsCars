# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Desktop app (Python + Tkinter + matplotlib + numpy) that computes and demonstrates **Dubins paths** — shortest curvature-constrained paths between two oriented planar configurations. Shows all six path words (LSL, RSR, LSR, RSL, RLR, LRL). Package name: `dubins_demo`.

## Commands

```bash
pip install -e ".[dev]"       # dev setup: numpy, matplotlib, pytest, ruff, pyright, pre-commit
python -m dubins_demo          # launch the app (needs Tkinter: apt install python3-tk on Debian/Ubuntu)
pytest                         # math-core tests only (dubins_demo/tests)
pytest dubins_demo/tests/test_dubins.py::test_name   # single test
pre-commit run --all-files     # full lint/format/type gate (ruff + pyright + taplo + yamllint + markdownlint + stylelint + htmlhint)
```

CI (`.github/workflows/ci.yml`) runs the **exact same** `pre-commit` hooks plus `pytest` (on py3.10/3.11/3.12). Local and CI never drift — change a hook once in `.pre-commit-config.yaml`. CI and the remote default branch are both `master` (a local-only `main` branch also exists).

## Architecture

Strict layering, dependencies point inward only:

- **`core/`** — pure math, UI-free (no Tkinter/matplotlib/model imports).
  - `angles.py` — angle/azimuth/degree conversions, `normalize()` to `[0, 2π)`.
  - `dubins.py` — the six word solvers, `Config`, `DubinsPath`, `solve_all()`, `shortest()`, `sample()`, `turning_centers()`. Uses the Shkel & Lumelsky (2001) canonical-frame method: transform to `(0,0,α) → (d,0,β)` at unit radius, solve each word closed-form, scale back.
  - `model.py` — `Scenario` (observable single source of truth), `RadiusPolicy`/`FixedRadius`, `Convention`/`Unit` display enums.
- **`ui/`** — Tkinter views (`app.py`, `input_panel.py`, `details_panel.py`, `plot_canvas.py`). Panels never import one another; they share the `Scenario` model + a `StatusSink` callback. Tk root is created only in `App.__init__`, never at import time (headless-safe imports).
- **`persistence/`** — `scenario_io.py`: JSON save/load (`LoadedScenario`), CSV waypoint export, `ScenarioError`. Folder is named `persistence` (not `io`) deliberately to avoid shadowing stdlib `io`. Schema is `SCHEMA_VERSION = 1`; any load error (bad JSON, missing key, wrong version) surfaces as a single `ScenarioError`.
- **`help/`** — offline `index.html` + `style.css`, opened via `webbrowser.open()`; shipped as package-data, no CDN.

Data flow: views subscribe with `Scenario.add_listener()` and re-read the model on notification — they never run solvers. All solving happens in `Scenario.update(**changes)`, which batches field changes into **one** `solve_all()` and **one** notification. Prefer a single `update()` with multiple fields over several calls.

## Domain invariants

- **Canonical units are meters + radians in the math convention** (0 = +X/East, counter-clockwise positive). Degrees and compass azimuth exist only at the UI boundary; `core` functions expect canonical `Config` values.
- Keep `core/` (especially `dubins.py`, `angles.py`) free of UI/model/persistence imports.
- Azimuth conversion is `π/2 − θ` (its own inverse); see `angles.py`.
- When changing serialization, update `persistence/scenario_io.py` and bump `SCHEMA_VERSION` per the design doc.

## Testing note

Only the pure math core is auto-tested. The Tkinter UI is **intentionally not** in the automated suite (spec T-5) — it's validated against the manual checklist in the design doc (section 7.2). Add numeric-tolerance/invariant tests alongside `tests/test_dubins.py` when touching solvers.

## Reference docs

- `docs/specs/2026-07-17-dubins-demo-spec.md` — requirements + testing plan.
- `docs/design/2026-07-17-dubins-demo-design.md` — architecture, algorithms, domain invariants, manual UI checklist (section 7.2).
