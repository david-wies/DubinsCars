# Dubins Path Demonstrator

A desktop application (Python + Tkinter + matplotlib + numpy) that computes and
demonstrates **Dubins paths** — the shortest curvature-constrained paths between
two oriented configurations in the plane. It's both a practical calculator and a
teaching tool: it shows *all* candidate path types (LSL, RSR, LSR, RSL, RLR, LRL),
their construction, and the math behind them.

## Running the app

Requires Python >= 3.10 with Tkinter (bundled with the standard CPython
installers on Windows and macOS; on Debian/Ubuntu install `python3-tk`).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e .          # installs numpy + matplotlib
python -m dubins_demo     # launches the desktop app
```

In the app: type start/goal configurations or drag the arrows on the plot
(base to move, head to rotate), set the turn radius with the slider, and read
every feasible path type in the details table. **File** saves/loads scenarios
as JSON and exports the highlighted path's waypoints as CSV; **Help** opens the
offline guide (usage + Dubins theory) in your browser.

## Documentation

- [Specification](docs/specs/2026-07-17-dubins-demo-spec.md) — functional and
  non-functional requirements, testing plan.
- [Design](docs/design/2026-07-17-dubins-demo-design.md) — architecture, package
  layout, algorithms, UI wiring, error handling.

## Development

Requires Python >= 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite (pure math-core only; the Tkinter UI is validated manually,
see the design doc's manual checklist):

```bash
pytest
```

Lint and format (Python via ruff; Markdown/YAML/TOML/JSON/HTML/CSS via the
hooks below):

```bash
pre-commit install          # optional: run automatically on every commit
pre-commit run --all-files  # run everything once, e.g. before pushing
```

CI (`.github/workflows/ci.yml`) runs the exact same `pre-commit` hooks plus
`pytest` on every push and pull request against `master`.

Install the dev extras (`pip install -e ".[dev]"`) to get `pytest`, `ruff`,
and `pre-commit`.
