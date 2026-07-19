# Dubins Path Demonstrator

[![CI](https://github.com/david-wies/DubinsCars/actions/workflows/ci.yml/badge.svg)](https://github.com/david-wies/DubinsCars/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

A desktop application (Python + Tkinter + matplotlib + numpy) that computes and
demonstrates **Dubins paths** — the shortest curvature-constrained paths between
two oriented configurations in the plane. It's both a practical calculator and a
teaching tool: it shows *all* candidate path types (LSL, RSR, LSR, RSL, RLR, LRL),
their construction, and the math behind them.

## Quick start

This is an application, not a library — clone it and run. The launcher script
creates a local virtualenv, installs the app into it, and starts the GUI (first
run only takes a moment; later runs launch instantly):

```bash
git clone https://github.com/david-wies/DubinsCars.git
cd DubinsCars
./run.sh          # Linux/macOS
```

On Windows (PowerShell), run `./run.ps1` instead.

Requires Python >= 3.12 with Tkinter (bundled with the standard CPython
installers on Windows and macOS; on Debian/Ubuntu install it with
`sudo apt install python3-tk`).

## Running the app manually

If you prefer to manage the environment yourself:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -e .          # installs numpy + matplotlib + sv-ttk
dubins-demo               # launches the desktop app
# equivalently: python -m dubins_demo
```

In the app: type start/goal configurations or drag the arrows on the plot
(base to move, head to rotate), set the turn radius with the spinbox, and read
every feasible path type in the details table. **File** saves/loads scenarios
as JSON and exports the highlighted path's waypoints as CSV; **Help** opens the
offline guide (usage + Dubins theory) in your browser.

## Documentation

- [Specification](docs/specs/2026-07-17-dubins-demo-spec.md) — functional and
  non-functional requirements, testing plan.
- [Design](docs/design/2026-07-17-dubins-demo-design.md) — architecture, package
  layout, algorithms, UI wiring, error handling.

## Development

Requires Python >= 3.12.

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
`pytest` on every push and pull request against `main`.

Install the dev extras (`pip install -e ".[dev]"`) to get `pytest`, `ruff`,
and `pre-commit`.

## License

This project is licensed under the [MIT License](LICENSE).
