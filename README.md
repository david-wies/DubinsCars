# Dubins Path Demonstrator

A desktop application (Python + Tkinter + matplotlib + numpy) that computes and
demonstrates **Dubins paths** — the shortest curvature-constrained paths between
two oriented configurations in the plane. It's both a practical calculator and a
teaching tool: it shows *all* candidate path types (LSL, RSR, LSR, RSL, RLR, LRL),
their construction, and the math behind them.

Status: scaffolding only. The math core, UI, and persistence layers land in
subsequent work — see the docs below for the full plan.

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

Once the app exists, it will run via:

```bash
python -m dubins_demo
```
