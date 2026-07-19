"""Dubins Path Demonstrator.

Desktop app (Tkinter + matplotlib + numpy) that computes and demonstrates
Dubins paths. See docs/specs and docs/design for the full specification.

The package is organized in strict inward-pointing layers: ``core`` (pure
math — angle conversions and the six Dubins word solvers), ``persistence``
(JSON/CSV scenario I/O), ``ui`` (Tkinter views), and ``help`` (offline docs).
See the design doc's package layout
(docs/design/2026-07-17-dubins-demo-design.md, section 1).
"""

__version__ = "0.1.0"
