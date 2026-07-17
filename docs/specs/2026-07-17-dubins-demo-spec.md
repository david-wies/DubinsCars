# Dubins Path Demonstrator — Specification

**Date:** 2026-07-17
**Status:** Approved

## 1. Purpose

A desktop application (Python + Tkinter + matplotlib) that calculates and demonstrates
Dubins paths: the shortest curvature-constrained paths between two oriented
configurations in the plane. The app is both a practical calculator and a teaching
tool — it shows *all* candidate path types, their construction, and the math behind
them.

## 2. Definitions and Conventions

- **Configuration**: a triple `(x, y, heading)` — position in meters plus heading.
- **Heading conventions** (user-selectable, converted automatically on switch):
  - **Angle** (math convention): 0 = East (+X axis), counter-clockwise positive.
  - **Azimuth** (compass convention): 0 = North (+Y axis), clockwise positive.
  - Conversion: `azimuth = 90° − angle (mod 360°)`.
- **Angle units** (user-selectable, converted automatically on switch): degrees or radians.
- **Internal representation** is always radians, math-angle convention, meters.
  Conversions happen only at the UI boundary.
- **Path types**: LSL, RSR, LSR, RSL (CSC family), RLR, LRL (CCC family), where
  L = left arc, R = right arc, S = straight segment, all arcs at the minimum turn
  radius.

## 3. Functional Requirements

### 3.1 Input

- **FR-1** User can type start and goal configurations: X, Y, heading — in entry fields.
- **FR-2** User can toggle heading convention (angle/azimuth) and units (deg/rad);
  toggling converts the displayed field values in place without changing the
  underlying configuration.
- **FR-3** User can set the turn radius via a slider (range 0.1–50 m) with a linked
  entry field for exact values. Values outside the range are clamped.
- **FR-4** User can drag the start/goal arrow **base** on the plot to move the
  configuration, and drag the arrow **head** to rotate its heading.
- **FR-5** Keyboard nudging: arrow keys move the selected configuration by ±0.1 m;
  Shift+arrows rotate the heading by ±1°.
- **FR-6** Typed input, dragging, and the slider all stay in sync (two-way binding
  through a single model).

### 3.2 Computation

- **FR-7** The app computes **all six** Dubins path types for the current scenario.
- **FR-8** Types that do not exist for the given scenario (e.g., CCC types when the
  configurations are too far apart) are reported as infeasible with a reason.
- **FR-9** For each feasible path the app computes: total length, and per-segment
  breakdown (segment kind L/S/R and its length).
- **FR-10** Paths can be sampled into `(x, y, heading)` waypoint sequences at a
  configurable step for plotting, animation, and export.

### 3.3 Display

- **FR-11** All feasible paths are drawn simultaneously on the matplotlib canvas,
  each path type with a fixed distinctive color; a legend maps colors to types.
- **FR-12** One path is **highlighted** (bold); the others are dimmed. Default
  highlight is the shortest feasible path. The user can highlight any path by
  clicking its row in the details table. After a re-solve the user's choice is
  kept if still feasible, otherwise selection falls back to the shortest.
- **FR-13** The details panel shows a table: path type, total length, segment
  breakdown. Rows are sortable by clicking column headers. Infeasible types appear
  grayed out with the infeasibility reason.
- **FR-14** Optional overlay (toggle): the four turning circles (left/right at
  start and goal).
- **FR-15** Animation: a play button animates a marker traveling along the
  highlighted path at a user-set speed in m/s.
- **FR-16** The plot keeps an equal aspect ratio, auto-fits the scenario with a
  margin, and includes the standard matplotlib navigation toolbar (pan/zoom/save
  PNG). Manual zoom is preserved until the configuration moves out of view.
- **FR-17** A status bar shows mouse coordinates, warnings, and infeasibility notes.

### 3.4 File I/O

- **FR-18** Save/load scenario as JSON (start, goal, radius policy, display prefs).
- **FR-19** Export the highlighted path's sampled waypoints `(x, y, heading)` as CSV.
- **FR-20** Export the figure as PNG (via the matplotlib toolbar).

### 3.5 Help

- **FR-21** Help menu opens a local HTML page in the system browser (`webbrowser`
  module). The page is fully offline and self-contained.
- **FR-22** The help page includes: a sidebar table of contents; a JavaScript text
  search with match highlighting and jump-to-result; a usage guide (inputs,
  dragging, table, keyboard shortcuts, file I/O); Dubins theory sections — problem
  statement, the six-word construction (CSC/CCC families), per-type derivations
  with formulas rendered in **MathML**; a worked example; and links to the original
  articles (Dubins 1957; Shkel & Lumelsky 2001).

### 3.6 Error Handling

- **FR-23** Non-numeric entry input turns the field red, shows a status-bar
  message, and leaves the model unchanged.
- **FR-24** Degenerate scenarios (start equals goal, goal inside the start turning
  circle, etc.) never crash: affected path types are marked infeasible with a
  reason; if no path exists a status warning is shown.
- **FR-25** All angles are normalized to `[0, 2π)` internally.

## 4. Non-Functional Requirements

- **NFR-1** Python ≥ 3.10; runtime dependencies limited to `numpy` and `matplotlib`
  (Tkinter from the standard library). `pytest` is a dev-only dependency.
- **NFR-2** Recomputing all six solvers is cheap enough to run live during drag
  (target: full re-solve + redraw under 50 ms on typical hardware).
- **NFR-3** The Dubins math layer is pure (no UI imports) and unit-tested.
- **NFR-4** The help page works offline in any modern browser (MathML: Chrome 109+,
  Firefox, Safari).

## 5. Extensibility (designed-for, not implemented)

- **EXT-1** Minimum turn radius derived from speed: the model obtains the radius
  through a `RadiusPolicy` interface (`min_radius() -> float`). The initial
  implementation is `FixedRadius(value)` driven by the slider. A future
  `SpeedBasedRadius` can implement e.g. car `r = v²/(μg)`, aircraft
  `r = v²/(g·tan φ_bank)`, or rate-limited `r = v/ω_max` — without touching the
  solvers or views.
- **EXT-2** The radius section of the input panel is a swappable sub-frame, so it
  can later be replaced by speed + vehicle-parameter inputs with a read-only
  derived radius display.
- **EXT-3** Animation speed is specified in m/s so a future speed model reuses it.
- **EXT-4** The scenario JSON stores the radius as
  `"radius_policy": {"type": "fixed", "value": …}` for forward compatibility.

## 6. Testing

- **T-1** pytest suite for the math layer; parametrized known cases for each of the
  six types (hand-computed and published examples).
- **T-2** Property tests: the sampled path endpoint matches the goal configuration
  (position and heading, tolerance 1e-6); total length equals the sum of segment
  lengths; sampled curves are continuous; solutions are invariant under rigid
  transformations (translate/rotate/mirror symmetry).
- **T-3** CCC existence boundary cases (center distance ≈ 4r).
- **T-4** Angle/azimuth and deg/rad conversion round-trips.
- **T-5** UI is not auto-tested; a manual test checklist is included in the design
  document.

## 7. Out of Scope

- Obstacles / collision avoidance.
- Variable curvature (clothoids), reverse gear (Reeds-Shepp paths).
- 3D / altitude.
- Multiple vehicles or waypoint sequences (single start → single goal only).
