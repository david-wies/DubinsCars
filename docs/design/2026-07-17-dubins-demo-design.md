# Dubins Path Demonstrator — Design

**Date:** 2026-07-17
**Status:** Approved
**Spec:** [../specs/2026-07-17-dubins-demo-spec.md](../specs/2026-07-17-dubins-demo-spec.md)

## 1. Architecture Overview

Layered package. The math core is pure and UI-free; a small observable model is the
single source of truth; Tkinter views subscribe to it. No widget talks to another
widget directly.

```
dubins_demo/
├── __main__.py           # entry point: build model, build app, mainloop
├── core/
│   ├── angles.py         # azimuth↔angle, rad↔deg, normalization
│   ├── dubins.py         # pure functions: 6 solvers, DubinsPath, sampling
│   └── model.py          # Scenario state + RadiusPolicy + observer notify
├── ui/
│   ├── app.py            # main window, menus, status bar, wiring
│   ├── input_panel.py    # entry fields, convention/unit toggles, radius sub-frame
│   ├── plot_canvas.py    # FigureCanvasTkAgg, drawing, drag/rotate, animation
│   └── details_panel.py  # sortable ttk.Treeview of paths, selection
├── persistence/
│   └── scenario_io.py    # JSON save/load, CSV waypoint export
│                         # (named "persistence", not "io", to avoid stdlib shadowing)
├── help/
│   ├── index.html        # TOC, search JS, MathML theory sections
│   └── style.css
└── tests/
    ├── test_angles.py
    └── test_dubins.py
```

Data flow:

```
    user input (typing / drag / spinbox / table click)
        │
        ▼
   Scenario model  ──update──▶  notify listeners
        │                            │
        ▼                            ▼
 core.dubins.solve_all()      views re-read model + solutions and redraw
```

## 2. Core Layer

### 2.1 `core/angles.py`

Pure conversion helpers, all operating on floats (radians internally):

```python
def normalize(theta: float) -> float          # → [0, 2π)
def deg_to_rad(v: float) -> float
def rad_to_deg(v: float) -> float
def angle_to_azimuth(theta: float) -> float   # az = π/2 − θ (mod 2π)
def azimuth_to_angle(az: float) -> float      # inverse (same formula, involution)
```

### 2.2 `core/dubins.py`

Pure functions and frozen dataclasses. No Tkinter, no model imports.

```python
class SegmentKind(Enum): L, S, R

class PathType(Enum): LSL, RSR, LSR, RSL, RLR, LRL

@dataclass(frozen=True)
class Config:
    x: float; y: float; theta: float          # meters, radians, math convention

@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    length: float                              # arc length, meters, ≥ 0

@dataclass(frozen=True)
class DubinsPath:
    path_type: PathType
    segments: tuple[Segment, ...]              # exactly 3
    radius: float
    start: Config
    @property
    def length(self) -> float                  # sum of segment lengths
    def sample(self, step: float = 0.05) -> np.ndarray   # (N, 3) of x, y, theta

@dataclass(frozen=True)
class Infeasible:
    path_type: PathType
    reason: str                                # human-readable, shown in table

def solve_all(start: Config, goal: Config, radius: float
              ) -> dict[PathType, DubinsPath | Infeasible]

def shortest(solutions) -> PathType | None
```

**Algorithm** (standard canonical-frame method):

1. Transform to canonical frame: translate start to origin, rotate so the goal
   lies on the +X axis, scale by `1/radius`. The problem reduces to
   `(0, 0, α) → (d, 0, β)` with unit radius.
2. Solve each word with closed-form trigonometric formulas
   (Shkel & Lumelsky 2001). Each solver returns three normalized segment
   parameters `(t, p, q)` or infeasibility:
   - CSC words (LSL, RSR, LSR, RSL): tangent-line construction between the two
     turning circles. LSR/RSL infeasible when the inner tangent does not exist
     (circles overlap: `d² < 4` in canonical units after the relevant offset).
   - CCC words (RLR, LRL): middle-circle construction; infeasible when the
     turning-circle centers are farther than `4·radius` apart.
3. Scale segment lengths back by `radius`; arcs get length `radius · |angle|`.

**Sampling** walks the three segments integrating the unicycle model analytically
(arc = rotation about the segment's circle center; straight = linear), returning
an `(N, 3)` array. The final sample is exactly the goal configuration (used by
tests).

**Turning-circle helper** for the overlay:

```python
def turning_centers(cfg: Config, radius: float) -> tuple[Point, Point]  # (left, right)
```

### 2.3 `core/model.py`

```python
class RadiusPolicy(Protocol):
    def min_radius(self) -> float

@dataclass
class FixedRadius:
    value: float
    def min_radius(self) -> float: return self.value

class Scenario:
    start: Config
    goal: Config
    radius_policy: RadiusPolicy
    heading_convention: Convention      # ANGLE | AZIMUTH   (display only)
    angle_unit: Unit                    # DEG | RAD         (display only)
    selected_type: PathType | None      # None → auto (shortest)
    show_circles: bool
    animation_speed: float              # m/s

    def add_listener(cb: Callable[[], None]) -> None
    def update(**changes) -> None       # set fields, re-solve, notify once
    solutions: dict[PathType, DubinsPath | Infeasible]   # cached, refreshed by update
    highlighted: PathType | None        # selected if feasible else shortest
```

`update()` batches changes → one `solve_all` call → one notification. Views never
compute; they read `scenario.solutions` and `scenario.highlighted`.

Display preferences (`heading_convention`, `angle_unit`) affect only how the
input panel formats/parses values; the stored `Config` is always canonical.

## 3. UI Layer

### 3.1 Layout (`ui/app.py`)

```
┌──────────────────────────────────────────────┐
│ Menu: File(Save/Load/Export CSV) Help(Open)  │
├───────────────┬──────────────────────────────┤
│ Input panel   │  Plot canvas                 │
│               │  (FigureCanvasTkAgg +        │
│ Details panel │   NavigationToolbar2Tk)      │
├───────────────┴──────────────────────────────┤
│ Status bar                                   │
└──────────────────────────────────────────────┘
```

`App` builds the model, the three panels, menus, and the status bar; it owns the
`webbrowser.open(help/index.html)` handler and the file dialogs (delegating I/O to
`persistence/scenario_io.py`). Status-bar messages go through a small `StatusSink` callable
passed to panels (avoids panels importing each other).

### 3.2 Input panel (`ui/input_panel.py`)

- Two labeled groups (Start / Goal), each: X, Y, Heading entries.
- Radio buttons: heading convention (angle/azimuth) and unit (deg/rad). Switching
  re-formats the heading entries from the canonical model value — no cumulative
  round-trip drift.
- Radius **sub-frame** (swappable per EXT-2): spinbox starting at `0.1`, bound
  to `FixedRadius.value`.
- Entry validation on `<FocusOut>`/`<Return>`: parse → convert to canonical →
  `model.update()`. Parse failure: field turns red, status message, model untouched.
- Panel listens to the model and rewrites entry text when the change originated
  elsewhere (drag/nudge), skipping the widget that has focus mid-edit.

### 3.3 Plot canvas (`ui/plot_canvas.py`)

Rendering (on every model notification):

- All feasible paths as `Line2D` from `path.sample()`; fixed color per type
  (colorblind-safe tab10 subset), highlighted path `linewidth=3, alpha=1`, others
  `linewidth=1.5, alpha=0.35`. Legend outside the axes.
- Start/goal drawn as `FancyArrow` (length ∝ radius, capped); start green,
  goal red.
- Optional turning circles (dashed, matching side L/R annotated).
- Equal aspect; auto-fit bounding box of configs + paths with 10% margin unless
  the user has zoomed (toolbar state checked; a config leaving the view resets
  auto-fit).

Interaction (matplotlib event handlers `button_press`, `motion_notify`,
`button_release`):

- Hit test in **display (pixel) space**: within 12 px of an arrow base → *move*
  mode; within 12 px of arrow head → *rotate* mode (heading = atan2 from base to
  cursor). Radius of grab zones scales with nothing — pixel space keeps feel
  constant across zoom.
- During drag: `model.update()` per motion event (full re-solve is O(1), NFR-2);
  redraw via `canvas.draw_idle()`.
- Click selecting an arrow also sets it as the keyboard-nudge target
  (visual cue: thin outline). Arrow keys ±0.1 m, Shift+arrows ±1°, bound on the
  canvas widget.

Animation:

- ▶/⏸ button + speed entry (m/s) below the toolbar.
- `matplotlib.animation.FuncAnimation` moving a marker (small triangle oriented
  by sampled `theta`) along `highlighted.sample()`; frame interval derived from
  speed and sampling step. Stops and resets on any model change.

### 3.4 Details panel (`ui/details_panel.py`)

- `ttk.Treeview`, columns: Type | Length [m] | Segments (e.g. `L 2.31 → S 5.10 → L 0.87`).
- Feasible rows sorted by clicking headers (default: length ascending). Infeasible
  rows always at the bottom, grayed, reason in the Segments column.
- Row click → `model.update(selected_type=…)`. Highlighted row kept visually in
  sync when selection falls back to shortest.

## 4. I/O (`persistence/scenario_io.py`)

- **JSON schema** (EXT-4):

```json
{
  "version": 1,
  "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
  "goal":  {"x": 10.0, "y": 5.0, "theta": 1.57},
  "radius_policy": {"type": "fixed", "value": 2.0},
  "display": {"heading_convention": "angle", "angle_unit": "deg"}
}
```

  `theta` stored canonically (radians, math convention). Unknown `radius_policy.type`
  on load → error dialog, scenario unchanged.

- **CSV export**: header `x,y,theta_rad`, rows from `highlighted.sample(step)`
  (step asked via simple dialog, default 0.05 m). Disabled when nothing feasible.

## 5. Help Page (`help/index.html`)

Self-contained (inline CSS/JS, no CDN). Structure:

- Fixed sidebar: nested TOC (links to section anchors) + search box.
- Search: JS walks section text nodes, highlights matches (`<mark>`), shows a
  result count, Enter/buttons jump between matches.
- Sections:
  1. **Using the app** — inputs, conventions, dragging, keyboard shortcuts,
     radius spinbox, details table, animation, save/load/export.
  2. **The Dubins problem** — statement, assumptions (constant speed, bounded
     curvature, forward-only), Dubins' theorem: optimal path ∈ {CSC, CCC}.
  3. **The six words** — geometry of each type with MathML derivations in the
     canonical frame: tangent constructions for LSL/RSR (outer tangents),
     LSR/RSL (inner tangents, existence condition), RLR/LRL (third-circle
     construction, `d < 4r` condition), and the segment-length formulas
     `(t, p, q)` per Shkel & Lumelsky.
  4. **Worked example** — one scenario computed step by step.
  5. **References** — Dubins (1957), *Amer. J. Math* 79(3):497–516,
     DOI 10.2307/2372560; Shkel & Lumelsky (2001), *IJRA* 34(2–3):179–202,
     DOI 10.1016/S0921-8890(00)00127-5. Links to publisher/DOI pages.

## 6. Error Handling Summary

| Case | Behavior |
| --- | --- |
| Non-numeric entry | Field red, status message, model unchanged (FR-23) |
| Radius out of range | Minimum clamped to 0.1 |
| CCC too far / inner tangent missing | `Infeasible(reason)`, grayed table row |
| start == goal | All types infeasible-or-zero-length handled; status note |
| JSON load: bad schema/policy | Error dialog, model unchanged |
| Export with no feasible path | Menu item disabled |

## 7. Testing

### 7.1 Automated (pytest, `tests/`)

- `test_angles.py`: conversion round-trips, normalization, involution of
  angle↔azimuth.
- `test_dubins.py`:
  - Parametrized known cases per type (hand-computed + published examples).
  - Endpoint property: `sample()[-1] ≈ goal` (pos + heading, 1e-6) for every
    feasible solution over a randomized grid of scenarios.
  - `length == sum(segment lengths)`; sample continuity (max step bound).
  - Rigid-transform invariance: translate/rotate scenario → same lengths;
    mirror → L/R-swapped types with same lengths.
  - CCC existence boundary: center distance just below / at / above `4r`.
  - Shortest-path selection consistency.

### 7.2 Manual UI checklist

1. Type coordinates → plot updates; drag arrow → fields update.
2. Toggle deg/rad and angle/azimuth repeatedly → values convert, no drift.
3. Slider vs radius entry stay in sync; paths update live during slide.
4. Table sort, row click highlights; infeasible rows grayed with reason.
5. Circles toggle; animation plays, pauses, resets on edit; speed change works.
6. Save → load round-trip restores scenario exactly; CSV opens in spreadsheet.
7. Help opens in browser; TOC links and search work offline.
8. Degenerate: drag goal onto start — no crash, sensible table.

## 8. Dependencies

- Runtime: `numpy`, `matplotlib` (Tk backend). Python ≥ 3.10.
- Dev: `pytest`.
- Packaging: plain `pyproject.toml`; run with `python -m dubins_demo`.
