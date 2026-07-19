# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-19

Initial public release.

### Added

- Dubins path computation for all six path words (LSL, RSR, LSR, RSL, RLR, LRL),
  finding the shortest curvature-constrained path between two oriented planar
  configurations.
- Tkinter desktop UI with draggable start/goal arrows (base to move, head to
  rotate) and a turn-radius control.
- Details table listing every feasible path type with its length.
- JSON scenario save/load.
- CSV export of the highlighted path's waypoints.
- Offline help guide (usage plus Dubins theory) opened in the browser.
