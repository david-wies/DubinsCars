"""Tests for :mod:`dubins_demo.core.angles`."""

from __future__ import annotations

import math

import pytest

from dubins_demo.core.angles import (
    angle_to_azimuth,
    azimuth_to_angle,
    deg_to_rad,
    normalize,
    rad_to_deg,
)

_TAU = 2.0 * math.pi


@pytest.mark.parametrize(
    ("theta", "expected"),
    [
        (0.0, 0.0),
        (_TAU, 0.0),
        (-0.0, 0.0),
        (math.pi, math.pi),
        (_TAU + 1.0, 1.0),
        (-1.0, _TAU - 1.0),
        (-_TAU, 0.0),
    ],
)
def test_normalize_known_values(theta: float, expected: float) -> None:
    assert normalize(theta) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("k", range(-5, 6))
def test_normalize_in_half_open_range(k: int) -> None:
    for base in (0.0, 0.3, math.pi, 3.5, 6.0):
        result = normalize(base + k * _TAU)
        assert 0.0 <= result < _TAU


def test_normalize_tiny_negative_stays_below_tau() -> None:
    # Guards the float-modulo edge case where -1e-17 % TAU rounds up to TAU.
    assert normalize(-1e-17) < _TAU


@pytest.mark.parametrize("deg", [0.0, 1.0, 45.0, 90.0, -30.0, 360.0, 123.456])
def test_deg_rad_round_trip(deg: float) -> None:
    assert rad_to_deg(deg_to_rad(deg)) == pytest.approx(deg, abs=1e-12)


def test_deg_to_rad_known() -> None:
    assert deg_to_rad(180.0) == pytest.approx(math.pi)
    assert rad_to_deg(math.pi) == pytest.approx(180.0)


def test_angle_azimuth_known_values() -> None:
    # Math angle 0 (East) -> azimuth pi/2 (East is 90 deg clockwise from North).
    assert angle_to_azimuth(0.0) == pytest.approx(math.pi / 2)
    # Math angle pi/2 (North) -> azimuth 0.
    assert angle_to_azimuth(math.pi / 2) == pytest.approx(0.0)


@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0, math.pi, 4.0, 5.9, _TAU - 0.01])
def test_angle_azimuth_involution(theta: float) -> None:
    assert azimuth_to_angle(angle_to_azimuth(theta)) == pytest.approx(normalize(theta), abs=1e-12)
    assert angle_to_azimuth(azimuth_to_angle(theta)) == pytest.approx(normalize(theta), abs=1e-12)


@pytest.mark.parametrize("theta", [-3.0, -0.2, 7.0, 100.0])
def test_angle_azimuth_output_normalized(theta: float) -> None:
    assert 0.0 <= angle_to_azimuth(theta) < _TAU
    assert 0.0 <= azimuth_to_angle(theta) < _TAU
