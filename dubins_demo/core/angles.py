"""Pure angle helpers.

All angles are handled internally in radians using the math convention
(0 = +X axis / East, counter-clockwise positive). Conversions to and from the
compass ``azimuth`` convention and to degrees happen only at the UI boundary.
"""

from __future__ import annotations

import math

_TAU = 2.0 * math.pi


def normalize(theta: float) -> float:
    """Map an angle in radians to the half-open interval ``[0, 2*pi)``.

    Floating-point modulo can occasionally return the divisor itself for tiny
    negative inputs (e.g. ``-1e-17 % (2*pi)`` rounds up to ``2*pi``); the guard
    below keeps the result strictly below ``2*pi``.
    """
    result = theta % _TAU
    if result >= _TAU:
        result -= _TAU
    return result


def deg_to_rad(v: float) -> float:
    """Convert degrees to radians."""
    return math.radians(v)


def rad_to_deg(v: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(v)


def angle_to_azimuth(theta: float) -> float:
    """Convert a math-convention angle to a compass azimuth, in ``[0, 2*pi)``.

    Azimuth measures clockwise from North (+Y): ``az = pi/2 - theta``.
    """
    return normalize(math.pi / 2.0 - theta)


def azimuth_to_angle(az: float) -> float:
    """Convert a compass azimuth back to a math-convention angle.

    Uses the same ``pi/2 - x`` formula as :func:`angle_to_azimuth`; the mapping
    is its own inverse (an involution modulo ``2*pi``).
    """
    return normalize(math.pi / 2.0 - az)
