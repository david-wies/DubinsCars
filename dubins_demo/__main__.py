"""Entry point: ``python -m dubins_demo`` launches the desktop application.

Builds a :class:`Scenario` with sensible defaults, constructs the Tk
application (which creates the root window), and enters the main loop.
"""

from __future__ import annotations

import math

from dubins_demo.core.dubins import Config
from dubins_demo.core.model import FixedRadius, Scenario
from dubins_demo.ui.app import App


def main() -> None:
    """Construct the default scenario and run the application."""
    model = Scenario(
        start=Config(0.0, 0.0, 0.0),
        goal=Config(10.0, 5.0, math.pi / 2.0),
        radius_policy=FixedRadius(2.0),
    )
    App(model).run()


if __name__ == "__main__":
    main()
