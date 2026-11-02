"""Guidance laws for the SpaceForge MVP.

Two deterministic reference controllers:

- ``glideslope_action``  : velocity-matching guidance that cancels the natural CW
                          drift and closes on the target on an exponential
                          "glideslope". Docks, repeatedly, within the episode budget.
- ``hot_approach_action``: deliberately reckless full-thrust slam. Used as the
                          adversarial case for the runtime-assurance interlock.

Both are pure functions of the state, so they can be replayed frame by frame for
visualisation without any training.
"""

from spaceforge_mvp.control.glideslope import (
    glideslope_action,
    glideslope_controller,
    hot_approach_action,
)

__all__ = ["glideslope_action", "glideslope_controller", "hot_approach_action"]
