"""The reference ladder's row for ``cstr`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and its structure's source say, and nothing read from a
score (PHYSICS.md section 10). Version 8 is run for what it makes: the reactor
temperature is held by the fresh cooling water, and the water valve is held short of
its limit by the feed, valve position control [S13].
"""

from __future__ import annotations

import math

from target_gym.reference.row import (
    Cv,
    Limit,
    Row,
    Valve,
)
from target_gym.registry import REGISTRY


def row() -> Row:
    spec = REGISTRY["cstr"]
    p = spec.make_test_params()

    def hold(o, xp):
        """The raw commands holding both valves where the observation shows them
        (obs 2 and 3, read exactly): the reset puts the reactor at rest there."""
        return xp.stack([o[2] / 50.0 - 1.0, o[3] / 50.0 - 1.0])

    dt = float(p.delta_t)
    # The reactor: a cooling step first reaches 95 % of its final value in 0.99
    # residence times open loop, 33 min of the design's 34 min residence; the window
    # is twice that. The response then overshoots and settles later (PHYSICS.md, T8).
    residence = float(p.V) / float(p.q)
    window = math.ceil(2 * 0.99 * residence / dt)
    # The temperature on the water, read as its gap to the setpoint (obs 0 less obs
    # 4), so the band the product is on spec in is a fixed zone while the setpoint
    # moves. Where it rests in the band is a placement (charter J1, zones): the
    # score charges ``off_spec`` for leaving it, and a hotter reactor converts more,
    # so the cheap edge is the top.
    band = float(p.T_band)
    T = Cv(
        "T on water",
        0,
        lambda o, xp: o[0] - o[4],
        lambda o, xp: xp.zeros((), xp.float32),
        "T",
        0.1,
        window,
        zone=(-band, band, "high"),
        exit_terms=("off_spec",),
    )
    # The water valve by the feed, held where [S13]'s example holds it, 95 % of its
    # span (0.9 raw) [S13, p. 438, Fig. 7]. It answers the feed through the closed
    # temperature loop over about an hour, so its test runs three hours.
    valve = Valve("valve by feed", lambda o, xp: o[2] / 50.0 - 1.0, 0.9, 1, 0.1, 720)
    # Shinskey's high-temperature override on the feed [S13, p. 436]: a low select
    # lets the temperature cut the feed past the band's top, backed off the probe's
    # resolution (ours, the ceiling's margin). Read on the feed with the water held,
    # the case it is for, over the temperature loop's window (owner, 2026-11-25).
    hot = Limit(
        "T high on feed",
        lambda o, xp: o[0] - o[4],
        band - float(p.e_floor),
        False,
        1,
        "T",
        0.1,
        window,
    )
    # Bump tests from a reset with both valves in mid-range (the water at 80 % of its
    # span), at the middle and at both ends of the feed's temperature range [S1,
    # Tables 3, 4]. The catalyst-lost and richer-feed corners have no rest inside the
    # valves' ranges at that reset.
    lo, hi = (float(x) for x in p.feed_T_range)
    measured = (
        "tried under the valve position loop on 2026-11-23 (commit 4acda1c): it "
        "leaves the band at every set point on the normal day"
    )
    return Row(
        240,
        hold,
        (T,),
        test={"valve_rest": 0.8},
        points=({"feed_T_range": (lo, lo)}, {"feed_T_range": (hi, hi)}),
        limits=(hot,),
        valves=(valve,),
        # The reset stands on the steady state of the day's feed temperature, which
        # is hidden: every rung takes over from the valves as the reset left them
        # (charter B5: a scenario starts in established operation).
        bumpless=True,
        why={
            "feedforward": measured,
            "cascade": measured,
            "decoupler": "one loop and one valve position loop",
        },
    )
