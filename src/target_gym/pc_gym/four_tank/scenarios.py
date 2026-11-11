"""The four-tank rig's scenario suite: its one window, as the benchmark has always
scored it (one scenario, weight 1).

It is declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1). A window runs a schedule of three setpoint moves, an extra outlet and
a cup, all ours in size and timing (PHYSICS.md, section 2); a continuing plant
starts none of it and draws the same events at the window's own density instead:
five in its 3300 s, about 5.45 an hour, in its mix (``event_shares``), each
setpoint move held 1000 s, the window's spacing between moves, before it returns.
No source gives how often the rig is upset, so these numbers are ours, set once
without tuning and open for the owner.
"""

_WINDOW_S = 3300.0
_EVENTS_PER_WINDOW = 5.0
_MOVE_HOLD_S = 1000.0


def four_tank_scenarios() -> tuple[dict, ...]:
    return (
        dict(
            name="normal",
            overrides={},
            source="PHYSICS.md, section 5",
            description="setpoint steps from the operating point, an extra outlet "
            "and a cup",
            weight=1.0,
            events_per_h=_EVENTS_PER_WINDOW * 3600.0 / _WINDOW_S,
            stream_overrides={"window_events": 0.0, "event_hold": _MOVE_HOLD_S},
        ),
    )
