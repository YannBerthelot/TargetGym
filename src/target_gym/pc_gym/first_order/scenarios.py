"""The first-order lag's scenario suite: its one window, as the benchmark has
always scored it (one scenario, weight 1).

It is declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1). A window starts off target and draws the target once; a continuing
plant starts on its target (``start_on_target``) and changes the target as events,
one per 7.5 s window on average (480 an hour), each a fresh draw over
``target_x_range``. This abstract plant has no plant hours or record, so the rate
is ours, set once without tuning and open for the owner.
"""

_WINDOW_S = 7.5


def first_order_scenarios() -> tuple[dict, ...]:
    return (
        dict(
            name="normal",
            overrides={},
            source="PHYSICS.md, section 4",
            description="a step to a target, held",
            weight=1.0,
            events_per_h=3600.0 / _WINDOW_S,
            stream_overrides={"start_on_target": 1.0},
        ),
    )
