"""The cement kiln's scenario suite: its one window, the normal kiln, as the
benchmark has always scored it (one scenario, weight 1).

It is declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1): a window draws the free-lime target once, a continuing plant changes
it. No source in this folder gives how often a kiln's free-lime target is revised,
so the rate is ours, set once without tuning and open for the owner: once a shift
(8 h) on average, each change a fresh draw over ``target_lime_range``.
"""

_SHIFT_H = 8.0


def cement_kiln_scenarios() -> tuple[dict, ...]:
    return (
        dict(
            name="normal",
            overrides={},
            source="PHYSICS.md, section 4",
            description="the kiln held on its free-lime target as the raw meal "
            "swings",
            weight=1.0,
            events_per_h=1.0 / _SHIFT_H,
        ),
    )
