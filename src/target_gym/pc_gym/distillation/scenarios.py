"""The distillation column's scenario suite (charter Amendment 3, rule 20).

Each scenario is an operating window the family's sources show, with timing and
sign drawn by the seed (``DistillationParams.step_window``). Sources are the
tags of this folder's ``REALITY.md``.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""


def _row(name, overrides, **kw):
    """One scenario as a plain row: the package imports nothing outside its
    fingerprint, and the registry builds the ``Scenario``."""
    return dict(name=name, overrides=overrides, **kw)


#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1), from REALITY.md
#: ("How often each window occurs", second search 2026-11-11). A normal day lasts a
#: day: a large process plant meets an abnormal situation almost daily [ST1]. The
#: rest is ours, set once without tuning and open for the owner (a crude unit
#: switches crude every 2 to 5 days [ST4], outside this splitter family): an upset
#: regime lasts a shift, 8 h (shifts run 8 to 12 h [ST2, ST3]), and brings one event
#: per visit on average, which steps every feed its
#: scenario moves. A feed step holds for a shift (8 h, 480 min), until the upstream
#: unit's rate or crude is restored, then the feed returns to nominal.
_DAY_H, _SHIFT_H = 24.0, 8.0
#: A stream starts none of its window's steps: only the stream starts them.
_NO_WINDOW_STEP = {"step_window": (10**9, 10**9 + 1)}
_UPSET = dict(
    dwell_h=_SHIFT_H,
    events_per_h=1.0 / _SHIFT_H,
    stream_overrides={**_NO_WINDOW_STEP, "event_hold": _SHIFT_H * 60.0},
)


def distillation_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            {"feed_step": 0.0, "zF_step": 0.0},
            source="[D4, p. 5]",
            description="the design point held in its zones at least boilup",
            weight=0.7,
            dwell_h=_DAY_H,
            stream_overrides=_NO_WINDOW_STEP,
        ),
        _row(
            "feed_rate_step",
            {"feed_step": 0.2, "zF_step": 0.0},
            source="[D1, p. 33, Fig. 18]",
            description="the feed rate steps 20 % up or down, metered",
            weight=0.1,
            **_UPSET,
        ),
        _row(
            "feed_composition_step",
            {"feed_step": 0.0, "zF_step": 0.1},
            source="[D1, p. 33, Fig. 18]",
            description="the feed composition steps 0.1, unmeasured",
            weight=0.1,
            **_UPSET,
        ),
        _row(
            "both_steps",
            {"feed_step": 0.2, "zF_step": 0.1},
            source="[D1, p. 33, Fig. 18]",
            description="both steps, at independent times, as Fig. 18 runs them",
            weight=0.1,
            **_UPSET,
        ),
    )
