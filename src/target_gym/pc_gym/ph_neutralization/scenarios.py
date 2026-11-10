"""The pH neutralization's scenario suite (charter Amendment 3, rule 20).

Each scenario is an operating window the family's sources show; timing and
size inside it are drawn by the seed. Sources are the tags of this folder's
``REALITY.md``.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""


def _row(name, overrides, **kw):
    """One scenario as a plain row: the package imports nothing outside its
    fingerprint, and the registry builds the ``Scenario``."""
    return dict(name=name, overrides=overrides, **kw)


_QUIET = {"acid_upset_size": 0.0, "flow_swing": 0.0}

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1). No source gives
#: how often each window occurs or how long an upset lasts (REALITY.md, "How often
#: each window occurs": not found), so every number here is ours, set once without
#: tuning and open for the owner. A normal day lasts a day; an upset regime lasts a
#: shift and brings one upset per visit on average (the buffer loss is the regime
#: itself and brings none). An influent upset holds 2 h, until the producing unit's
#: batch ends, then ramps back.
_DAY_H, _SHIFT_H, _UPSET_HOLD_H = 24.0, 8.0, 2.0
#: A stream starts none of its window's upsets: only the stream starts them.
_NO_WINDOW_UPSETS = {"window_upsets": 0.0}


def _upset(events=True):
    """An upset regime's stream fields: a shift's dwell, one upset per visit."""
    return dict(
        dwell_h=_SHIFT_H,
        events_per_h=1.0 / _SHIFT_H if events else 0.0,
        stream_overrides={**_NO_WINDOW_UPSETS, "event_hold": _UPSET_HOLD_H * 3600.0},
    )


def ph_neutralization_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            dict(_QUIET),
            source="[P5, p. 1]; [P2]",
            description="waste flows in, buffering drifts, reagent holds the pH",
            weight=0.7,
            dwell_h=_DAY_H,
            stream_overrides=_NO_WINDOW_UPSETS,
        ),
        _row(
            "acid_strength",
            {**_QUIET, "acid_upset_size": 0.2},
            source="[P3, p. 4]",
            description="unknown changes in the influent's strong acid",
            weight=0.1,
            **_upset(),
        ),
        _row(
            "flow_swing",
            {**_QUIET, "flow_swing": 0.2},
            source="[P3, Figure 2]; charter Amendment 3",
            description="the influent flow swings, metered",
            weight=0.1,
            **_upset(),
        ),
        _row(
            "buffer_loss",
            {**_QUIET, "q2_nominal": 0.0},
            source="[P1]; [P2]",
            description="buffering falls away and the curve steepens",
            weight=0.05,
            **_upset(events=False),
        ),
        _row(
            "bad_day",
            {"acid_upset_size": 0.2, "flow_swing": 0.2},
            source="[P3, p. 8]",
            description="calm to rocking and reeling: acid and flow both move",
            weight=0.05,
            **_upset(),
        ),
    )
