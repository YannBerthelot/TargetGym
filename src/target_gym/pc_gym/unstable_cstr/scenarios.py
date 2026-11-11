"""The LDPE autoclave's scenario suite (charter Amendment 3, rules 20, J4).

Each scenario is a window of operation the family's sources show; its event's
second is drawn by the seed (``UnstableCSTRParams.event_window``). Weights are
shares of operating time: the grade change's comes from the sources; the others
are ours, inside the family range its PHYSICS.md states. Sources are this
folder's ``REALITY.md``.
"""

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1), from REALITY.md
#: (Amendment 3, "Stream numbers, second search", 2026-11-11). A normal day lasts a
#: day: a large process plant meets an abnormal situation almost daily [ST1]. The
#: rest is ours, set once without tuning and open for the owner: an upset regime
#: lasts a shift, 8 h (shifts run 8 to 12 h [ST2, ST3]), and brings one event per
#: visit on average. A throughput step holds for a shift, a warm feed until its
#: cooler recovers (2 h), an initiator's lost efficiency until its batch is
#: changed (a day).
_DAY_H, _SHIFT_H = 24.0, 8.0
#: A stream starts no event of its own window: only the stream starts events.
_NO_WINDOW_EVENT = {"event_window": (1.0e9, 1.0e9)}

#: A grade is a held level (owner, 2026-11-11). The plant runs on one of two
#: grades, the zones held at 513 and 537 K [L1, p. 3], 24 K apart, and a grade
#: change moves it to the other. A specialty line sees about 2 transitions a
#: month [ST5], so a grade holds about two weeks, and a transition takes 50 min to
#: 3 h 5 min [ST4]; 2 h, the middle, is ours. The regime is the transition, with
#: one move a visit: its share is the time spent in transitions.
_GRADE_STEP = 24.0  # K
_MONTH_H = 365.25 * 24.0 / 12.0  # 730.5 h
_TRANSITION_H = 2.0
_GRADE_SHARE = 2.0 / _MONTH_H * _TRANSITION_H  # 0.005476
#: Every regime of a stream runs on the grade the plant holds.
_STREAM = {**_NO_WINDOW_EVENT, "grades": (0.0, _GRADE_STEP)}


def _upset(hold_h):
    """An upset regime's stream fields: a shift's dwell, one event per visit, its
    event held ``hold_h`` hours before it returns."""
    return dict(
        dwell_h=_SHIFT_H,
        events_per_h=1.0 / _SHIFT_H,
        stream_overrides={**_STREAM, "event_hold": hold_h * 3600.0},
    )


def _row(name, overrides, **kw):
    """One scenario as a plain row: the package imports nothing outside its
    fingerprint, and the registry builds the ``Scenario``."""
    return dict(name=name, overrides=overrides, **kw)


def unstable_cstr_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            {},
            source="[L1, p. 3]",
            description="the zone held at its grade's setpoint, 513 K in a window",
            # The rest of the time: the plant runs on a held grade.
            weight=1.0 - 0.1 - 0.05 - 0.05 - _GRADE_SHARE,
            dwell_h=_DAY_H,
            stream_overrides=_STREAM,
        ),
        _row(
            "grade_change",
            {"setpoint_step": _GRADE_STEP},
            source="[L1, p. 3: zones at 513 to 537 K]; [L2, section 3.2]; "
            "[ST4]; [ST5]",
            description="a grade change moves the zone's setpoint up 24 K; in a "
            "stream, to the other grade, held until the next change",
            weight=_GRADE_SHARE,
            dwell_h=_TRANSITION_H,
            events_per_h=1.0 / _TRANSITION_H,
            stream_overrides=_STREAM,
        ),
        _row(
            "feed_temperature",
            {"feed_T_step": 5.0},
            source="[L1, p. 3]",
            description="the cold feed, the reactor's only cooling, warms 5 K "
            "(size ours)",
            weight=0.1,
            **_upset(hold_h=2.0),
        ),
        _row(
            "initiator_efficiency",
            {"efficiency_drop": 0.2},
            source="[L2, p. 21-22]",
            description="the initiator yields 20 % fewer radicals (size ours)",
            weight=0.05,
            **_upset(hold_h=_DAY_H),
        ),
        _row(
            "throughput_step",
            {"feed_flow_step": 0.1},
            source="[L2, section 3.2]; [L3]",
            description="the ethylene feed rises 10 % (size ours)",
            weight=0.05,
            **_upset(hold_h=_SHIFT_H),
        ),
    )
