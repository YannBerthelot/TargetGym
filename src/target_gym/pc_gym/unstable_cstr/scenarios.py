"""The LDPE autoclave's scenario suite (charter Amendment 3, rules 20, J4).

Each scenario is a window of operation the family's sources show; its event's
second is drawn by the seed (``UnstableCSTRParams.event_window``). Weights are
shares of operating time: none is in the sources, so each is ours, inside the
family range its PHYSICS.md states. Sources are this folder's ``REALITY.md``.
"""

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1), from REALITY.md
#: (Amendment 3, "Stream numbers, second search", 2026-11-11). A normal day lasts a
#: day: a large process plant meets an abnormal situation almost daily [ST1]. The
#: rest is ours, set once without tuning and open for the owner: an upset regime
#: lasts a shift, 8 h (shifts run 8 to 12 h [ST2, ST3]), and brings one event per
#: visit on average. A real grade holds about two weeks and its transition takes 1 to
#: 3 h [ST4, ST5]; the 8 h hold below is kept until the owner rules on a grade change
#: as a level switch (REALITY.md). A grade change and a
#: throughput step hold for a shift, a warm feed until its cooler recovers (2 h), an
#: initiator's lost efficiency until its batch is changed (a day).
_DAY_H, _SHIFT_H = 24.0, 8.0
#: A stream starts no event of its own window: only the stream starts events.
_NO_WINDOW_EVENT = {"event_window": (1.0e9, 1.0e9)}


def _upset(hold_h):
    """An upset regime's stream fields: a shift's dwell, one event per visit, its
    event held ``hold_h`` hours before it returns."""
    return dict(
        dwell_h=_SHIFT_H,
        events_per_h=1.0 / _SHIFT_H,
        stream_overrides={**_NO_WINDOW_EVENT, "event_hold": hold_h * 3600.0},
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
            description="the zone held at 513 K",
            weight=0.7,
            dwell_h=_DAY_H,
            stream_overrides=_NO_WINDOW_EVENT,
        ),
        _row(
            "grade_change",
            {"setpoint_step": 24.0},
            source="[L1, p. 3: zones at 513 to 537 K]; [L2, section 3.2]",
            description="a grade change moves the zone's setpoint up 24 K",
            weight=0.1,
            **_upset(hold_h=_SHIFT_H),
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
