"""The jacketed CSTR's scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of operation the family's sources show; its event's
minute is drawn by the seed (``CSTRParams.event_window``). Sources are the
tags of this folder's ``REALITY.md``.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1). No source gives
#: how often each window occurs or how long an upset lasts (REALITY.md, "How often
#: each window occurs": not found), so every number here is ours, set once without
#: tuning and open for the owner. A normal day lasts a day; an upset regime lasts a
#: shift and brings one event per visit on average. A setpoint move holds for a
#: shift, a catalyst loss until the next catalyst change (a day), a feed rise until
#: its batch runs out (2 h); the cooling loss keeps its 5 min.
_DAY_H, _SHIFT_H = 24.0, 8.0
#: A stream starts no event of its own window: only the stream starts events.
_NO_WINDOW_EVENT = {"event_window": (1.0e9, 1.0e9)}


def _upset(hold_min=None):
    """An upset regime's stream fields: a shift's dwell, one event per visit."""
    held = {} if hold_min is None else {"event_hold": hold_min}
    return dict(
        dwell_h=_SHIFT_H,
        events_per_h=1.0 / _SHIFT_H,
        stream_overrides={**_NO_WINDOW_EVENT, **held},
    )


def _row(name, overrides, **kw):
    """One scenario as a plain row: the package imports nothing outside its
    fingerprint, and the registry builds the ``Scenario``."""
    return dict(name=name, overrides=overrides, **kw)


def cstr_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            {},
            source="[S1, Tables 3, 4]",
            description="the setpoint held; the feed at 23 to 25 C by the day",
            weight=0.7,
            dwell_h=_DAY_H,
            stream_overrides=_NO_WINDOW_EVENT,
        ),
        _row(
            "setpoint_step",
            {"setpoint_step": 2.0},
            source="[S1, Fig. 2]",
            description="the reactor's setpoint moved up 2 K",
            weight=0.1,
            **_upset(hold_min=_SHIFT_H * 60.0),
        ),
        _row(
            "catalyst_loss",
            {"k0_drop": 0.4, "k0_decay": 90.0},
            source="[S1, Fig. 3; p. 6]",
            description=(
                "S1's 40 % loss of activity as a decay over 90 min (rate ours:"
                " the slowest that ends inside the episode)"
            ),
            weight=0.1,
            **_upset(hold_min=_DAY_H * 60.0),
        ),
        _row(
            "feed_buildup",
            {"feed_rise": 0.1},
            source="[S3, PDF pp. 36-37, 46]",
            description="the feed's reactant rises 10 % (size ours)",
            weight=0.05,
            **_upset(hold_min=120.0),
        ),
        _row(
            "cooling_loss",
            {"cooling_loss": 5.0},
            source="[S2, PDF p. 17]; [S3, Fig. 11]",
            description="the cooling water fails for 5 min (length ours)",
            weight=0.05,
            **_upset(),
        ),
    )
