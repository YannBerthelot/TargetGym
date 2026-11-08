"""The LDPE autoclave's scenario suite (charter Amendment 3, rules 20, J4).

Each scenario is a window of operation the family's sources show; its event's
second is drawn by the seed (``UnstableCSTRParams.event_window``). Weights are
shares of operating time: none is in the sources, so each is ours, inside the
family range its PHYSICS.md states. Sources are this folder's ``REALITY.md``.
"""


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
        ),
        _row(
            "grade_change",
            {"setpoint_step": 24.0},
            source="[L1, p. 3: zones at 513 to 537 K]; [L2, section 3.2]",
            description="a grade change moves the zone's setpoint up 24 K",
            weight=0.1,
        ),
        _row(
            "feed_temperature",
            {"feed_T_step": 5.0},
            source="[L1, p. 3]",
            description="the cold feed, the reactor's only cooling, warms 5 K "
            "(size ours)",
            weight=0.1,
        ),
        _row(
            "initiator_efficiency",
            {"efficiency_drop": 0.2},
            source="[L2, p. 21-22]",
            description="the initiator yields 20 % fewer radicals (size ours)",
            weight=0.05,
        ),
        _row(
            "throughput_step",
            {"feed_flow_step": 0.1},
            source="[L2, section 3.2]; [L3]",
            description="the ethylene feed rises 10 % (size ours)",
            weight=0.05,
        ),
    )
