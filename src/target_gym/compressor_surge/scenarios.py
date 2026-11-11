"""The compressor's scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of the consumers' demand the family's sources
describe; levels and timing inside it are drawn by the seed. Sources are the
tags of this folder's ``REALITY.md``. Sizes and rates are ours: the sources
name the cases, not their numbers.

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
#: rest is ours, set once without tuning and open for the owner (no source gives how
#: long low or swinging demand lasts): an upset regime lasts a shift, 8 h (shifts
#: run 8 to 12 h [ST2, ST3]). No regime brings events of its own: its demand moves
#: are its 20 s blocks, which keep coming as the schedule renews
#: (``env.renew_schedule``), so the window's event-free suite carries over as it is.
_DAY_H, _SHIFT_H = 24.0, 8.0


def compressor_surge_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            {"demand_range": (0.7, 0.95), "demand_ramp_s": 5.0},
            source="[C3, p. 2]",
            description="demand high, the point right of the control line, "
            "the recycle shut or nearly",
            weight=0.7,
            dwell_h=_DAY_H,
        ),
        _row(
            "slow_approach",
            {"demand_range": (0.25, 0.6), "demand_ramp_s": 15.0},
            source="[C3, p. 2]",
            description="demand falls slowly toward the control line",
            weight=0.1,
            dwell_h=_SHIFT_H,
        ),
        _row(
            "fast_approach",
            {"demand_range": (0.2, 0.6), "demand_ramp_s": 1.0},
            source="[C3, p. 2]",
            description="demand falls fast; rate action or the backup line",
            weight=0.05,
            dwell_h=_SHIFT_H,
        ),
        _row(
            "load_swings",
            {"demand_range": (0.35, 0.95), "demand_ramp_s": 5.0},
            source="[C3, pp. 2-4]",
            description="demand swings across the map, both approaches",
            weight=0.15,
            dwell_h=_SHIFT_H,
        ),
    )
