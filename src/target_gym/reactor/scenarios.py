"""The reactor's scenario suite: days of load follow the family's sources show
(REALITY.md, section 6; charter rule 20). Inside each, the seed draws when the ramp
starts and, where the source has one, how long the hold lasts.

Each entry is the keyword arguments of a ``target_gym.scenarios.Scenario``; the
registry builds them, so this package imports nothing outside itself.
"""

import math

HOUR = 3600.0

# Shares of operating time (REALITY.md section 7). EDF's units load-follow 250 h a
# year (IEA 2019 workshop, EDF, slide 7); an EDF day is off full power for its mean
# hold, 5.5 h, and two 25 min ramps. The five load-follow days split that share
# equally (ours: no source splits it); baseload days take the rest.
LOAD_FOLLOW_HOURS = 250.0
DAY_OFF_FULL = 5.5 + 2.0 * 25.0 / 60.0
LOAD_FOLLOW = LOAD_FOLLOW_HOURS / DAY_OFF_FULL / 365.0
EACH_DAY = LOAD_FOLLOW / 5.0

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1). REALITY.md
#: section 8 gives the sources and the searches; a number marked ours was set once,
#: without tuning. A load-follow scenario is a day: the dispatch plans one day's
#: program in it on average. Baseload days have no event.
DAY_H = 24.0
LOAD_FOLLOW_DWELL_H = DAY_H  # ours
BASELOAD_DWELL_H = DAY_H  # ours: each day is a load-follow day at its share
#: A stream refuels the core every 30 days, renewed at its cycle point (owner,
#: 2026-11-11; 30 days ours: at 2.3 to 3.3 ppm a day of burnup it keeps the boron
#: within about 100 ppm of the start of cycle the coefficients describe).
RENEWAL_DAYS = 30.0
#: A stream starts with no ramp of its own window, nothing ahead and full load:
#: only the stream plans days.
_NO_WINDOW_DAY = {"ramp_down_range": (-1.0e7, -1.0e7)}


def _stream(dwell_h, events_per_h):
    """A regime's stream fields: its dwell and its rate of load-follow days."""
    return dict(
        dwell_h=dwell_h,
        events_per_h=events_per_h,
        stream_overrides={**_NO_WINDOW_DAY, "renewal_days": RENEWAL_DAYS},
    )


#: One draw a visit; ``suite`` raises it so that one day a visit is delivered.
_DAY = _stream(LOAD_FOLLOW_DWELL_H, 1.0 / DAY_H)

#: A day's program as ``ReactorParams`` draws it, unless a row overrides it: the
#: dispatch announces it ``day_notice_range`` ahead of the ramp down, the load ramps
#: to ``load_min`` at ``load_ramp_rate``, holds over ``hold_range`` and ramps back.
NOTICE_RANGE = (1800.0, 7200.0)  # s
DEFAULT_DAY = {
    "load_min": 0.5,
    "load_ramp_rate": 0.02 / 60.0,
    "hold_range": (3.0 * HOUR, 8.0 * HOUR),
}


def _outlasts(lo, hi):
    """E[exp(-T / dwell)] for T uniform over [lo, hi] s: the chance a load-follow
    visit outlasts it."""
    k = 1.0 / (LOAD_FOLLOW_DWELL_H * HOUR)
    if hi == lo:
        return math.exp(-k * lo)
    return (math.exp(-k * lo) - math.exp(-k * hi)) / (k * (hi - lo))


def _one_day_a_visit(row):
    """``row`` drawing load-follow days at the rate that delivers one a visit on
    average (owner, 2026-11-11), EDF's 39.5 days a year over the load-follow
    share's 39.5 visits.

    A day drawn while one is under way, from its announcement until its return
    ramp ends, is absorbed (``env_jax.start_event``), so one draw a visit
    delivered 0.70 to 0.85 days. A visit that starts with no day under way gets a
    draw before it ends with chance p = lam / (lam + 1/dwell) and outlasts a day's
    program with chance q, so it delivers p / (1 - p q) days; one a visit needs
    lam = 1 / (dwell q). q multiplies the chances the visit outlasts the notice,
    the two ramps and the hold."""
    day = {**DEFAULT_DAY, **row["overrides"]}
    ramp = (1.0 - day["load_min"]) / day["load_ramp_rate"]
    q = _outlasts(*NOTICE_RANGE) * _outlasts(2 * ramp, 2 * ramp)
    q *= _outlasts(*day["hold_range"])
    return {**row, "events_per_h": 1.0 / (LOAD_FOLLOW_DWELL_H * q)}


def suite():
    # Each load-follow row draws its days faster than one a visit, by its own day's
    # shape, so that it delivers one a visit (``_one_day_a_visit``).
    return tuple(
        _one_day_a_visit(row) if row["events_per_h"] else row for row in _rows()
    )


def _rows():
    return (
        dict(
            name="edf_daily",
            weight=EACH_DAY,
            overrides={},
            source="REALITY.md section 6, [S1] annex II-4.3 and [S2] section 2.2",
            description="100-50-100 at 2 %/min, held 3 to 8 h",
            **_DAY,
        ),
        dict(
            name="edf_short_hold",
            weight=EACH_DAY,
            overrides={"hold_range": (2.0 * HOUR, 3.0 * HOUR)},
            source="REALITY.md section 6, [S1] annex II-4.3: plateaus of at least 2 h",
            description="100-50-100 at 2 %/min, held the shortest plateau, 2 to 3 h",
            **_DAY,
        ),
        dict(
            name="krsko_slow",
            weight=EACH_DAY,
            overrides={"load_ramp_rate": 0.08 / HOUR, "hold_range": (0.0, 0.0)},
            source="REALITY.md section 6, [S3] section 3, day 1",
            description="100-50-100 at about 8 %/h and straight back",
            **_DAY,
        ),
        dict(
            name="krsko_shallow",
            weight=EACH_DAY,
            overrides={
                "load_min": 0.75,
                "load_ramp_rate": 0.04 / HOUR,
                "hold_range": (0.0, 0.0),
            },
            source="REALITY.md section 6, [S3] section 3, day 2",
            description="100-75-100 at about 4 %/h and straight back",
            **_DAY,
        ),
        dict(
            name="fast_return",
            weight=EACH_DAY,
            overrides={"load_ramp_rate": 0.05 / 60.0},
            source="REALITY.md section 6, [S3] quoting the EUR, and [S8]",
            description="100-50-100 at 5 %/min, the EUR's fastest, held 3 to 8 h",
            **_DAY,
        ),
        dict(
            name="baseload",
            weight=1.0 - LOAD_FOLLOW,
            overrides={"ramp_down_range": (1e6, 1e6)},
            source="REALITY.md section 6, [S1] section 2: frequency control",
            description="full load all day with the frequency-control wobble",
            **_stream(BASELOAD_DWELL_H, 0.0),
        ),
    )
