"""The reactor's scenario suite: days of load follow the family's sources show
(REALITY.md, section 6; charter rule 20). Inside each, the seed draws when the ramp
starts and, where the source has one, how long the hold lasts.

Each entry is the keyword arguments of a ``target_gym.scenarios.Scenario``; the
registry builds them, so this package imports nothing outside itself.
"""

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


_DAY = _stream(LOAD_FOLLOW_DWELL_H, 1.0 / DAY_H)


def suite():
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
