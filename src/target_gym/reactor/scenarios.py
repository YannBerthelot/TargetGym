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


def suite():
    return (
        dict(
            name="edf_daily",
            weight=EACH_DAY,
            overrides={},
            source="REALITY.md section 6, [S1] annex II-4.3 and [S2] section 2.2",
            description="100-50-100 at 2 %/min, held 3 to 8 h",
        ),
        dict(
            name="edf_short_hold",
            weight=EACH_DAY,
            overrides={"hold_range": (2.0 * HOUR, 3.0 * HOUR)},
            source="REALITY.md section 6, [S1] annex II-4.3: plateaus of at least 2 h",
            description="100-50-100 at 2 %/min, held the shortest plateau, 2 to 3 h",
        ),
        dict(
            name="krsko_slow",
            weight=EACH_DAY,
            overrides={"load_ramp_rate": 0.08 / HOUR, "hold_range": (0.0, 0.0)},
            source="REALITY.md section 6, [S3] section 3, day 1",
            description="100-50-100 at about 8 %/h and straight back",
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
        ),
        dict(
            name="fast_return",
            weight=EACH_DAY,
            overrides={"load_ramp_rate": 0.05 / 60.0},
            source="REALITY.md section 6, [S3] quoting the EUR, and [S8]",
            description="100-50-100 at 5 %/min, the EUR's fastest, held 3 to 8 h",
        ),
        dict(
            name="baseload",
            weight=1.0 - LOAD_FOLLOW,
            overrides={"ramp_down_range": (1e6, 1e6)},
            source="REALITY.md section 6, [S1] section 2: frequency control",
            description="full load all day with the frequency-control wobble",
        ),
    )
