"""The drum boiler's scenario suite (charter Amendment 3, rule 20).

Each scenario is an hour of the unit's operation drawn from the family's
sources (REALITY.md): normal load following, fast and industrial load swings,
low-load operation, and the two bad days the record names for drum level, an
ID fan runback and a feed pump trip. Timing and sizes inside each are drawn by
the seed from the ranges in ``env.BoilerDrumParams``.

Each row holds ``target_gym.scenarios.Scenario``'s fields; the registry builds
them, so this package imports nothing outside its fingerprint.
"""

import math

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1). REALITY.md ("How
#: often each window occurs", second search 2026-11-11) sources one: a runback or a
#: feed pump trip lasts 2.775 h, the median of the six ID fan and feed pump forced
#: derates in one plant-year of four units ([ST1], 0.35 to 67 h). It brings one
#: upset a visit and holds the unit back that long before it ramps up; at the
#: weights below that is 0.5 runbacks and one feed pump trip a year (sourced, 0.5
#: and 0.75 to 1.25 a unit-year [ST1, ST2]). [S3] has setpoints
#: "change several times during every 24 h". The rest is ours, set once without
#: tuning and open for the owner:
#: - normal load following lasts a day and changes load four times in it (S3's
#:   "several"; no count a year was found);
#: - industrial swings and low load last a shift and bring one load change a visit.
_DAY_H, _SHIFT_H, _UPSET_H = 24.0, 8.0, 2.775
_LOAD_CHANGES_PER_DAY = 4.0
#: A stream starts with neither of the window's events: its plan is flat until the
#: stream starts a load change or an upset.
_NO_WINDOW_EVENT = {"window_ramp": False, "event_time_range": (1.0e9, 1.0e9)}
_REGIME = dict(
    dwell_h=_SHIFT_H, events_per_h=1.0 / _SHIFT_H, stream_overrides=_NO_WINDOW_EVENT
)
#: An upset keeps the unit master busy, a draw then absorbed (``env_jax.start_event``),
#: for its hold and its ramp back to the plan: from the rows' 75 to 100 % load, run
#: back to 50 %, it climbs 37.5 % on average at the plan's 1 to 6 %/min, ln(6) / 5
#: min a % on average, so 13.4 min.
_UPSET_BUSY_H = _UPSET_H + 37.5 * math.log(6.0) / 5.0 / 60.0
#: Upsets drawn an hour in an upset visit, so that it delivers one on average (owner,
#: 2026-11-11): one draw a visit delivered 0.6. A visit that starts with no upset
#: under way gets a draw before it ends with chance p = lam / (lam + 1/dwell) and
#: outlasts an upset with chance q = exp(-busy/dwell), so it delivers p / (1 - p q)
#: upsets; one a visit needs lam = 1 / (dwell q), 1.06 an hour. The long-visit
#: form, r / (1 - r busy), has no solution here (r busy = 1.08), but a visit starts
#: free: a quarter of the visits bring no upset and a fifth bring two or more.
_UPSETS_PER_H = 1.0 / (_UPSET_H * math.exp(-_UPSET_BUSY_H / _UPSET_H))
_UPSET = dict(
    dwell_h=_UPSET_H,
    events_per_h=_UPSETS_PER_H,
    stream_overrides={**_NO_WINDOW_EVENT, "event_hold": _UPSET_H * 3600.0},
)


def boiler_drum_scenarios() -> tuple:
    return (
        dict(
            name="normal",
            overrides={},
            source="REALITY.md sec. 4 and 5: ramps of 1 to 2 %/min [S10], 1.5 to 6 "
            "for coal units [S17]; setpoints change several times a day [S3]",
            description="one planned ramp at 1 to 6 %/min between 40 and 100 % load",
            # hours a year. Ours, family range 4000 to 7000: most of a utility unit's running hours load-follow [S3, S10]
            weight=5600.0,
            dwell_h=_DAY_H,
            events_per_h=_LOAD_CHANGES_PER_DAY / _DAY_H,
            stream_overrides=_NO_WINDOW_EVENT,
        ),
        dict(
            name="industrial_swing",
            overrides={"ramp_rate_range": (15.0, 25.0)},
            source="REALITY.md sec. 4: industrial steam flow ramps 'up to 20-25% per "
            "minute' [S15]; unmeasured user demand at Abbott [S3]",
            description="one steam demand swing at 15 to 25 %/min",
            # hours a year. Ours, family range 0 to 2000: the industrial members of the family [S3, S15]
            weight=800.0,
            **_REGIME,
        ),
        dict(
            name="low_load",
            overrides={"load_range": (0.25, 0.50)},
            source="REALITY.md sec. 5: 'most trips occur due to high-level trip "
            "events' at low load [S11]; swell larger at low load [S1, S2]; turndown "
            "4:1 [S16]",
            description="one planned ramp at 1 to 6 %/min between 25 and 50 % load",
            # hours a year. Ours, family range 500 to 3000: part-load hours [S11, S17]
            weight=1600.0,
            **_REGIME,
        ),
        dict(
            name="id_fan_runback",
            overrides={"event_kind": 1, "load_range": (0.75, 1.0)},
            source="REALITY.md sec. 5: an ID fan trip ran a 95 MW unit back 'in just "
            "over two minutes' to below the three-mill minimum [S10, Figs. 5, 6]",
            description="an ID fan trip caps firing and runs the unit back from 75 to "
            "100 % load",
            # hours a year: 0.5 ID fan runbacks a unit-year of 2.775 h, one plant's 4 unit-years (REALITY.md [ST1]; owner, 2026-11-11)
            weight=1.4,
            **_UPSET,
        ),
        dict(
            name="feed_pump_trip",
            overrides={"event_kind": 2, "load_range": (0.75, 1.0)},
            source="REALITY.md sec. 5: a feed pump runback from 75 % load dropped the "
            "level 6.5 in against a 7.7 in trip [S10, Fig. 7]; the standby pump takes "
            "over 30 s [S11]",
            description="one of two feed pumps trips, the unit runs back, and the "
            "standby pump delivers after 30 to 60 s",
            # hours a year: 1 feed pump trip a unit-year of 2.775 h; sourced 0.75 to 1.25 a unit-year (REALITY.md [ST1, ST2]; owner, 2026-11-11)
            weight=2.8,
            **_UPSET,
        ),
    )
