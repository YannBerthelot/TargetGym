"""The drum boiler's scenario suite (charter Amendment 3, rule 20).

Each scenario is an hour of the unit's operation drawn from the family's
sources (REALITY.md): normal load following, fast and industrial load swings,
low-load operation, and the two bad days the record names for drum level, an
ID fan runback and a feed pump trip. Timing and sizes inside each are drawn by
the seed from the ranges in ``env.BoilerDrumParams``.

Each row holds ``target_gym.scenarios.Scenario``'s fields; the registry builds
them, so this package imports nothing outside its fingerprint.
"""


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
        ),
        dict(
            name="industrial_swing",
            overrides={"ramp_rate_range": (15.0, 25.0)},
            source="REALITY.md sec. 4: industrial steam flow ramps 'up to 20-25% per "
            "minute' [S15]; unmeasured user demand at Abbott [S3]",
            description="one steam demand swing at 15 to 25 %/min",
            # hours a year. Ours, family range 0 to 2000: the industrial members of the family [S3, S15]
            weight=800.0,
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
        ),
        dict(
            name="id_fan_runback",
            overrides={"event_kind": 1, "load_range": (0.75, 1.0)},
            source="REALITY.md sec. 5: an ID fan trip ran a 95 MW unit back 'in just "
            "over two minutes' to below the three-mill minimum [S10, Figs. 5, 6]",
            description="an ID fan trip caps firing and runs the unit back from 75 to "
            "100 % load",
            # hours a year. Ours, family range 1 to 5 runbacks a year, an hour each; no source gives the rate
            weight=3.0,
        ),
        dict(
            name="feed_pump_trip",
            overrides={"event_kind": 2, "load_range": (0.75, 1.0)},
            source="REALITY.md sec. 5: a feed pump runback from 75 % load dropped the "
            "level 6.5 in against a 7.7 in trip [S10, Fig. 7]; the standby pump takes "
            "over 30 s [S11]",
            description="one of two feed pumps trips, the unit runs back, and the "
            "standby pump delivers after 30 to 60 s",
            # hours a year. Ours, family range 1 to 3 trips a year, an hour each; no source gives the rate
            weight=2.0,
        ),
    )
