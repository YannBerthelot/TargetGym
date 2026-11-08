"""Scenario suites of ``plane3d_heading`` and ``plane3d_racetrack`` (charter
Amendment 3, rule 20).

Each scenario is a window the family's sources show (plane/REALITY.md); inside it
the seed still draws the level, the headings, the pattern's orientation (so a
fixed wind meets each seed's pattern from its own side) and the zone. Numbers
marked "ours" lie inside the family range beside them (plane3d/PHYSICS.md).
"""

from target_gym.scenarios import Scenario

FT, KT = 0.3048, 1852.0 / 3600.0
#: The light zone's 0.55 m/s (0.2 g peak to peak, derived for the aircraft at
#: 5 500 m, plane/PHYSICS.md) and the recorded worst, 0.5 g: 2.5 times it.
WORST = 2.5 * 0.55
#: Steady winds aloft, ours, inside the family range: calm up to the holding
#: area's design wind, 2h + 47 kt at h thousand feet (PANS-OPS Doc 8168 as
#: quoted by secondary sources, plane/REALITY.md), 67 kt at 10 000 ft.
MODERATE, STRONG = 30.0 * KT, 50.0 * KT


def _common(strong=True):
    return (
        Scenario(
            "normal",
            {},
            source="BEA447 Figure 24: one light zone of 3 to 30 min",
            description="one light zone, still mean air",
        ),
        Scenario(
            "calm",
            {"turbulence_sigma": 0.0},
            source="RVSM A.4.1.4 (nonturbulent air)",
            description="still air",
        ),
        Scenario(
            "worst_zone",
            {"turbulence_sigma": WORST},
            source="BEA447 Figure 24 (zones up to 0.5 g peak to peak)",
            description="one zone at the recorded worst",
        ),
        Scenario(
            "moderate_wind",
            {"wind_x": MODERATE},
            source="IFH ch. 10 (holds and tracks are flown correcting for wind); ours, 30 kt",
            description="a steady 30 kt wind",
        ),
    ) + (_STRONG if strong else ())


_STRONG = (
    Scenario(
        "strong_wind",
        {"wind_x": STRONG},
        source="IFH ch. 10; ours, 50 kt, under the design wind at every level",
        description="a steady 50 kt wind",
    ),
)


def heading_suite():
    return _common() + (
        Scenario(
            "long_zone",
            {"zone_onset_range": (0.0, 0.0), "zone_duration_range": (600.0, 600.0)},
            source="BEA447 Figure 24 (zones of up to 30 min)",
            description="vectors through a zone covering the whole window",
        ),
    )


def racetrack_suite():
    low = (3_000.0, 14_000.0 * FT)
    return _common(strong=False) + (
        Scenario(
            "low_hold_wind",
            {"wind_x": STRONG, "target_altitude_range": low},
            source="IFH ch. 10 (1 min legs and 230 KIAS to 14 000 ft); ours, 50 kt",
            description="a low hold, 1 min legs, in a 50 kt wind",
        ),
        Scenario(
            "design_wind_low",
            {"wind_x": 67.0 * KT, "target_altitude_range": low},
            source="PANS-OPS design wind 2h + 47 kt at 10 000 ft (secondary, REALITY.md)",
            description="a low hold, 1 min legs, in the wind its area is built for",
        ),
        Scenario(
            "high_hold_wind",
            {"wind_x": STRONG, "target_altitude_range": (14_500.0 * FT, 8_000.0)},
            source="IFH ch. 10 (1.5 min legs and 265 KIAS above 14 000 ft); ours, 50 kt",
            description="a high hold, 1.5 min legs, in a 50 kt wind",
        ),
    )
