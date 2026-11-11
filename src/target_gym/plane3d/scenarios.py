"""Scenario suites of ``plane3d_heading`` and ``plane3d_racetrack`` (charter
Amendment 3, rule 20).

Each scenario is a window the family's sources show (plane/REALITY.md); inside it
the seed still draws the level, the headings, the pattern's orientation (so a
fixed wind meets each seed's pattern from its own side) and the zone. Numbers
marked "ours" lie inside the family range beside them (plane3d/PHYSICS.md).
"""

from target_gym.scenarios import Scenario

# Weights, each scenario's share of operating time (Amendment 3, J5). Turbulence
# as plane/scenarios.py [BEA447 Figure 24]. Winds aloft: no source in REALITY.md
# gives how often a wind is met, so the shares are ours, family range: a steady
# 30 kt wind 0.1 to 0.3 of the time, 50 kt 0.05 to 0.2, the 67 kt design wind
# 0.005 to 0.05 (the holding area is built to contain a rare wind).

FT, KT = 0.3048, 1852.0 / 3600.0
#: The light zone's 0.564 m/s (0.2 g peak to peak, derived for the aircraft at
#: 5 500 m on its own lift slope, plane3d/PHYSICS.md) and the recorded worst,
#: 0.5 g: 2.5 times it.
WORST = 2.5 * 0.564
#: MIL-F-8785C's severe intensity at 5 500 m (Figure 7, 1e-5 exceedance, as
#: JSBSim tabulates it): 22.1 ft/s at 15 000 ft, 20.0 at 25 000, so 21.5 ft/s.
SEVERE = 21.46 * FT
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
            weight=0.35,
        ),
        Scenario(
            "calm",
            {"turbulence_sigma": 0.0},
            source="RVSM A.4.1.4 (nonturbulent air)",
            description="still air",
            weight=0.3,
        ),
        Scenario(
            "worst_zone",
            {"turbulence_sigma": WORST},
            source="BEA447 Figure 24 (zones up to 0.5 g peak to peak)",
            description="one zone at the recorded worst",
            weight=0.05,
        ),
        Scenario(
            "moderate_wind",
            {"wind_x": MODERATE},
            source="IFH ch. 10 (holds and tracks are flown correcting for wind); ours, 30 kt",
            description="a steady 30 kt wind",
            weight=0.2,
        ),
    ) + (_strong() if strong else ())


def _strong():
    return (
        Scenario(
            "strong_wind",
            {"wind_x": STRONG},
            source="IFH ch. 10; ours, 50 kt, under the design wind at every level",
            description="a steady 50 kt wind",
            weight=0.1,
        ),
    )


def heading_suite():
    return _common() + (
        Scenario(
            "severe_zone",
            {"turbulence_sigma": SEVERE},
            source="MIL-F-8785C Figure 7, severe (1e-5 exceedance) at 5 500 m",
            description="vectors through one zone of severe turbulence, 6.5 m/s",
            weight=1e-4,
        ),
        Scenario(
            "long_zone",
            {"zone_onset_range": (0.0, 0.0), "zone_duration_range": (600.0, 600.0)},
            source="BEA447 Figure 24 (zones of up to 30 min)",
            description="vectors through a zone covering the whole window",
            weight=0.05,
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
            weight=0.07,
        ),
        Scenario(
            "design_wind_low",
            {"wind_x": 67.0 * KT, "target_altitude_range": low},
            source="PANS-OPS design wind 2h + 47 kt at 10 000 ft (secondary, REALITY.md)",
            description="a low hold, 1 min legs, in the wind its area is built for",
            weight=0.02,
        ),
        Scenario(
            "high_hold_wind",
            {"wind_x": STRONG, "target_altitude_range": (14_500.0 * FT, 8_000.0)},
            source="IFH ch. 10 (1.5 min legs and 265 KIAS above 14 000 ft); ours, 50 kt",
            description="a high hold, 1.5 min legs, in a 50 kt wind",
            weight=0.08,
        ),
    )
