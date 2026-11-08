"""``plane``'s scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of cruise the family's sources show (plane/REALITY.md);
inside it the seed still draws the level, the zone's onset and length. Numbers
marked "ours" lie inside the family range given beside them (plane/PHYSICS.md).
"""

from target_gym.scenarios import Scenario

# Weights, each scenario's share of cruise time (Amendment 3, J5). Turbulence:
# zones of 3 to 30 min with calm of over an hour between [BEA447 Figure 24], so a
# zone is in force at most about a third of the time (30 / 90) and typically a
# fifth (16.5 / 76.5); a window with a zone in it is more likely than one without
# (ours: normal 0.40, calm 0.35, family range 0.2 to 0.6 each). The worst
# recorded zone and a zone over the whole window are rare (ours, 0.05 each,
# family range 0.01 to 0.1: Figure 24 shows most zones at 0.2 g or below). The
# mass and level corners (ours, 0.075 each, family range 0.05 to 0.2).

FT = 0.3048
#: The light zone's sigma, 0.2 g peak to peak a minute at FL330 (derived,
#: plane/PHYSICS.md); the worst recorded zone, 0.5 g, is 2.5 times it, the
#: gust load being linear in the gust.
LIGHT = 0.598
WORST = 2.5 * LIGHT


def suite():
    return (
        Scenario(
            "normal",
            {},
            source="BEA447 Figure 24: cruise on autopilot, one zone of 3 to 30 min",
            description="one light zone, 0.2 g peak to peak, anywhere in 30 min",
            weight=0.4,
        ),
        Scenario(
            "calm",
            {"turbulence_sigma": 0.0},
            source="RVSM A.4.1.4 (nonturbulent air); BEA447 Figure 24 (calm over an hour)",
            description="still air the whole window",
            weight=0.35,
        ),
        Scenario(
            "worst_zone",
            {"turbulence_sigma": WORST},
            source="BEA447 Figure 24 (zones up to 0.5 g peak to peak); ICAO light (< 0.5 g)",
            description="one zone at the recorded worst, 0.5 g peak to peak",
            weight=0.05,
        ),
        Scenario(
            "long_zone",
            {"zone_onset_range": (0.0, 0.0), "zone_duration_range": (1800.0, 1800.0)},
            source="BEA447 Figure 24 (zones of up to 30 min)",
            description="a 30 min zone covering the whole window",
            weight=0.05,
        ),
        Scenario(
            "heavy_high",
            {
                "initial_mass": 78_000.0,
                "target_altitude_range": (35_000.0 * FT, 37_000.0 * FT),
                "initial_altitude_range": (35_000.0 * FT, 37_000.0 * FT),
            },
            source="TCDS (MTOW 78 t, maximum altitude 39 100 ft); RVSM levels",
            description="78 t at FL350 to 370, the thinnest thrust and buffet margin",
            weight=0.075,
        ),
        Scenario(
            "light_low",
            {
                "initial_mass": 60_000.0,
                "target_altitude_range": (29_000.0 * FT, 31_000.0 * FT),
                "initial_altitude_range": (29_000.0 * FT, 31_000.0 * FT),
            },
            source="ours: 60 t, family range 42.6 t (OEW) to 78 t (MTOW) [PHYSICS.md]",
            description="60 t at FL290 to 310, the lightest wing loading, most gust response",
            weight=0.075,
        ),
    )
