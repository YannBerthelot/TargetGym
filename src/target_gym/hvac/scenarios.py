"""The slab building's scenario suite (charter Amendment 3, rule 20).

Two weeks each, from REALITY.md sec. 4 and 5: a normal heating fortnight, a
cold snap, the shoulder season near the heating limit, a sunny spell, and low
internal gains. The weather's draw and the day's gains still come from the
seed and the ranges in ``env.HVACParams``.

Each row holds ``target_gym.scenarios.Scenario``'s fields; the registry builds
them, so this package imports nothing outside its fingerprint.
"""


def hvac_scenarios() -> tuple:
    return (
        dict(
            name="normal",
            overrides={},
            source="REALITY.md sec. 5: the slab on a near-constant setpoint from "
            "the 24 h outdoor mean, gains absorbed by self-regulation [7, p. 7], "
            "[9, p. 7]",
            description="a heating fortnight, 5 C mean",
            # fortnights of a 13-fortnight heating season. Ours, family range 4 to 8
            weight=6.0,
        ),
        dict(
            name="cold_snap",
            overrides={"T_out_mean": -5.0},
            source="REALITY.md sec. 5: a heater on outdoor reset 'may never "
            "recover from setback' at design conditions [15]; tune the curve "
            "below 0 C [11, p. 1]",
            description="a cold fortnight, -5 C mean, near the -10 C design point",
            # fortnights a season. Ours, family range 0.5 to 2
            weight=1.0,
        ),
        dict(
            name="shoulder_season",
            overrides={"T_out_mean": 12.0},
            source="REALITY.md sec. 2 and 5: heating limits of 8 to 15 C [11, p. 3], "
            "13.1 C [9, p. 5]; tune the curve again above 10 C [11, p. 1]",
            description="a mild fortnight, 12 C mean, around the heating limit",
            # fortnights a season. Ours, family range 3 to 5: the months around the heating limit [11]
            weight=4.0,
        ),
        dict(
            name="sunny_spell",
            overrides={"solar_peak": 350.0},
            source="REALITY.md sec. 5: sun and gains above design; shading left out "
            "to produce intra-day swings [6, pp. 4-5]; buildings that perform well "
            "shade [8, pp. 20-21]",
            description="a clear fortnight: twice the season's mean sun on the glass",
            # fortnights a season. Ours, family range 1 to 3
            weight=1.5,
        ),
        dict(
            name="low_gains",
            overrides={"gain_occupied": 2.0},
            source="REALITY.md sec. 5: weekends and other days with fewer internal "
            "gains are a bad day for TABS control [9, part 2]",
            description="occupied hours with the unoccupied gains only",
            # fortnights a season. Ours, family range 0.3 to 1: holidays and low-occupancy weeks [9]
            weight=0.5,
        ),
    )
