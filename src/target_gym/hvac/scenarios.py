"""The slab building's scenario suite (charter Amendment 3, rule 20).

Two weeks each, from REALITY.md sec. 4 and 5: a normal heating fortnight, a
cold snap, the shoulder season near the heating limit, a sunny spell, and low
internal gains. The weather's draw and the day's gains still come from the
seed and the ranges in ``env.HVACParams``.

Each row holds ``target_gym.scenarios.Scenario``'s fields; the registry builds
them, so this package imports nothing outside its fingerprint.
"""

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1), from REALITY.md
#: ("How long each regime lasts", 2026-11-11). A cold snap lasts 5.5 days on
#: average: western European cold spells last 5 to 6 days in KNMI's model composite
#: [ST1]. The other regimes last, on average, the fortnight this suite defines them
#: as, inside the week to a month weather regimes last [ST2]; that number is ours,
#: set once without tuning and open for the owner (a sunny spell is not a block, and
#: office closures were not found). The regimes carry no event of their own:
#: ``events_per_h`` stays 0.
_FORTNIGHT_H = 14 * 24.0
_COLD_SPELL_H = 5.5 * 24.0


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
            dwell_h=_FORTNIGHT_H,
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
            dwell_h=_COLD_SPELL_H,
        ),
        dict(
            name="shoulder_season",
            overrides={"T_out_mean": 12.0},
            source="REALITY.md sec. 2 and 5: heating limits of 8 to 15 C [11, p. 3], "
            "13.1 C [9, p. 5]; tune the curve again above 10 C [11, p. 1]",
            description="a mild fortnight, 12 C mean, around the heating limit",
            # fortnights a season. Ours, family range 3 to 5: the months around the heating limit [11]
            weight=4.0,
            dwell_h=_FORTNIGHT_H,
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
            dwell_h=_FORTNIGHT_H,
        ),
        dict(
            name="low_gains",
            overrides={"gain_occupied": 2.0},
            source="REALITY.md sec. 5: weekends and other days with fewer internal "
            "gains are a bad day for TABS control [9, part 2]",
            description="occupied hours with the unoccupied gains only",
            # fortnights a season. Ours, family range 0.3 to 1: holidays and low-occupancy weeks [9]
            weight=0.5,
            dwell_h=_FORTNIGHT_H,
        ),
    )
