"""The slab building's scenario suite (charter Amendment 3, rule 20).

From REALITY.md sec. 4 and 5: a normal heating fortnight, a cold snap and the
shoulder season near the heating limit. The weather's draw and the day's gains
still come from the seed and the ranges in ``env.HVACParams``.

The evaluation cut (owner, 2026-11-24): a scenario stays only if removing it would
hide a named behaviour or change which controller looks better. Normal shows the
night's charge of the slab, the cold snap the heat source at its size (it alone
places the reference's set point), and the shoulder season the work near the
heating limit. The low-gains weeks repeated normal's night charge and the sunny
spell the shoulder's heating limit, so their weights (0.5 and 1.5 fortnights a
season) fold into those two. Their overrides stay plant parameters any run may set.

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
            # fortnights of a 13-fortnight heating season: 6, and the low-gains
            # half fortnight folded in. Ours, family range 4 to 8
            weight=6.5,
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
            # fortnights a season: 4, and the sunny spell's 1.5 folded in. Ours,
            # family range 3 to 5 before the fold: the months around the heating
            # limit [11]
            weight=5.5,
            dwell_h=_FORTNIGHT_H,
        ),
    )


#: A day of 15-minute steps.
_DAY = 96


def hvac_evaluation() -> dict:
    """What an evaluation reads (``target_gym.evaluation.Evaluation``; owner,
    2026-11-24). Days 1 to 3 burn the slab in: the cost gap between the reference
    and the ceiling is gone after day 3, where day 1 alone held 46 % of the
    fortnight's (hvac review, 2026-11-24). Days 4 to 10 are scored, seven of the two
    weeks a heating curve is judged over [11, p. 1]. Days 11 and 12 are the
    ceiling's tail: charged in its plan, read by no line, so it cannot spend the
    slab's charge on the last scored night (day 14 held 11 % of the gap). Lines run
    on five seeds, and five more where the saving interval touches 0. The heat the
    slab and the mass lost across the scored days is charged once, at the env's
    ``stored_value`` (owner, 2026-11-25)."""
    return dict(
        unscored_steps=3 * _DAY,
        scored_steps=7 * _DAY,
        seeds=tuple(range(100, 105)),
        extra_seeds=tuple(range(105, 110)),
        edge_charge="stored_value",
    )
