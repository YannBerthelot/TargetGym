"""The reference ladder's row for ``ph_neutralization`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Row,
    corners,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.pc_gym.ph_neutralization.experts import (
        TitrationCurve,
        reagent_demand,
    )

    spec = REGISTRY["ph_neutralization"]
    nominal = spec.make_test_params()
    test = {"q2_nominal": float(nominal.q2_min), "q2_noise_std": 0.0}
    curve = TitrationCurve.of(nominal)
    lab = TitrationCurve.of(spec.make_test_params(**test))
    low, span = float(nominal.q3_min), float(nominal.q3_max - nominal.q3_min)
    top = 2 * float(nominal.q3_max)

    def raw(q3, xp):
        return xp.reshape(2 * (q3 - low) / span - 1, (1,))

    # The loop runs in reagent demand, the titration curve's linearisation.
    demand = Cv(
        "demand on q3",
        0,
        lambda o, xp: xp.clip(reagent_demand(o[0], curve, xp), 0.0, top),
        lambda o, xp: reagent_demand(o[2], curve, xp),
        "pH",
        0.1 * 2 / span,
        120,
    )
    return Row(
        300,
        lambda o, xp: raw(reagent_demand(o[2], lab, xp), xp),
        (demand,),
        test=test,
        points=corners(target_pH_range=tuple(nominal.target_pH_range)),
        balance=lambda o, xp: raw(reagent_demand(o[2], curve, xp), xp),
        why={
            "cascade": "pH is the only reading",
            "decoupler": "one loop",
            "override": "no reachable limit",
        },
    )
