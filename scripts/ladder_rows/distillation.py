"""The reference ladder's row for ``distillation`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    p = REGISTRY["distillation"].make_test_params()
    L0, V0 = 2.70629, 3.20629  # the column's rest, from its tray equations
    per_L, per_V = 2 / float(p.L_max - p.L_min), 2 / float(p.V_max - p.V_min)
    raw = np.array(
        [per_L * (L0 - float(p.L_min)) - 1, per_V * (V0 - float(p.V_min)) - 1]
    )

    def rest(o, xp):
        return xp.asarray(raw, dtype=xp.float32)

    yD = Cv("yD on L", 0, lambda o, xp: o[0], lambda o, xp: o[4], "x", 0.05 * 2, 600)
    xB = Cv("xB on V", 1, lambda o, xp: o[1], lambda o, xp: o[5], "x", 0.05 * 2, 600)
    tray = Inner(
        1,
        "T_15 on V",
        lambda o, xp: o[6],
        None,
        "",
        0.005 * per_V,
        20,
        0.002,
        600,
        model="integrating_lag",
    )
    return Row(
        600,
        rest,
        (yD, xB),
        test={"zF_noise_std": 0.0},
        cascades=(tray,),
        decouple=True,
        why={
            "feedforward": "no closed-form balance: L and V need the tray model, and "
            "no feed reading exists",
            "override": "the trip reads through a 5 min analyser",
            "points": "the rest at another purity needs the tray model solved, which "
            "the row doesn't have: the column is tested at its one rest",
        },
    )
