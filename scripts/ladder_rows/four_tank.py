"""The reference ladder's row for ``four_tank`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
    corners,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.pc_gym.four_tank.experts import (
        four_tank_decoupled_pi,
        steady_voltages,
    )

    p = REGISTRY["four_tank"].make_test_params()
    prm = four_tank_decoupled_pi(1.0, 1.0, 1.0, 1.0, np.eye(2), p).params
    v_max = float(p.v_max)
    per_volt = 2 / (v_max - float(p.v_min))

    def balance(o, xp):
        v = steady_voltages(o[4], o[5], prm, xp)
        return xp.stack([per_volt * v[0] - 1, per_volt * v[1] - 1])

    def upper(i, gamma, k, a):
        """The upper tank's level at rest on its pump, ``a sqrt(2 g h) = (1 - gamma)
        k v``."""

        def rest(o, xp):
            v = steady_voltages(o[4], o[5], prm, xp)[i]
            return ((1 - gamma) * k * v / a) ** 2 / (2 * p.g)

        return rest

    du = 0.3 * per_volt
    h1 = Cv("h1 on v2", 1, lambda o, xp: o[0], lambda o, xp: o[4], "h1", du, 1500)
    h1 = dataclasses.replace(h1, valve=False)
    h2 = Cv("h2 on v1", 0, lambda o, xp: o[1], lambda o, xp: o[5], "h1", du, 1500)
    h2 = dataclasses.replace(h2, valve=False)
    return Row(
        1200,
        balance,
        (h1, h2),
        test={"load_scale": 0.0},
        points=corners(
            target_h1_range=tuple(p.target_h1_range),
            target_h2_range=tuple(p.target_h2_range),
        ),
        balance=balance,
        cascades=(
            Inner(
                0,
                "h3 on v2",
                lambda o, xp: o[2],
                upper(1, p.gamma2, p.k2, p.a3),
                "h1",
                du,
                600,
                0.005,
                1500,
                valve=False,
            ),
            Inner(
                1,
                "h4 on v1",
                lambda o, xp: o[3],
                upper(0, p.gamma1, p.k1, p.a4),
                "h1",
                du,
                600,
                0.005,
                1500,
                valve=False,
            ),
        ),
        decouple=True,
        why={
            "override": "low trip has 30 mm of margin at worst load",
            "points": "only the target corners: the lower tanks' loads are steps the "
            "episode switches during a run, and no parameter holds one, so the tests "
            "run at no load",
        },
    )
