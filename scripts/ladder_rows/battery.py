"""The reference ladder's row for ``battery`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Limit,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    p = REGISTRY["battery"].make_test_params()
    power = float(p.power_max)
    meter = Cv(
        "P on inverter",
        0,
        lambda o, xp: o[3],
        lambda o, xp: o[4],
        "power",
        0.1,
        20,
        valve=False,
    )

    def rest(o, xp):
        return xp.reshape(o[4], (1,))

    def balance(o, xp):
        return xp.reshape((o[4] * 1e6 + float(p.aux_load)) / power, (1,))

    # Discharging lowers the charge: d soc / dt = -P / E, per raw of the inverter.
    slope = -power / float(p.energy_nominal)
    soc = dict(read=lambda o, xp: o[0], mv=0, slope=slope, theta=float(p.delta_t))
    low, high = 0.10, 0.90  # 5 % inside the pack's trips, MESA-1's operators' margin
    return Row(
        120,
        rest,
        (meter,),
        test={"dispatch_noise_std": 0.0},
        balance=balance,
        limits=(
            Limit("charge floor", limit=low, floor=True, **soc),
            Limit("charge ceiling", limit=high, floor=False, **soc),
        ),
        why={
            "cascade": "meter power is algebraic on the inverter",
            "decoupler": "one loop",
            "points": "meter power is the inverter's, scaled: one gain at any dispatch",
        },
    )
