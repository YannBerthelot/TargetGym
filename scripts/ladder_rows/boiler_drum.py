"""The reference ladder's row for ``boiler_drum`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Row,
)

from target_gym.identification import Loop
from target_gym.registry import REGISTRY


def row() -> Row:
    p = REGISTRY["boiler_drum"].make_test_params()
    q_max, Q_max = float(p.q_feed_max), float(p.Q_max)
    per_kg = 2.0 / q_max
    level_gain = q_max / (2.0 * float(p.level_trip))
    h = 2.75e6 - float(p.h_feedwater)

    def balance(o, xp):
        return xp.stack([2 * o[2] * h / Q_max - 1, per_kg * o[2] - 1])

    def rest(o, xp):
        feed = xp.clip(o[2] + level_gain * (o[5] - o[0]), 0.0, q_max)
        return xp.stack([2 * o[2] * h / Q_max - 1, per_kg * feed - 1])

    pressure = Cv(
        "pressure on firing",
        0,
        lambda o, xp: o[1],
        lambda o, xp: o[6],
        "pressure",
        0.02,
        120,
        room=1.0,
    )
    level = Cv(
        "level on feed",
        1,
        lambda o, xp: o[0],
        lambda o, xp: o[5],
        "level",
        2.0 * per_kg,
        120,
        model="integrating_lag",
    )
    return Row(
        400,
        rest,
        (pressure, level),
        test={"q_steam_noise_std": 0.0},
        # The episode draws its steam load from a normal: its ends taken at 2 sigma,
        # 95 % of draws (ours). The rest follows the steam reading.
        points=tuple(
            {"q_steam_nominal": float(p.q_steam_nominal) + d}
            for d in (-2 * float(p.q_steam_noise_std), 2 * float(p.q_steam_noise_std))
        ),
        balance=balance,
        holds=(Loop(1, 1, level_gain * per_kg, 0),),
        why={
            "cascade": "riser and inventory are hidden; only actuator positions read",
            "decoupler": "RGA is 1.03 on the diagonal",
            "override": "no limit a select could hold: the MVs are spent on the CVs",
        },
    )
