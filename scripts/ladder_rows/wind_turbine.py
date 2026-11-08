"""The reference ladder's row for ``wind_turbine`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Row,
    corners,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.energy.wind_turbine.env import balance_pitch

    spec = REGISTRY["wind_turbine"]
    p = spec.make_test_params()
    v = 0.5 * sum(p.v_mean_range)
    test = {"turbulence_std": 0.0, "v_mean_range": (v, v)}
    rpm = float(p.omega_rated_rpm)
    rated = rpm * 2.0 * np.pi / 60.0
    tau_max, power = float(p.torque_max), float(p.target_power_range[1])
    pitch = float(balance_pitch(rated, v, power / float(p.eta_gen), p))
    pitch_max = float(p.pitch_max)
    raw_pitch = 2 * pitch / pitch_max - 1

    def torque(o, xp):
        omega = o[0] * 2 * np.pi / 60
        tau = o[4] * 1e6 / (float(p.eta_gen) * float(p.N_gear) * omega)
        return 2 * xp.clip(tau, 0.0, tau_max) / tau_max - 1

    def balance(o, xp):
        return xp.stack([xp.asarray(raw_pitch, dtype=xp.float32), torque(o, xp)])

    def rest_at(q):
        v, P = float(q.v_mean_range[0]), float(q.target_power_range[1])
        pitch = float(balance_pitch(rated, v, P / float(q.eta_gen), q))
        held = 2 * pitch / pitch_max - 1

        def rest(o, xp):
            return xp.stack([xp.asarray(held, dtype=xp.float32), torque(o, xp)])

        return rest

    speed = Cv(
        "speed on pitch",
        0,
        lambda o, xp: o[0],
        lambda o, xp: xp.asarray(rpm, dtype=xp.float32),
        "omega",
        0.02,
        80,
        valve=False,
    )
    grid = Cv(
        "power on torque",
        1,
        lambda o, xp: o[3],
        lambda o, xp: o[4],
        "power",
        0.02,
        40,
        valve=False,
    )

    def published(s):
        return s.make_pid_controller()

    return Row(
        100,
        balance,
        (speed, grid),
        test=test,
        points=corners(
            v_mean_range=tuple(p.v_mean_range),
            target_power_range=tuple(p.target_power_range),
        ),
        rest_at=rest_at,
        balance=balance,
        published=published,
        why={
            "cascade": "the actuators are faster than a step",
            "decoupler": "the coupling moves with wind, 0.07 to 0.82 on the diagonal",
            "override": "the MVs are spent on the CVs",
        },
    )
