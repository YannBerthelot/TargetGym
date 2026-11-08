"""The reference ladder's row for ``hvac`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import math

from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.hvac.env import total_heat_loss_coefficient

    spec = REGISTRY["hvac"]
    quiet = spec.make_test_params()
    design_T = float(quiet.setpoint_occupied)
    test = {
        "T_out_amplitude": 0.0,
        "T_out_noise_std": 0.0,
        "solar_peak": 0.0,
        "gain_occupied": float(quiet.gain_unoccupied),
        "setpoint_setback": design_T,
        "setpoint_occupied_range": (design_T, design_T),
    }
    tp = spec.make_test_params(**test)
    H, Q_max = float(total_heat_loss_coefficient(tp)), float(quiet.Q_heat_max)
    gains = float(quiet.gain_unoccupied * quiet.A_floor)
    solar = float(quiet.solar_peak * quiet.A_window * quiet.g_window)

    def load(o, xp):
        return H * (o[6] - o[1]) - gains - o[3] * solar

    def balance(o, xp):
        return xp.reshape(xp.clip(2 * load(o, xp) / Q_max - 1, -1.0, 1.0), (1,))

    def raw_at(T):
        return 2 * (H * (T - float(tp.T_out_mean)) - gains) / Q_max - 1

    def rest_at(q):
        raw = raw_at(float(q.setpoint_occupied_range[0]))
        return lambda o, xp: xp.full((1,), raw)

    ends = tuple(quiet.setpoint_occupied_range)
    points = tuple(
        {"setpoint_occupied_range": (T, T), "setpoint_setback": T} for T in ends
    )
    raw = raw_at(design_T)
    window = math.ceil((5 * float(tp.emitter_tau) + 450.0) / 900.0)
    air = Cv(
        "T_air on heat",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[6],
        "T_air",
        0.1,
        window,
        valve=False,
    )
    heat = Inner(
        0,
        "emitter on heat",
        lambda o, xp: o[2],
        lambda o, xp: 100.0 * load(o, xp) / Q_max,
        "Q_emitter",
        0.1,
        window,
        5.0,
        window,
        valve=False,
    )

    def published(s):
        return s.make_pid_controller()

    return Row(
        720,
        lambda o, xp: xp.full((1,), raw),
        (air,),
        test=test,
        points=points,
        rest_at=rest_at,
        balance=balance,
        cascades=(heat,),
        published=published,
        why={"decoupler": "one loop", "override": "the comfort band is not a trip"},
    )
