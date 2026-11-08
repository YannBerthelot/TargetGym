"""The reference ladder's row for ``compressor_surge`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

import jax
import numpy as np
from reference_ladder import (  # noqa: E402
    NOISE,
    SEEDS,
    Cv,
    Limit,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.compressor_surge import experts as X

    spec = REGISTRY["compressor_surge"]
    p = spec.make_test_params()
    line = 0.10
    env = spec.make_env()
    full = spec.make_test_params(sensor_noise_scale=NOISE, max_steps_in_episode=1200)
    starts = [env.reset_env(jax.random.PRNGKey(s), full)[1] for s in SEEDS]
    rests = [
        X.settled_point(
            float(s.setpoint_levels[-1]),
            float(s.demand_levels[-1]),
            {"surge_line": line},
            full,
        )
        for s in starts
    ]
    held = np.stack([X.raw_from_commands(r[2], r[4], full) for r in rests])
    reset = np.asarray(X.raw_from_commands(p.N_reset, p.x_reset, p))
    open_seeds = [i for i, r in enumerate(rests) if r[4] > 0.02]
    span = float(p.N_max - p.N_min)
    du = 0.05 * span * 2 / span
    pressure = Cv(
        "dp on speed",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[6],
        "",
        du,
        100,
        valve=False,
    )
    margin = Limit(
        "margin on recycle",
        lambda o, xp: X.surge_margin_from_obs(o, p),
        line,
        True,
        1,
        "",
        0.1,
        100,
        seeds=open_seeds,
    )
    published = lambda s: s.make_pid_controller()  # noqa: E731
    return Row(
        1200,
        lambda o, xp: xp.asarray(reset, dtype=xp.float32),
        (pressure,),
        limits=(margin,),
        start=lambda a, m: held,
        published=published,
        why={
            "feedforward": "the consumer's opening is unread and the map's speed needs it",
            "cascade": "the drive is the slowest element on the way to pressure",
            "decoupler": "the second MV serves a limit, not a target",
            "points": "each seed starts held at its own drawn setpoint and demand, "
            "inside the ranges, not at their ends: the start isn't a parameter to pin",
        },
    )
