"""Reference controller (the MPC slot) for the four-tank task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, ShrinkingHorizonNLP, SamplingMPC and the v2 objective helpers)
stays in ``target_gym.experts.mpc``, which is in every task's baseline
fingerprint.
"""

import numpy as np

from target_gym.experts.mpc import (
    ShrinkingHorizonNLP,
    casadi_step_map,
    protocol_burn_in,
)

try:
    import casadi
except ImportError:  # ShrinkingHorizonNLP refuses to build without it
    pass


# ---------------------------------------------------------------------------
# FourTank
# ---------------------------------------------------------------------------

#: Typical level of each tank, m, which scales the NLP's state variables.
_X_SCALE = (0.25, 0.25, 0.25, 0.25)
#: How far inside the trip limits ``h_min`` and ``h_max`` the plan keeps every
#: level, m.
_LEVEL_MARGIN = 0.005


def _safe_sqrt(h):
    """``env._safe_sqrt`` in CasADi: ``sqrt(h)`` above zero, 0 at or below,
    with the inner guard that keeps the untaken branch's derivative finite."""
    positive = h > 0.0
    return casadi.if_else(positive, casadi.sqrt(casadi.if_else(positive, h, 1.0)), 0.0)


def four_tank_step_map(env, params):
    """The four-tank's discrete step, ``F(x, u) -> x_next``, as a CasADi
    Function.

    ``x = [h1, h2, h3, h4]`` and ``u`` the two raw pump commands in
    ``[-1, 1]``. The right-hand side is ``env.compute_velocity`` term for
    term, and the integration is the environment's own
    (``env.integration_method``, ``rk4_1`` as registered), so on the action
    box this is ``step_env`` in float64. The env clips the commands to the
    box first; the NLP bounds them there, so the clip is left out to keep the
    map smooth at the bounds.
    """
    p = params
    g2 = np.sqrt(2 * p.g)

    def velocity(x, u):
        h1, h2, h3, h4 = x[0], x[1], x[2], x[3]
        v1 = p.v_min + 0.5 * (u[0] + 1.0) * (p.v_max - p.v_min)
        v2 = p.v_min + 0.5 * (u[1] + 1.0) * (p.v_max - p.v_min)
        return casadi.vertcat(
            -(p.a1 / p.A1) * g2 * _safe_sqrt(h1)
            + (p.a3 / p.A1) * g2 * _safe_sqrt(h3)
            + (p.gamma1 * p.k1 / p.A1) * v1,
            -(p.a2 / p.A2) * g2 * _safe_sqrt(h2)
            + (p.a4 / p.A2) * g2 * _safe_sqrt(h4)
            + (p.gamma2 * p.k2 / p.A2) * v2,
            -(p.a3 / p.A3) * g2 * _safe_sqrt(h3) + ((1 - p.gamma2) * p.k2 / p.A3) * v2,
            -(p.a4 / p.A4) * g2 * _safe_sqrt(h4) + ((1 - p.gamma1) * p.k1 / p.A4) * v1,
        )

    method = getattr(env, "integration_method", "rk4_1")
    return casadi_step_map(velocity, 4, 2, float(p.delta_t), method)


def make_four_tank_mpc(
    env,
    params,
    window_start: int | None = None,
    resolve_every: int = 25,
    pre_weight: float = 1e-3,
):
    """The four-tank's oracle: a shrinking-horizon NLP over the rest of the
    episode.

    It re-solves every 25 steps on the env's own step
    (``four_tank_step_map``), with both lower tanks' tracking cost in floor
    units and every level held 5 mm inside the trip limits. ``window_start``
    is where the protocol starts scoring, read by default from its burn-in
    (``protocol_burn_in``: 250 steps of the registered 500), and the steps
    before it weigh ``pre_weight``, so the window weighs 1000 times more.
    That weighting is what matters here: the plant is non-minimum-phase
    (``gamma1 + gamma2`` = 0.4), and the exact optimum of an unweighted
    all-step cost leaves a tail in the scored window larger than the old
    oracle's.

    It replaced a do-mpc controller on a 10-step horizon of 20 s steps, whose
    stage cost charged every step alike (oracle audit, 2026-10). That one
    gave 0.0074 per step on the protocol seeds and this one 3.1e-8, with zero
    trips (``tests/pc_gym/four_tank``).
    """
    if window_start is None:
        window_start = protocol_burn_in("four_tank", params)
    return ShrinkingHorizonNLP(
        env,
        params,
        step_map=four_tank_step_map(env, params),
        state_fields=("h1", "h2", "h3", "h4"),
        target_fields=("target_h1", "target_h2"),
        tracked=(0, 1),
        window_start=window_start,
        x_scale=_X_SCALE,
        x_lb=np.full(4, float(params.h_min) + _LEVEL_MARGIN),
        x_ub=np.full(4, float(params.h_max) - _LEVEL_MARGIN),
        pre_weight=pre_weight,
        resolve_every=resolve_every,
    )
