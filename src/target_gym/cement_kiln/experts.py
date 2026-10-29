"""Reference controller (the MPC slot) for the cement kiln task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.mpc import (
    SamplingMPC,
    _is_v1,
    _v2_objective,
)
from target_gym.experts.pid import Controller, load_gains


def _cement_kiln_objective(state, params):
    """Quadratic in the free-lime error, sharing the reward's minimiser.

    The environment's reward clips flat once free lime is more than
    ``lime_band`` from target -- exactly the situation the controller is called
    on to fix -- so a quadratic that stays informative far from target is what
    the optimiser needs.
    """
    err = (state.lime[-1] - state.target_lime) / params.lime_band
    fuel = (state.fuel - params.fuel_min) / (params.fuel_max - params.fuel_min)
    return -(err**2 + 0.02 * fuel)


def make_cement_kiln_mpc(
    env,
    params,
    horizon: int = 40,
    n_samples: int = 96,
    n_elite: int = 12,
    n_iter: int = 32,
    init_std: float = 0.1,
    return_best: bool = True,
    **kwargs,
):
    """Sampling (CEM) MPC for the rotary kiln.

    Gradient-free by necessity, not preference -- see ``SamplingMPC`` for the
    measured reason.

    A 40-step horizon is 20 minutes at dt = 30 s, most of the ~25 minute
    transport delay. That is the point: a controller whose horizon is shorter
    than the delay is choosing fuel whose consequences it cannot see.

    Each solve starts from the shifted previous plan, so it searches near it
    (``init_std`` 0.1) and runs enough iterations for the smoothed CEM spread
    to contract; it returns the best sequence it evaluated. Measured on the
    protocol seeds, planning on the mean
    feed: the earlier budget (4 iterations from a 0.5 spread, smoothed mean)
    scored 1.26; 384 samples and 8 iterations 0.264; this 96 x 32 setting
    0.0705 at the same cost per step, and 64 iterations 0.0699. What is left
    is mostly fuel above the hold rate, which the raw-meal swings drive.
    """
    objective = _cement_kiln_objective
    if not _is_v1(params):
        from target_gym.cement_kiln.env import compute_reward

        objective = _v2_objective(compute_reward)
    return SamplingMPC(
        env,
        params,
        objective=objective,
        action_dim=2,
        action_lb=-1.0,
        action_ub=1.0,
        horizon=horizon,
        n_samples=n_samples,
        n_elite=n_elite,
        n_iter=n_iter,
        init_std=init_std,
        return_best=return_best,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# The expert: a fuzzy supervisor on free lime over the burning-zone loop
# ---------------------------------------------------------------------------

#: Swain and Subudhi (1996), Appendix 2: the burning-zone setpoint move, in
#: steps of ``u``, for free lime LOW / OK / HIGH (rows) and the kiln-speed
#: trend INCR / OK / DECR (columns). L/M/S DECR are -3/-2/-1, ZERO 0 and
#: S/M/L INCR +1/+2/+3.
RULES = np.array(
    [
        [1.0, -2.0, -3.0],  # LOW:  SINCR MDECR LDECR
        [2.0, 0.0, -2.0],  # OK:   MINCR ZERO  MDECR
        [3.0, 2.0, -1.0],  # HIGH: LINCR MINCR SDECR
    ]
)


@struct.dataclass
class CementKilnFuzzyParams:
    """The rule base's scaling, the burning-zone loop and the plant's limits.

    ``lime_halfwidth`` (free-lime fraction) and ``step`` (K per 30 s) are the
    two scaling factors a fuzzy supervisor is commissioned with, tuned on the
    training seeds; ``speed_halfwidth`` is the speed change, in % of its range,
    that one feed standard deviation causes. The rule block is published; the
    triangles, singletons, min and centroid are ours, as the paper gives no
    breakpoints.
    """

    lime_halfwidth: float
    step: float
    speed_halfwidth: float
    Kp_inner: float
    Ki_inner: float
    T_bz_base: float
    fuel_min: float
    fuel_max: float
    fuel_nominal: float
    rpm_min: float
    rpm_max: float
    rpm_nominal: float
    feed_nominal: float
    dt: float = 30.0
    # The setpoint is bounded so the supervisor cannot drive the kiln to a
    # temperature that would form rings or let it go cold.
    sp_trim_max: float = 120.0
    trend_steps: int = struct.field(pytree_node=False, default=10)


@struct.dataclass
class CementKilnFuzzyState:
    """The setpoint trim, the burning-zone integral, and the speed over the
    last ``trend_steps`` steps (oldest first), filled on the first reading."""

    sp_trim: float = 0.0
    inner_int: float = 0.0
    speeds: tuple = (0.0,) * 10
    started: float = 0.0


def _three_sets(x, halfwidth, xp):
    """Memberships of ``x`` in (low, mid, high): triangles of half-width
    ``halfwidth`` about 0, the outer two saturating beyond it."""
    low = xp.clip(-x / halfwidth, 0.0, 1.0)
    high = xp.clip(x / halfwidth, 0.0, 1.0)
    return xp.stack([low, 1.0 - low - high, high], axis=-1)


def cement_kiln_fuzzy_step(params, state, obs, xp=jnp):
    """One 30 s step: the rule base moves the burning-zone setpoint, fuel
    holds it, kiln speed follows feed.

    obs: [lime_pct, T_bz, T_exhaust, T_back_end, feed, fuel_pct, speed_pct,
          target_lime_pct]
    """
    p = params
    lime_err = (obs[..., 0] - obs[..., 7]) / 100.0
    T_bz, feed, speed = obs[..., 1], obs[..., 4], obs[..., 6]

    # Speed trend over the last five minutes (no trend on the first reading).
    speeds = xp.asarray(state.speeds)
    speeds = xp.where(state.started > 0.5, speeds, speed[..., None] + 0.0 * speeds)
    trend = speed - speeds[..., 0]
    speeds = xp.concatenate([speeds[..., 1:], speed[..., None]], axis=-1)

    # Mamdani: AND as min, the centroid of the singleton outputs.
    mu_lime = _three_sets(lime_err, p.lime_halfwidth, xp)
    # Columns are INCR / OK / DECR: reverse the (low, mid, high) order.
    mu_speed = _three_sets(trend, p.speed_halfwidth, xp)[..., ::-1]
    w = xp.minimum(mu_lime[..., :, None], mu_speed[..., None, :])
    move = xp.sum(w * RULES, axis=(-2, -1)) / xp.maximum(
        xp.sum(w, axis=(-2, -1)), 1e-12
    )
    sp_trim = xp.clip(state.sp_trim + p.step * move, -p.sp_trim_max, p.sp_trim_max)
    T_bz_sp = p.T_bz_base + sp_trim

    # Fuel holds the burning-zone temperature (PI, integral frozen at the stops).
    err_i = T_bz_sp - T_bz
    inner_int = state.inner_int + err_i * p.dt
    fuel = p.fuel_nominal + p.Kp_inner * err_i + p.Ki_inner * inner_int
    fuel_clipped = xp.clip(fuel, p.fuel_min, p.fuel_max)
    inner_int = xp.where(fuel != fuel_clipped, inner_int - err_i * p.dt, inner_int)
    fuel_raw = 2.0 * (fuel_clipped - p.fuel_min) / (p.fuel_max - p.fuel_min) - 1.0

    # Kiln speed follows feed, holding bed depth constant.
    rpm = xp.clip(p.rpm_nominal * feed / p.feed_nominal, p.rpm_min, p.rpm_max)
    rpm_raw = 2.0 * (rpm - p.rpm_min) / (p.rpm_max - p.rpm_min) - 1.0

    new_state = CementKilnFuzzyState(
        sp_trim=sp_trim, inner_int=inner_int, speeds=speeds, started=1.0 + 0.0 * T_bz
    )
    return xp.stack([fuel_raw, rpm_raw], axis=-1), new_state


def make_cement_kiln_pid_controller() -> Controller:
    """The expert: the Swain and Subudhi rule base over the burning-zone loop,
    as a :class:`Controller`. Its scaling and loop gains are stored in
    ``pid_gains.json`` by ``scripts/tune_industrial_experts.py``."""
    from target_gym.cement_kiln.env import CementKilnParams

    pr = CementKilnParams()
    _p = load_gains().get("cement_kiln", {})
    # One feed standard deviation moves the speed by this much of its range.
    speed_halfwidth = (
        100.0
        * pr.rpm_nominal
        * pr.feed_noise_std
        / pr.raw_meal_nominal
        / (pr.rpm_max - pr.rpm_min)
    )
    params = CementKilnFuzzyParams(
        lime_halfwidth=float(_p.get("lime_halfwidth", 0.002)),
        step=float(_p.get("step", 0.5)),
        speed_halfwidth=float(speed_halfwidth),
        Kp_inner=float(_p.get("Kp_inner", 0.025)),
        Ki_inner=float(_p.get("Ki_inner", 5.0e-6)),
        T_bz_base=float(_p.get("T_bz_base", 1765.0)),
        fuel_min=pr.fuel_min,
        fuel_max=pr.fuel_max,
        fuel_nominal=pr.fuel_nominal,
        rpm_min=pr.rpm_min,
        rpm_max=pr.rpm_max,
        rpm_nominal=pr.rpm_nominal,
        feed_nominal=pr.raw_meal_nominal,
        dt=pr.delta_t,
    )
    state = CementKilnFuzzyState(speeds=np.zeros(params.trend_steps, np.float32))
    return Controller(params, state, cement_kiln_fuzzy_step)
