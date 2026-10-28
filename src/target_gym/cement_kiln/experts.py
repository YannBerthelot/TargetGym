"""Reference controller (the MPC slot) for the cement kiln task.

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp
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
# Cascade PID as one pure step (host NumPy and jit/vmap alike)
# ---------------------------------------------------------------------------


@struct.dataclass
class CementKilnPIDParams:
    """Gains and plant constants of the cascade controller.

    Every field is a Python float, so NumPy treats it as weakly typed and the
    host arithmetic stays in the observation's dtype, as it did when
    the baseline was recorded.
    """

    Kp_outer: float
    Ki_outer: float
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
    # The setpoint trim is bounded so the outer loop cannot drive the kiln to
    # a temperature that would form rings or let it go cold.
    sp_trim_max: float = 120.0


@struct.dataclass
class CementKilnPIDState:
    """Outer (free lime) and inner (burning zone) integrators."""

    outer_int: float = 0.0
    inner_int: float = 0.0


def cement_kiln_pid_step(params, state, obs, xp=jnp):
    """One step of the cement kiln's cascaded PID, written against ``xp``.

    Keep the arithmetic order: under ``numpy`` it reproduces the scored
    baseline bit for bit.

    obs: [lime_pct, T_bz, T_exhaust, T_back_end, feed, fuel_pct, speed_pct,
          target_lime_pct]
    """
    p = params
    lime = obs[..., 0] / 100.0
    T_bz = obs[..., 1]
    feed = obs[..., 4]
    target_lime = obs[..., 7] / 100.0

    # --- Outer: free lime sets the burning-zone target ---
    err_o = lime - target_lime
    outer_int = state.outer_int + err_o * p.dt
    trim = p.Kp_outer * err_o + p.Ki_outer * outer_int
    trim_clipped = xp.clip(trim, -p.sp_trim_max, p.sp_trim_max)
    outer_int = xp.where(trim != trim_clipped, outer_int - err_o * p.dt, outer_int)
    T_bz_sp = p.T_bz_base + trim_clipped

    # --- Inner: fuel holds the burning-zone temperature ---
    err_i = T_bz_sp - T_bz
    inner_int = state.inner_int + err_i * p.dt
    fuel = p.fuel_nominal + p.Kp_inner * err_i + p.Ki_inner * inner_int
    fuel_clipped = xp.clip(fuel, p.fuel_min, p.fuel_max)
    inner_int = xp.where(fuel != fuel_clipped, inner_int - err_i * p.dt, inner_int)
    fuel_raw = 2.0 * (fuel_clipped - p.fuel_min) / (p.fuel_max - p.fuel_min) - 1.0

    # --- Speed follows feed, holding bed depth constant ---
    rpm = p.rpm_nominal * feed / p.feed_nominal
    rpm = xp.clip(rpm, p.rpm_min, p.rpm_max)
    rpm_raw = 2.0 * (rpm - p.rpm_min) / (p.rpm_max - p.rpm_min) - 1.0

    new_state = CementKilnPIDState(outer_int=outer_int, inner_int=inner_int)
    return xp.stack([fuel_raw, rpm_raw], axis=-1), new_state


def make_cement_kiln_pid_controller() -> Controller:
    """The cascade controller as a :class:`Controller`.

    Reads the same tuned gains, with the same defaults, as
    ``make_cement_kiln_stateful_pid``.
    """
    from target_gym.cement_kiln.env import CementKilnParams

    pr = CementKilnParams()
    _p = load_gains().get("cement_kiln", {})
    params = CementKilnPIDParams(
        Kp_outer=float(_p.get("Kp_outer", 12000.0)),
        Ki_outer=float(_p.get("Ki_outer", 6.0)),
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
    return Controller(params, CementKilnPIDState(), cement_kiln_pid_step)
