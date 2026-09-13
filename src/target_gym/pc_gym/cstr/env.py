from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward


@struct.dataclass
class CSTRParams(EnvParams):
    q: float = 100.0
    V: float = 100.0
    rho: float = 1000.0
    C: float = 0.239
    deltaHr: float = -5e4
    EA_over_R: float = 8750.0
    k0: float = 7.2e10
    UA: float = 5e4
    Ti: float = 350.0
    Caf: float = 1.0

    T_c_max: float = 302.0
    T_c_min: float = 295.0

    T_max: float = 350.0
    T_min: float = 300.0
    C_a_min: float = 0.7
    # Composition analyser resolution. Below this the reward would be paying
    # for differences an online GC cannot report.
    precision_floor: float = 1e-4  # mol/L
    C_a_max: float = 1.0

    target_CA_range: Tuple[float, float] = (0.84, 0.91)
    initial_CA_range: Tuple[float, float] = (0.8, 0.85)
    initial_T: float = 330.0
    delta_t: float = 0.25
    #: ``delta_t`` is in minutes (the PC-gym model's unit); seconds per unit.
    time_unit_seconds: float = 60.0
    max_steps_in_episode: int = 100

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # No disturbance and a fixed target: the shipped MPC holds the
    # concentration to 1e-6 mol/L after settling (`scripts/measure_hold.py`),
    # so the achievable floor is effectively zero and the normalisation uses a
    # documented minimum, the online analyser's resolution. The span costs
    # (0.3 / 1e-4)^2 = 9e6 per step; a runaway twice that.
    reward_version: int = 2
    e_floor: float = 1e-4  # mol/L, composition analyser resolution
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    failure_cost: float = 1.8e7
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    rho_floor_tracking: float = 1.0
    rho_floor: float = 1.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


@struct.dataclass
class CSTRState(EnvState):
    C_a: float
    T: float
    target_CA: float

    # For rendering
    T_c: float


def compute_velocity(position, action, params: CSTRParams):
    T_c = action
    C_a, T = position

    rA = params.k0 * jnp.exp(-params.EA_over_R / T) * C_a

    velocity_C_a = params.q / params.V * (params.Caf - C_a) - rA
    velocity_T = (
        params.q / params.V * (params.Ti - T)
        + ((-params.deltaHr) * rA) * (1 / (params.rho * params.C))
        + params.UA * (T_c - T) * (1 / (params.rho * params.C * params.V))
    )

    return jnp.array([velocity_C_a, velocity_T]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    T_c_raw: float,
    state: CSTRState,
    params: CSTRParams,
    integration_method: str = "rk4_1",
):
    """
    T_c_raw : Cooling temperature raw input
    """
    dt = params.delta_t
    T_c = convert_raw_action_to_range(
        T_c_raw, min_action=params.T_c_min, max_action=params.T_c_max
    )
    _compute_velocity = partial(compute_velocity, action=T_c, params=params)

    (C_a, T), metrics = integrate_dynamics(
        positions=jnp.array([state.C_a, state.T]),
        delta_t=dt,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    return (
        state.replace(C_a=C_a, T=T, T_c=T_c, time=state.time + 1),
        metrics,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(
    state: CSTRState,
    params: CSTRParams,
):
    return jnp.array([state.C_a, state.T, state.target_CA])


def check_is_terminal(state: CSTRState, params: CSTRParams, xp=jnp):
    terminated_1 = jnp.logical_or(state.T <= params.T_min, state.T >= params.T_max)
    terminated_2 = jnp.logical_or(
        state.C_a <= params.C_a_min, state.C_a >= params.C_a_max
    )
    terminated = jnp.logical_or(terminated_1, terminated_2)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: CSTRState, params: CSTRParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    return {
        "tracking": R.tracking_cost(
            state.target_CA - state.C_a,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
        "failure": R.failure_cost(terminated, params.failure_cost, xp),
    }


def compute_reward_v1(state: CSTRState, params: CSTRParams, xp=jnp):
    # Log-scaled: every halving of the concentration error is worth the same,
    # down to what the analyser can resolve. The previous form divided the error
    # by the whole 0.3 mol/L envelope and squared it, which is nearly flat over
    # any error a working controller produces -- checklist check 1.
    return log_scaled_reward(
        xp.abs(state.target_CA - state.C_a),
        params.precision_floor,
        params.C_a_max - params.C_a_min,
        xp,
    )


def compute_reward(state: CSTRState, params: CSTRParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
