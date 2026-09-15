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
class FirstOrderParams(EnvParams):
    K: float = 1.0
    tau: float = 0.5
    u_min: float = -2.0
    u_max: float = 2.0
    x_min: float = -3.0
    x_max: float = 3.0

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # No disturbance and a fixed target: the achievable hold error is zero
    # (the shipped MPC holds 0.0 after settling, `scripts/measure_hold.py`), so
    # the floor-normalisation uses a documented minimum instead -- a thousandth
    # of the span, finer than any sensor this dimensionless plant stands in
    # for. An error of one such unit costs 1 per step; the span costs
    # (6 / 6e-3)^2 = 1e6, and a failure twice that.
    reward_version: int = 2
    e_floor: float = 6e-3
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    failure_cost: float = 2.0e6
    #: Restart time priced into a trip (``reward.trip_cost``; 5 s at 0.05 s steps; a generic loop's reset, provisional).
    restart_steps: int = 100
    #: The NEA reference. Zero: the plant is deterministic, so exact hold is
    #: achievable and ``e_floor`` is a scale, not a floor (a reference of 1,
    #: the cost at the resolution, put the MPC above the reference and NEA > 1).
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True
    #: Version-1 reward only: the log-scaling's resolution floor.
    precision_floor: float = 6e-3

    target_x_range: Tuple[float, float] = (0.5, 1.5)
    initial_x_range: Tuple[float, float] = (-0.5, 0.5)
    delta_t: float = 0.05
    max_steps_in_episode: int = 100


@struct.dataclass
class FirstOrderState(EnvState):
    x: float
    target_x: float

    # For rendering
    u: float


def compute_velocity(position, action, params: FirstOrderParams):
    x = position[0]
    u = action
    dxdt = (params.K * u - x) / params.tau
    return jnp.array([dxdt]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    u_raw: float,
    state: FirstOrderState,
    params: FirstOrderParams,
    integration_method: str = "rk4_1",
):
    u = convert_raw_action_to_range(u_raw, params.u_min, params.u_max)
    _compute_velocity = partial(compute_velocity, action=u, params=params)
    (x,), metrics = integrate_dynamics(
        positions=jnp.array([state.x]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    return state.replace(x=x, u=u, time=state.time + 1), metrics


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: FirstOrderState, params: FirstOrderParams):
    return jnp.array([state.x, state.target_x])


def check_is_terminal(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    terminated = jnp.logical_or(state.x <= params.x_min, state.x >= params.x_max)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_x - state.x,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
    }
    return R.with_trip(terms, terminated, R.trip_cost(params), xp)


def compute_reward_v1(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    """Version-1 reward: log-scaled tracking, capped at 1."""
    return log_scaled_reward(
        xp.abs(state.target_x - state.x),
        params.precision_floor,
        params.x_max - params.x_min,
        xp,
    )


def compute_reward(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
