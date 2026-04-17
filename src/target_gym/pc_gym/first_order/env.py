from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range


@struct.dataclass
class FirstOrderParams(EnvParams):
    K: float = 1.0
    tau: float = 0.5
    u_min: float = -2.0
    u_max: float = 2.0
    x_min: float = -3.0
    x_max: float = 3.0

    target_x_range: Tuple[float, float] = (0.5, 1.5)
    initial_x_range: Tuple[float, float] = (-0.5, 0.5)
    delta_t: float = 0.05
    max_steps_in_episode: int = 200


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


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: FirstOrderState, params: FirstOrderParams):
    return jnp.array([state.x, state.target_x])


def check_is_terminal(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    terminated = jnp.logical_or(state.x <= params.x_min, state.x >= params.x_max)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: FirstOrderState, params: FirstOrderParams, xp=jnp):
    max_diff = params.x_max - params.x_min
    reward = ((max_diff - xp.abs(state.target_x - state.x)) / max_diff) ** 2
    return reward
