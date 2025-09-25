from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range


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
    C_a_max: float = 1.0

    target_CA_range: Tuple[float, float] = (0.85, 0.85)
    initial_CA_range: Tuple[float, float] = (0.8, 0.85)
    initial_T: float = 330.0
    delta_t: float = 1e-3
    max_steps_in_episode: int = 1_000


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
        state.replace(C_a=C_a, T=T, T_c=T_c_raw, time=state.time + 1),
        metrics,
    )


@partial(jax.jit, static_argnames=["params"])
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


def compute_reward(state: CSTRState, params: CSTRParams, xp=jnp):
    max_C_a_diff = params.C_a_max - params.C_a_min
    reward = ((max_C_a_diff - xp.abs(state.target_CA - state.C_a)) / max_C_a_diff) ** 2

    return reward
