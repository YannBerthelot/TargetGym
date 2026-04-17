from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range


@struct.dataclass
class FourTankParams(EnvParams):
    # Gravitational acceleration (m/s²)
    g: float = 9.81
    # Pump split ratios (fraction of pump flow to lower tanks)
    gamma1: float = 0.2
    gamma2: float = 0.2
    # Pump gain constants (m³/s per Volt)
    k1: float = 0.00085
    k2: float = 0.00095
    # Outlet orifice areas (m²)
    a1: float = 0.0035
    a2: float = 0.003
    a3: float = 0.002
    a4: float = 0.0025
    # Tank cross-sectional areas (m²)
    A1: float = 1.0
    A2: float = 1.0
    A3: float = 1.0
    A4: float = 1.0

    # Pump voltage bounds (V)
    v_min: float = 0.0
    v_max: float = 10.0

    # Level bounds (m)
    h_min: float = 0.05
    h_max: float = 1.5

    # Target level ranges for tanks 1 and 2
    target_h1_range: Tuple[float, float] = (0.5, 1.0)
    target_h2_range: Tuple[float, float] = (0.5, 1.0)

    # Initial level ranges for all four tanks
    initial_h1_range: Tuple[float, float] = (0.2, 0.4)
    initial_h2_range: Tuple[float, float] = (0.2, 0.4)
    initial_h3_range: Tuple[float, float] = (0.1, 0.2)
    initial_h4_range: Tuple[float, float] = (0.1, 0.2)

    delta_t: float = 1.0
    max_steps_in_episode: int = 500


@struct.dataclass
class FourTankState(EnvState):
    h1: float  # Level of tank 1 (controlled)
    h2: float  # Level of tank 2 (controlled)
    h3: float  # Level of tank 3 (upper, feeds tank 1)
    h4: float  # Level of tank 4 (upper, feeds tank 2)
    target_h1: float
    target_h2: float

    # For rendering
    v1: float
    v2: float


def compute_velocity(position, action, params: FourTankParams):
    h1, h2, h3, h4 = position[0], position[1], position[2], position[3]
    v1, v2 = action[0], action[1]

    sqrt_h1 = jnp.sqrt(jnp.maximum(h1, 0.0))
    sqrt_h2 = jnp.sqrt(jnp.maximum(h2, 0.0))
    sqrt_h3 = jnp.sqrt(jnp.maximum(h3, 0.0))
    sqrt_h4 = jnp.sqrt(jnp.maximum(h4, 0.0))

    dh1dt = (
        -(params.a1 / params.A1) * jnp.sqrt(2 * params.g) * sqrt_h1
        + (params.a3 / params.A1) * jnp.sqrt(2 * params.g) * sqrt_h3
        + (params.gamma1 * params.k1 / params.A1) * v1
    )
    dh2dt = (
        -(params.a2 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h2
        + (params.a4 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h4
        + (params.gamma2 * params.k2 / params.A2) * v2
    )
    dh3dt = (
        -(params.a3 / params.A3) * jnp.sqrt(2 * params.g) * sqrt_h3
        + ((1 - params.gamma2) * params.k2 / params.A3) * v2
    )
    dh4dt = (
        -(params.a4 / params.A4) * jnp.sqrt(2 * params.g) * sqrt_h4
        + ((1 - params.gamma1) * params.k1 / params.A4) * v1
    )

    return jnp.array([dh1dt, dh2dt, dh3dt, dh4dt]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: FourTankState,
    params: FourTankParams,
    integration_method: str = "rk4_1",
):
    v1 = convert_raw_action_to_range(action_raw[0], params.v_min, params.v_max)
    v2 = convert_raw_action_to_range(action_raw[1], params.v_min, params.v_max)
    action = jnp.array([v1, v2])

    _compute_velocity = partial(compute_velocity, action=action, params=params)
    (h1, h2, h3, h4), metrics = integrate_dynamics(
        positions=jnp.array([state.h1, state.h2, state.h3, state.h4]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    return (
        state.replace(h1=h1, h2=h2, h3=h3, h4=h4, v1=v1, v2=v2, time=state.time + 1),
        metrics,
    )


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: FourTankState, params: FourTankParams):
    return jnp.array(
        [state.h1, state.h2, state.h3, state.h4, state.target_h1, state.target_h2]
    )


def check_is_terminal(state: FourTankState, params: FourTankParams, xp=jnp):
    h_vals = jnp.array([state.h1, state.h2, state.h3, state.h4])
    terminated = jnp.any(jnp.logical_or(h_vals <= params.h_min, h_vals >= params.h_max))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: FourTankState, params: FourTankParams, xp=jnp):
    max_diff = params.h_max - params.h_min
    r1 = ((max_diff - xp.abs(state.target_h1 - state.h1)) / max_diff) ** 2
    r2 = ((max_diff - xp.abs(state.target_h2 - state.h2)) / max_diff) ** 2
    return (r1 + r2) / 2.0
