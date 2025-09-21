from typing import Callable, Optional, Tuple

# Optional: jax imports (only used in jax env)
import numpy as np
from flax import struct
from jax.tree_util import Partial as partial

try:
    import chex
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
    chex = None

from jax import grad

from target_gym.base import EnvParams, EnvState
from target_gym.integration import (
    integrate_dynamics,
)


@struct.dataclass
class CarState(EnvState):
    x: float
    velocity: float
    throttle: float
    target_velocity: float


@struct.dataclass
class CarParams(EnvParams):
    gravity: float = 9.81
    initial_mass: float = 1_000.0
    min_velocity: float = 0.0
    max_velocity: float = 200 / 3.6
    target_velocity_range: Tuple[float, float] = (100 / 3.6, 130 / 3.6)
    initial_velocity_range: Tuple[float, float] = (30 / 3.6, 50 / 3.6)
    initial_z: float = 0.0
    initial_throttle: float = 0.0
    n_sensors: int = 10
    sensors_range: int = 100
    use_road_profile: int = 0
    max_steps_in_episode: int = 1_000

    # Drivetrain parameters
    gear_ratio: float = 4.0
    final_drive: float = 3.5
    wheel_radius: float = 0.35
    drivetrain_efficiency: float = 0.9

    # Resistances
    Cd: float = 0.3
    A: float = 2.2
    rho: float = 1.225
    Cr: float = 0.015

    # Braking
    braking_force: float = 1_000.0  # Max braking force [N]
    braking_friction: float = 0.5  # Coefficient of friction when braking

    # Electric engine
    base_rpm: float = 4000.0
    max_torque: float = 150.0


def electric_torque_from_rpm(rpm: float, throttle: float, params: CarParams):
    """
    Simple EV motor model:
    - Constant torque until base_rpm
    - Then constant power (torque decreases ~ 1/rpm)
    """
    rpm = jnp.asarray(rpm)

    base_rpm = getattr(params, "base_rpm", 4000.0)
    max_torque = getattr(params, "max_torque", 300.0)  # [Nm]
    max_power = (max_torque * base_rpm * 2 * jnp.pi) / 60.0  # [W]

    # Region 1: constant torque
    torque_const = max_torque * throttle

    # Region 2: constant power (T = P / ω)
    omega = rpm * 2 * jnp.pi / 60.0
    torque_power = (max_power / (omega + 1e-8)) * throttle

    torque = jnp.where(rpm <= base_rpm, torque_const, torque_power)
    return torque


def road_profile(x):
    """
    More realistic road profile with alternating climbs, plateaus, and descents.
    Designed to challenge velocity maintenance.
    Elevation changes are on the order of ±100 m.
    """
    # Normalize input to kilometers for readability
    km = x / 1000.0

    # Long-term trend: alternating climbs/descents
    trend = (
        50.0 * jnp.tanh((km - 2.0) * 0.5)  # first big climb around 2 km
        - 40.0 * jnp.tanh((km - 5.0) * 0.3)  # descent around 5 km
        + 30.0 * jnp.tanh((km - 8.0) * 0.4)  # climb again near 8 km
    )

    # Plateaus: gentle offsets at certain intervals
    plateaus = 20.0 * jnp.tanh((km - 3.0) * 2.0) * jnp.tanh(
        -(km - 4.0) * 2.0
    ) + 15.0 * jnp.tanh((km - 7.0) * 2.0) * jnp.tanh(-(km - 8.0) * 2.0)

    # Small irregularities (avoid perfectly flat sections)
    roughness = jnp.sin(km * 3.5) * 2.0 + jnp.sin(km * 11.0) * 1.0
    return (trend + plateaus + roughness) * 0


@partial(jax.jit, static_argnames=["road_profile"])
def compute_theta_from_position(
    x, road_profile: Optional[Callable[[float], float]], use_road_profile: int = 0
):
    if road_profile is None:
        return 0.0
    dzdx = grad(road_profile)
    slope = dzdx(x)
    return jnp.arctan(slope) * use_road_profile


def check_is_terminal(state: EnvState, params: CarParams, xp=jnp):
    terminated = jnp.logical_or(
        state.velocity <= params.min_velocity, state.velocity >= params.max_velocity
    )
    truncated = state.t >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: EnvState, params: CarParams, xp=jnp):
    terminated, _ = check_is_terminal(state, params, xp=xp)
    max_velocity_diff = params.max_velocity - params.min_velocity
    true_reward = (
        (max_velocity_diff - xp.abs(state.target_velocity - state.velocity))
        / max_velocity_diff
    ) ** 4
    reward = xp.where(
        terminated,
        -1.0 * params.max_steps_in_episode,
        true_reward,
    )
    return reward


def compute_thrust(throttle: float, velocity: float, params: CarParams):
    wheel_omega = velocity / params.wheel_radius
    engine_omega = wheel_omega * params.gear_ratio * params.final_drive
    rpm = engine_omega * 60.0 / (2 * jnp.pi)
    T_engine = electric_torque_from_rpm(rpm, throttle, params)
    T_wheel = (
        T_engine * params.gear_ratio * params.final_drive * params.drivetrain_efficiency
    )
    return T_wheel / params.wheel_radius


def compute_acceleration(velocity, position, action, params: CarParams):
    throttle = action
    theta = compute_theta_from_position(position, road_profile, params.use_road_profile)
    m, g = params.initial_mass, params.gravity

    # Engine thrust (positive throttle)
    F_thrust = jnp.where(throttle >= 0, compute_thrust(throttle, velocity, params), 0.0)

    # Braking (negative throttle)
    F_brake = jnp.where(
        throttle < 0,
        -throttle * params.braking_force + params.braking_friction * m * g,
        0.0,
    )

    F_drag = 0.5 * params.rho * params.Cd * params.A * velocity**2
    F_roll = params.Cr * m * g * jnp.cos(theta)
    F_gravity = m * g * jnp.sin(theta)

    F_total = F_thrust - F_drag - F_roll - F_gravity - F_brake
    metrics = {
        "F_thrust": F_thrust,
        "F_brake": F_brake,
        "F_drag": F_drag,
        "F_roll": F_roll,
        "F_gravity": F_gravity,
        "F_all": F_total,
    }
    return F_total / m, metrics


EPS = 1e-8


def compute_next_power(requested_power, current_power, delta_t):
    requested_power = jnp.clip(requested_power, 0.0 + EPS, 1.0)
    power_diff = requested_power - current_power
    current_power += (
        0.01 * delta_t * power_diff
    )  # TODO : parametrize how fast we reach the desired value
    # jax.debug.callback(check_power, current_power)
    return current_power


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    throttle_requested: float,
    state: EnvState,
    params: CarParams,
    integration_method: str = "rk4_1",
):
    dt = params.delta_t
    throttle_requested = jnp.clip(throttle_requested, -1.0, 1.0)
    throttle = compute_next_power(throttle_requested, state.throttle, delta_t=dt)

    _compute_acceleration = partial(
        compute_acceleration, action=throttle, params=params
    )

    velocity, position, metrics = integrate_dynamics(
        velocities=state.velocity,
        positions=state.x,
        delta_t=dt,
        compute_acceleration=_compute_acceleration,
        method=integration_method,
    )

    return (
        state.replace(x=position, velocity=velocity, throttle=throttle, t=state.t + 1),
        metrics,
    )


@partial(jax.jit, static_argnames=["params", "road_profile", "xp"])
def get_obs(
    state: EnvState,
    params: CarParams,
    road_profile: Callable[[float], float] = None,
    xp=jnp,
):
    sensor_x = state.x + jnp.linspace(0, params.sensors_range, num=params.n_sensors)
    sensor_theta = jax.vmap(compute_theta_from_position, in_axes=(0, None, None))(
        sensor_x, road_profile, params.use_road_profile
    )
    scalars = xp.stack([state.velocity, state.target_velocity])
    return xp.concatenate([scalars, sensor_theta])
