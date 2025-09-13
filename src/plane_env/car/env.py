from typing import Callable, Tuple

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


@struct.dataclass
class EnvState:
    x: float
    velocity: float
    t: int
    throttle: float
    target_velocity: float


@struct.dataclass
class EnvParams:
    gravity: float = 9.81
    initial_mass: float = 1_000.0
    delta_t: float = 1.0
    n_substeps: int = 5

    max_steps_in_episode: int = 1_000
    min_velocity: float = 0.0
    max_velocity: float = 200 / 3.6
    target_velocity_range: Tuple[float, float] = (100 / 3.6, 130 / 3.6)
    initial_velocity_range: Tuple[float, float] = (30 / 3.6, 50 / 3.6)
    initial_z: float = 0.0
    initial_throttle: float = 0.0
    n_sensors: int = 10
    sensors_range: int = 100

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

    # Engine torque curve
    idle_rpm: float = 800.0
    redline_rpm: float = 6000.0
    peak_rpm: float = 3000.0
    peak_torque: float = 150.0


def engine_torque_from_rpm(rpm: float, throttle: float, params: EnvParams):
    rpm = jnp.asarray(rpm)
    rpm_clipped = jnp.clip(rpm, params.idle_rpm, params.redline_rpm)
    rpm_norm = (rpm_clipped - params.idle_rpm) / (params.redline_rpm - params.idle_rpm)
    peak_norm = (params.peak_rpm - params.idle_rpm) / (
        params.redline_rpm - params.idle_rpm
    )

    sigma = getattr(params, "torque_sigma", 0.30)
    gauss = jnp.exp(-0.5 * ((rpm_norm - peak_norm) / sigma) ** 2)

    torque_curve = gauss
    return throttle * params.peak_torque * torque_curve


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
def compute_theta_from_position(x, road_profile: Callable[[float], float]):
    dzdx = grad(road_profile)
    slope = dzdx(x)
    return jnp.arctan(slope)


def check_is_terminal(state: EnvState, params: EnvParams, xp=jnp):
    terminated = jnp.logical_or(
        state.velocity <= params.min_velocity, state.velocity >= params.max_velocity
    )
    truncated = state.t >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: EnvState, params: EnvParams, xp=jnp):
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


def compute_thrust(throttle: float, velocity: float, params: EnvParams):
    wheel_omega = velocity / params.wheel_radius
    engine_omega = wheel_omega * params.gear_ratio * params.final_drive
    rpm = engine_omega * 60.0 / (2 * jnp.pi)
    T_engine = engine_torque_from_rpm(rpm, throttle, params)
    T_wheel = (
        T_engine * params.gear_ratio * params.final_drive * params.drivetrain_efficiency
    )
    return T_wheel / params.wheel_radius


def compute_acceleration(throttle, velocity, theta, params: EnvParams):
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
    return F_total / m


EPS = 1e-8


def compute_next_power(requested_power, current_power, delta_t):
    requested_power = jnp.clip(requested_power, 0.0 + EPS, 1.0)
    power_diff = requested_power - current_power
    current_power += (
        0.01 * delta_t * power_diff
    )  # TODO : parametrize how fast we reach the desired value
    # jax.debug.callback(check_power, current_power)
    return current_power


def compute_next_state(
    throttle_requested: float,
    state: EnvState,
    params: EnvParams,
    n_substeps: int = 1,
    xp=jnp,
):
    dt = params.delta_t / n_substeps
    throttle_requested = jnp.clip(throttle_requested, -1.0, 1.0)

    def step_fn(current_state, _):
        throttle = compute_next_power(
            throttle_requested, current_state.throttle, delta_t=dt
        )
        theta = compute_theta_from_position(current_state.x, road_profile)
        acc = compute_acceleration(throttle, current_state.velocity, theta, params)
        v_new = current_state.velocity + acc * dt
        v_new = jnp.clip(v_new, -params.max_velocity, params.max_velocity)
        x_new = current_state.x + v_new * dt
        new_state = current_state.replace(x=x_new, velocity=v_new, throttle=throttle)
        return new_state, None

    if xp is jnp:
        final_state, _ = jax.lax.scan(step_fn, state, None, length=n_substeps)
        final_state = final_state.replace(t=state.t + 1)
    else:
        final_state = state
        for _ in range(n_substeps):
            final_state, _ = step_fn(final_state, None)
        final_state = final_state.replace(t=state.t + 1)

    return final_state


@partial(jax.jit, static_argnames=["params", "road_profile", "xp"])
def get_obs(state: EnvState, params: EnvParams, road_profile, xp=jnp):
    sensor_x = state.x + jnp.linspace(0, params.sensors_range, num=params.n_sensors)
    sensor_theta = jax.vmap(compute_theta_from_position, in_axes=(0, None))(
        sensor_x, road_profile
    )
    scalars = xp.stack([state.velocity, state.target_velocity])
    return xp.concatenate([scalars, sensor_theta])
