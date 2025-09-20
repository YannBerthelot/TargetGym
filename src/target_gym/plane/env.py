from typing import Tuple

# Optional: jax imports (only used in jax env)
import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.integration import (
    integrate_dynamics,
)
from target_gym.plane.dynamics import (
    compute_acceleration,
    compute_air_density_from_altitude,
    compute_alpha,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_stick,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    compute_velocity_from_horizontal_and_vertical_speed,
)

SPEED_OF_SOUND = 343.0
DEBUG = False


@struct.dataclass
class EnvMetrics:
    drag: float
    lift: float
    S_x: float
    S_z: float
    C_x: float
    C_z: float
    F_x: float
    F_z: float


@struct.dataclass
class EnvState:
    x: float
    x_dot: float
    z: float
    z_dot: float
    theta: float
    theta_dot: float
    alpha: float
    gamma: float
    m: float
    power: float
    stick: float
    fuel: float
    t: int
    target_altitude: float

    @property
    def rho(self):
        return compute_air_density_from_altitude(self.z)

    @property
    def speed_of_sound(self):
        return compute_speed_of_sound_from_altitude(self.z)

    @property
    def M(self):
        return compute_Mach_from_velocity_and_speed_of_sound(
            compute_velocity_from_horizontal_and_vertical_speed(self.x_dot, self.z_dot),
            self.speed_of_sound,
        )


@struct.dataclass
class EnvParams:
    gravity: float = 9.81
    initial_mass: float = 73_500.0
    thrust_output_at_sea_level: float = 240_000.0
    air_density_at_sea_level: float = 1.225
    frontal_surface: float = 12.6
    wings_surface: float = 122.6
    C_x0: float = 0.095
    C_z0: float = 0.9
    M_crit: float = 0.78
    initial_fuel_quantity: float = 23860 / 1.25
    specific_fuel_consumption: float = 17.5 / 1000

    cl_alpha: float = 0.04  # per deg
    cl0: float = 0.2  # zero-lift AoA
    cd0: float = 0.02  # zero-lift drag
    k: float = 0.045  # induced drag factor
    aoa_stall: float = 15.0  # deg
    CL_max: float = 1.5
    M_crit: float = 0.80
    k_drag: float = 5.0

    speed_of_sound: float = SPEED_OF_SOUND
    I: float = 9_000_000
    moment_arm_stabilizer: float = 15.0
    moment_arm_wings: float = 1.5
    stabilizer_surface: float = 27
    elevator_surface: float = 10

    max_steps_in_episode: int = 10_000
    min_alt: float = 0.0
    max_alt: float = 40_000.0 / 3.281
    target_altitude_range: Tuple[float, float] = (5_000.0, 5_000.0)
    initial_altitude_range: Tuple[float, float] = (3_000.0, 5_000.0)
    initial_z_dot: float = 0.0
    initial_x_dot: float = 200.0
    initial_theta_dot: float = 0.0
    initial_theta: float = 0.0
    initial_power: float = 1.0
    initial_stick: float = 0.0

    delta_t: float = 1.0


def check_mass_does_not_increase(old_mass, new_mass, xp=jnp):
    """Check that mass does not increase. Safe for JIT if wrapped in callback."""
    if jax is not None and xp is jnp:
        jax.debug.callback(
            lambda o, n: None if o >= n else AssertionError("Mass increased"),
            old_mass,
            new_mass,
        )
    else:
        assert old_mass >= new_mass


def check_is_terminal(state: EnvState, params: EnvParams, xp=jnp):
    """Return True if the episode should terminate."""
    terminated = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    truncated = state.t >= params.max_steps_in_episode

    # done = xp.logical_or(done_alt, done_steps)
    return terminated, truncated


def check_no_nan(x, id=None):
    """Assert that no NaNs are present in arrays, scalars, or EnvState."""
    if isinstance(x, EnvState):
        # Iterate over fields of the dataclass
        for name, value in x.__dict__.items():
            try:
                check_no_nan(value, id=f"{id}.{name}" if id else name)
            except AssertionError as e:
                raise AssertionError(str(e)) from None
    else:
        if jnp.isnan(x).any():
            raise AssertionError(f"NaN detected in {id}: {x}")


def compute_reward(state: EnvState, params: EnvParams, xp=jnp):
    """Return reward for a given state. Safe for JIT."""
    xp = jnp
    done_alt = xp.logical_or(state.z <= params.min_alt, state.z >= params.max_alt)
    max_alt_diff = params.max_alt - params.min_alt
    reward = xp.where(
        done_alt,
        -1.0 * params.max_steps_in_episode,
        ((max_alt_diff - xp.abs(state.target_altitude - state.z)) / max_alt_diff) ** 2,
    )
    return reward


def get_obs(state: EnvState, xp=jnp):
    """Applies observation function to state."""
    return xp.stack(
        [
            state.x_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.gamma,
            state.target_altitude,
            state.power,
            state.stick,
        ]
    )


@partial(jax.jit, static_argnames=["min", "max"])
def clip_acceleration(a: jnp.ndarray, min: tuple, max: tuple):
    return jnp.clip(a, min=jnp.array(min), max=jnp.array(max))


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    power_requested: float,
    stick_requested: float,
    state: EnvState,
    params: EnvParams,
    integration_method: str = "rk4_1",
):
    """Compute next state and metrics using multiple sub-steps with jax.lax.scan."""
    dt = params.delta_t
    power = compute_next_power(power_requested, state.power, dt)
    stick = compute_next_stick(stick_requested, state.stick, dt)

    # Compute thrustx
    thrust = compute_thrust_output(
        power=power,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        rho=state.rho,
        M=state.M,
    )
    positions = jnp.array([state.x, state.z, state.theta])
    velocities = jnp.array([state.x_dot, state.z_dot, state.theta_dot])
    _compute_acceleration = partial(
        compute_acceleration,
        action=(thrust, stick),
        params=params,
        clip=True,
        min_clip_boundaries=(-100, -100, -1.5),
        max_clip_boundaries=(100, 100, 1.5),
    )

    (x_dot, z_dot, theta_dot), (x, z, theta), metrics = integrate_dynamics(
        velocities=velocities,
        positions=positions,
        delta_t=dt,
        compute_acceleration=_compute_acceleration,
        method=integration_method,
    )

    alpha, gamma = compute_alpha(theta, x_dot, z_dot)
    m = params.initial_mass + state.fuel

    new_state = EnvState(
        x=x,
        x_dot=x_dot,
        z=z,
        z_dot=z_dot,
        theta=theta,
        theta_dot=theta_dot,
        alpha=alpha,
        gamma=gamma,
        m=m,
        power=power,
        stick=stick,
        fuel=state.fuel,
        t=state.t + 1,
        target_altitude=state.target_altitude,
    )
    return new_state, metrics
