from typing import Any, Optional
import jax
import chex
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax

from plane.utils import compute_norm_from_coordinates
from plane.dynamics import (
    compute_acceleration,
    compute_next_power,
    compute_thrust_output,
    compute_speed_and_pos_from_acceleration,
    compute_air_density_from_altitude,
)

Action = Any
SPEED_OF_SOUND = 343.0


@struct.dataclass
class EnvState:
    """
    # State variables (as opposed to constants) used for dynamics calculation.
     The units are purposefuly in non-aeronautical units, but would be converted and\
          displayed in the correct aeronautical units.
    """

    # plane
    x: float  # horizontal position (in neters) relative to the starting position
    x_dot: float  # horizontal velocity (in m.s-1) w.r.t the ground
    z: float  # altitude (in meters) relative to the ground.
    z_dot: float  # altitude variation (in feet per minute), positive is going up.
    theta: float  # pitch angle (in degrees) (angle between ground and plane)
    alpha: float  # angle of attack (in degrees) angle between the relative wind and the plane
    gamma: float  # slope (in degrees) : angle between the ground and the plane's velocity vector.
    m: float  # the mass of the airplane (in kilograms)
    power: float  # power (in Watts) deployed by the engine.
    fuel: float  # the mass (in kilograms) of fuel available.

    # air
    rho: float  # air density (in kg.m^-3)

    @property
    def atltitude_factor(self):
        const = -9.33e-5
        return jnp.exp(const * self.z)

    @property
    def M(self):
        """
        The mach number
        """
        return (
            compute_norm_from_coordinates(jnp.array([self.x_dot, self.z_dot]))
            / SPEED_OF_SOUND
        )

    # simulation
    t: int  # the time (in seconds TODO : check this) since starting the simulation.

    # target
    target_altitude: float  # the altitude (in meters) the plane should try staying at.

    # TODO : add wind.


@struct.dataclass
class EnvParams:
    """
    Constants of the simulation
    """

    gravity: float = 9.81  # the gravitational acceleration (in m.s-2).
    initial_mass: float = 73500.0  # the initial mass (in kilograms) of our plane.
    thrust_output_at_sea_level: float = (
        280000.0  # the maximal force (in Newtons) deployed by the engine at sea level.
    )
    air_density_at_sea_level: float = 1.225  # the air density (kg.m-3) at sea level.
    frontal_surface: float = 12.6  # the frontal surface (in m^2) of the plane.
    wings_surface: float = 122.6  # the wings surface (in m^2) of the plane.
    C_x0: float = 0.095  # initial drag coefficient (no unit).
    C_z0: float = 0.9  # initial lift coefficient (no unit).
    M_crit: float = 0.78  # the critical Mach number (no unit).
    initial_fuel_quantity: float = 23860 / 1.25  # initial fuel mass (in kilograms).
    specific_fuel_consumption: float = (
        17.5 / 1000
    )  # thrust-specific fuel consumption (kg.kN^-1.s-1)
    delta_t: int = 1  # timestep size (in seconds)
    speed_of_sound: float = SPEED_OF_SOUND  # speed of sounds (in m.s-1)

    # Episodes
    max_steps_in_episode: int = 1000
    min_alt: float = 1000.0
    max_alt: float = 15000.0
    target_altitude_range: tuple[float, float] = (
        5000.0,
        10000.0,
    )  # the min and max values for the target_altitude (in meters)
    initial_altitude_range: tuple[float, float] = (
        5000.0,
        10000.0,
    )  # the min and max values for the initial altitude (in meters)


def check_mass_does_not_increase(old_mass, new_mass):
    assert old_mass >= new_mass


def compute_next_state(
    power_requested: float, state: EnvState, params: EnvParams
) -> EnvState:
    # power
    power = compute_next_power(power_requested, state.power)
    thrust = compute_thrust_output(
        power=power,
        altitude_factor=state.atltitude_factor,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
    )
    # acceleration, speed and position
    a_x, a_z = compute_acceleration(
        thrust=thrust,
        m=state.m,
        gravity=params.gravity,
        x_dot=state.x_dot,
        z_dot=state.z_dot,
        air_density_at_sea_level=params.air_density_at_sea_level,
        atltitude_factor=state.atltitude_factor,
        frontal_surface=params.frontal_surface,
        wings_surface=params.wings_surface,
        alpha=state.alpha,
        M=state.M,
        M_crit=params.M_crit,
        C_x0=params.C_x0,
        C_z0=params.C_z0,
        gamma=state.gamma,
        theta=state.theta,
    )
    (
        x_dot,
        z_dot,
        x,
        z,
    ) = compute_speed_and_pos_from_acceleration(
        state.x_dot, state.z_dot, state.x, state.z, a_x, a_z, params.delta_t
    )

    # time
    t = state.t + 1

    # angles
    gamma = jnp.arcsin(
        z_dot / compute_norm_from_coordinates(jnp.array([x_dot, z_dot + 1e-6]))
    )
    alpha = state.theta - gamma

    # mass
    m = params.initial_mass + state.fuel
    jax.debug.callback(check_mass_does_not_increase, state.m, m)

    new_state = EnvState(
        x=x,
        x_dot=x_dot,
        z=z,
        z_dot=z_dot,
        theta=state.theta,  # no change atm
        alpha=alpha,
        gamma=gamma,
        m=m,  # no change atm
        power=power,
        fuel=state.fuel,  # no change atm
        rho=compute_air_density_from_altitude(
            params.air_density_at_sea_level, altitude_factor=state.atltitude_factor
        ),
        t=t,
        target_altitude=state.target_altitude,
    )
    return new_state


def compute_reward(state: EnvState, params: EnvParams) -> jnp.float32:
    done1 = jnp.logical_or(
        state.z < params.min_alt,
        state.z > params.max_alt,
    )
    reward = jax.lax.select(
        done1,
        -jnp.array(1.0) * params.max_steps_in_episode,
        -(((state.target_altitude - state.z) / params.max_alt) ** 2),
    )
    return reward


class Airplane2D(environment.Environment):
    """
    JAX Compatible version of CartPole-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (6,)  # TODO : adapt

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for Airplane2D
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: float, params: EnvParams
    ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)

        state = compute_next_state(action, state, params)
        reward = compute_reward(state, params)
        # Update state dict and evaluate termination conditions
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        initial_x = 0.0
        key, altitude_key = jax.random.split(key)
        initial_z = jax.random.uniform(
            altitude_key,
            minval=params.initial_altitude_range[0],
            maxval=params.initial_altitude_range[1],
        )
        initial_z_dot = 0.0
        initial_x_dot = 250.0  # TODO : improve this to have a "stable" start ?
        initial_theta = 0.0
        initial_gamma = jnp.arcsin(
            initial_z_dot
            / compute_norm_from_coordinates(
                jnp.array([initial_x_dot, initial_z_dot + 1e-6])
            )
        )
        initial_alpha = initial_theta - initial_gamma
        initial_m = params.initial_mass + params.initial_fuel_quantity
        initial_power = 0.5
        initial_fuel = params.initial_fuel_quantity
        initial_rho = params.air_density_at_sea_level
        key, target_key = jax.random.split(key)
        target_altitude = jax.random.uniform(
            target_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )
        state = EnvState(
            x=initial_x,
            x_dot=initial_x_dot,
            z=initial_z,
            z_dot=initial_z_dot,
            theta=initial_theta,
            alpha=initial_alpha,
            gamma=initial_gamma,
            m=initial_m,
            power=initial_power,
            fuel=initial_fuel,
            rho=initial_rho,
            t=0,
            target_altitude=target_altitude,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        obs = jnp.stack(
            [
                state.z,
                state.x_dot,
                state.z_dot,
                state.theta,
                state.gamma,
                state.target_altitude,
            ]
        )
        return obs

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(low=0, high=1.0, shape=())

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check termination criteria
        done1 = jnp.logical_or(
            state.z < params.min_alt,
            state.z > params.max_alt,
        )

        # Check number of steps in episode termination condition
        done_steps = state.t >= params.max_steps_in_episode
        done = jnp.logical_or(done1, done_steps)
        return done
