from typing import Any

import chex
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax
from plane.utils import compute_norm_from_coordinates

Action = Any


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
    z: float  # altitude (in feet) relative to the ground.
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
        return compute_norm_from_coordinates([self.x_dot, self.z_dot]) / 340

    # simulation
    t: int  # the time (in seconds TODO : check this) since starting the simulation.


@struct.dataclass
class EnvParams:
    """
    Constants of the simulation
    """

    gravity: float = 9.81  # the gravitational acceleration (in m.s-2).
    initial_mass: float = 73500  # the initial mass (in kilograms) of our plane.
    thrust_output_at_sea_level: float = (
        280000  # the maximal force (in Newtons) deployed by the engine at sea level.
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
    speed_of_sound: float = 340  # speed of sounds (in m.s-1)


def compute_next_state(action: Action, state: EnvState, params: EnvParams) -> EnvState:
    raise NotImplementedError


def compute_reward(state: EnvState, params: EnvParams) -> float:
    raise NotImplementedError


class CartPole(environment.Environment):
    """
    JAX Compatible version of CartPole-v1 OpenAI gym environment. Source:
    github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (4,)  # TODO : adapt

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for CartPole-v1
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: float, params: EnvParams
    ) -> tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)

        state = compute_next_state(action, state, params)
        reward = compute_reward(state, params)
        # Update state dict and evaluate termination conditions
        state = EnvState()  # Todo : fill
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )
