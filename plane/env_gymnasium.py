from typing import Any, Optional, Sequence
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from gymnasium.error import DependencyNotInstalled
from dataclasses import dataclass
from plane.dynamics import (
    compute_acceleration,
    compute_air_density_from_altitude,
    compute_next_power,
    compute_speed_and_pos_from_acceleration,
    compute_thrust_output,
)
from plane.utils import compute_norm_from_coordinates, plot_features_from_trajectory

Action = Any
SPEED_OF_SOUND = 343.0


@dataclass
class EnvMetrics:
    drag: float
    lift: float
    S_x: float
    S_z: float
    C_x: float
    C_z: float
    F_x: float
    F_z: float


@dataclass
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
        return np.exp(const * self.z)

    @property
    def M(self):
        """
        The mach number
        """
        return (
            compute_norm_from_coordinates(np.array([self.x_dot, self.z_dot]))
            / SPEED_OF_SOUND
        )

    # simulation
    t: int  # the time (in seconds TODO : check this) since starting the simulation.

    # target
    target_altitude: float  # the altitude (in meters) the plane should try staying at.

    # TODO : add wind.

    # Metrics for analysis
    metrics: Optional[EnvMetrics] = None


@dataclass
class EnvParams:
    """
    Constants of the simulation
    """

    gravity: float = 9.81  # the gravitational acceleration (in m.s-2).
    initial_mass: float = 73500.0  # the initial mass (in kilograms) of our plane.
    thrust_output_at_sea_level: float = (
        240000.0  # the maximal force (in Newtons) deployed by the engine at sea level.
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
        2000.0,
        5000.0,
    )  # the min and max values for the target_altitude (in meters)
    initial_altitude_range: tuple[float, float] = (
        2000.0,
        5000.0,
    )  # the min and max values for the initial altitude (in meters)


def check_mass_does_not_increase(old_mass, new_mass):
    assert old_mass >= new_mass


def check_is_terminal(state: EnvState, params: EnvParams) -> bool:
    """Check whether state is terminal."""
    # Check termination criteria
    terminated = np.logical_or(
        state.z < params.min_alt,
        state.z > params.max_alt,
    )

    # Check number of steps in episode termination condition
    truncated = state.t >= params.max_steps_in_episode
    return terminated, truncated


def compute_next_state(
    power_requested: float, state: EnvState, params: EnvParams
) -> EnvState:
    # power
    power_requested = (power_requested + 1) / 10
    power = compute_next_power(power_requested, state.power)

    thrust = compute_thrust_output(
        power=power,
        altitude_factor=state.atltitude_factor,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
    )
    # acceleration, speed and position
    a_x, a_z, metrics = compute_acceleration(
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
    gamma = np.arcsin(
        z_dot / compute_norm_from_coordinates(np.array([x_dot, z_dot + 1e-6]))
    )
    alpha = state.theta - gamma

    # mass
    m = params.initial_mass + state.fuel
    check_mass_does_not_increase(state.m, m)

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
        metrics=EnvMetrics(*metrics),
    )
    return new_state


def compute_reward(state: EnvState, params: EnvParams) -> np.float32:
    max_alt_diff = params.max_alt - params.min_alt
    if state.z < params.min_alt or state.z > params.max_alt:
        return -np.array(1.0) * params.max_steps_in_episode
    return (max_alt_diff - np.abs(params.max_alt - state.z)) / max_alt_diff


class Airplane2D(gym.Env):
    """
    JAX Compatible 2D-Airplane environment.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, params=None, render_mode: Optional[str] = None):
        super().__init__()
        self.obs_shape = (6,)  # TODO : adapt
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape
        )
        if params is None:
            self.params = self.default_params
        else:
            self.params = params

        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.frames = []

        self.x_threshold = 2.4

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters for Airplane2D
        return EnvParams()

    def step(self, action: float):
        """Performs step transitions in the environment."""
        self.state = compute_next_state(action, self.state, self.params)
        reward = compute_reward(self.state, self.params)
        terminated, truncated = self.is_terminal(self.state, self.params)

        return (
            self.get_obs(self.state),
            reward,
            terminated,
            truncated,
            {"state": self.state},
        )

    def reset(self, seed=None, options=None):
        """Performs resetting of environment."""
        super().reset(seed=seed)
        initial_x = 0.0
        initial_z = np.random.uniform(
            low=self.params.initial_altitude_range[0],
            high=self.params.initial_altitude_range[1],
        )
        initial_z_dot = 0.0
        initial_x_dot = 200.0  # TODO : improve this to have a "stable" start ?
        initial_theta = 0.0
        initial_gamma = np.arcsin(
            initial_z_dot
            / compute_norm_from_coordinates(
                np.array([initial_x_dot, initial_z_dot + 1e-6])
            )
        )
        initial_alpha = initial_theta - initial_gamma
        initial_m = self.params.initial_mass + self.params.initial_fuel_quantity
        initial_power = 0.5
        initial_fuel = self.params.initial_fuel_quantity
        initial_rho = self.params.air_density_at_sea_level
        target_altitude = np.random.uniform(
            low=self.params.target_altitude_range[0],
            high=self.params.target_altitude_range[1],
        )
        self.state = EnvState(
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
            metrics=EnvMetrics(
                drag=0.0, lift=0.0, S_x=0.0, S_z=0.0, C_x=0.0, C_z=0.0, F_x=0.0, F_z=0.0
            ),  # TODO : add real values
        )
        return self.get_obs(self.state), {"state": self.state}

    def get_obs(self, state: EnvState):
        """Applies observation function to state."""
        obs = np.stack(
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

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return check_is_terminal(state, params)

    def visualize(
        self,
        states: Sequence[EnvState],
        params: EnvParams,
        exp_name: Optional[str] = None,
    ) -> None:
        if exp_name is None:
            folder = "figs"
            os.makedirs(folder, exist_ok=True)

        else:
            folder = os.path.join("figs", exp_name)
            os.makedirs(folder, exist_ok=True)

        plot_features_from_trajectory(states, folder)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 343 * 1000
        scale = self.screen_width / world_width
        scale_y = self.screen_height / self.params.max_alt
        plane_width = 50.0
        plane_height = 10.0

        if self.state is None:
            return None

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = (
            -plane_width / 2,
            plane_width / 2,
            plane_height / 2,
            -plane_height / 2,
        )
        axleoffset = plane_height / 4.0
        planex = self.state.x * scale  # MIDDLE OF CART
        planey = self.state.z * scale_y  # TOP OF CART
        plane_coords = [(l, b), (l, t), (r, t), (r, b)]
        plane_coords = [(c[0] + planex, c[1] + planey) for c in plane_coords]
        gfxdraw.aapolygon(self.surf, plane_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, plane_coords, (0, 0, 0))

        gfxdraw.hline(
            self.surf,
            0,
            self.screen_width,
            int(self.state.target_altitude * scale_y),
            (0, 0, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
        self.frames.append(frame)
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "rgb_array_list":
            return self.frames


if __name__ == "__main__":
    from gymnasium.utils.save_video import save_video

    env = Airplane2D(render_mode="rgb_array_list")
    env.reset()
    step_starting_index = 0
    episode_index = 0
    step_index = 0
    done = False
    while not done:
        action = 9
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        frames = env.render()
        if done:
            save_video(
                frames,
                "videos",
                fps=env.metadata["render_fps"],
                step_starting_index=step_starting_index,
                episode_index=episode_index,
            )
            step_starting_index = step_index + 1
            step_index += 1
            episode_index += 1
    env.close()
