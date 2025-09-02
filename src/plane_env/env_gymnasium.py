from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pygame

from plane_env.env import (
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_env_classes,
    save_video,
)
from plane_env.rendering import _render

EnvState, EnvParams, EnvMetrics = get_env_classes(use_jax=False)


class Airplane2D(gym.Env):
    """
    JAX Compatible 2D-Airplane environment.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400
    max_steps = 1_000

    def __init__(
        self, params=None, render_mode: Optional[str] = None, mode="power_and_stick"
    ):
        super().__init__()
        self.obs_shape = (10,)  # TODO : adapt
        # self.action_space = gym.spaces.MultiDiscrete([10,31])  # 10 power levels (0-9) and 30 pitch levels (-15 to 15 degrees)
        if mode == "power_and_stick":
            self.action_space = gym.spaces.Box(
                low=np.array(
                    [0, -1]
                ),  # Power from 0 to 9, pitch from -15 to 15 degrees
                high=np.array([1, 1]),
                dtype=np.float32,
            )
        elif mode == "power_only":
            self.action_space = gym.spaces.Box(
                low=np.array([0]),  # Power from 0 to 9
                high=np.array([1]),
                dtype=np.float32,
            )
        elif mode == "stick_only":
            self.action_space = gym.spaces.Box(
                low=np.array([-1]),  # pitch from -15 to 15 degrees
                high=np.array([1]),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape
        )
        if params is None:
            self.params = self._default_params
        else:
            self.params = params

        # Rendering
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.frames = []

        self.x_threshold = 2.4

        # Add new attribute for rendering
        self.positions_history = []

    @property
    def _default_params(self) -> EnvParams:
        # Default environment parameters for Airplane2D
        return EnvParams()

    def step(self, action, params: EnvParams = None):
        state = self.state
        params = self.params if params is None else params
        power, stick = action  # cannot just unpack action because of different modes
        stick = np.deg2rad(stick * 15)
        new_state, metrics = compute_next_state(power, stick, state, params, xp=np)
        reward = compute_reward(new_state, params, xp=np)
        terminated, truncated = check_is_terminal(new_state, params)
        obs = self.get_obs(new_state)
        self.state = new_state
        return obs, reward, terminated, truncated, {"metrics": metrics}

    def reset(self, seed=None, options=None):
        """Performs resetting of environment."""
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        initial_x = 0.0
        initial_z = rng.uniform(
            self.params.initial_altitude_range[0], self.params.initial_altitude_range[1]
        )
        initial_z_dot = self.params.initial_z_dot
        initial_x_dot = self.params.initial_x_dot
        initial_theta = np.deg2rad(self.params.initial_theta)
        initial_gamma = np.arcsin(
            initial_z_dot / np.linalg.norm([initial_x_dot, initial_z_dot + 1e-6])
        )
        initial_alpha = initial_theta - initial_gamma
        initial_m = self.params.initial_mass + self.params.initial_fuel_quantity
        initial_power = self.params.initial_power
        initial_stick = np.deg2rad(self.params.initial_stick)
        initial_fuel = self.params.initial_fuel_quantity
        target_altitude = rng.uniform(
            self.params.target_altitude_range[0], self.params.target_altitude_range[1]
        )

        self.state = EnvState(
            x=initial_x,
            x_dot=initial_x_dot,
            z=initial_z,
            z_dot=initial_z_dot,
            theta=initial_theta,
            theta_dot=np.deg2rad(self.params.initial_theta_dot),
            alpha=initial_alpha,
            gamma=initial_gamma,
            m=initial_m,
            power=initial_power,
            stick=initial_stick,
            fuel=initial_fuel,
            t=0,
            target_altitude=target_altitude,
        )
        return self.get_obs(self.state), self.state

    def get_obs(self, state: EnvState):
        """Applies observation function to state."""
        obs = np.stack(
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
                state.z - state.target_altitude,
            ]
        )
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return check_is_terminal(state, params)

    def render(self):
        # if self.render_mode is None:
        #     assert self.spec is not None
        #     gym.logger.warn(
        #         "You are calling render method without specifying any render mode. "
        #         "You can specify the render_mode at initialization, "
        #         f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
        #     )
        #     return

        # try:
        #     import pygame
        # except ImportError as e:
        #     raise DependencyNotInstalled(
        #         "pygame is not installed, run `pip install gymnasium[classic-control]`"
        #     ) from e

        # Use the shared render function
        out = self.render_plane(self.screen, self.state, self.params, [], self.clock)
        if out is not None:
            frames, screen, clock = out
            self.screen = screen
            self.clock = clock

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            return None

        elif self.render_mode == "rgb_array":
            return frames[-1] if frames else None

        if self.render_mode == "rgb_array_list":
            self.frames.extend(frames)
            return self.frames

    def save_video(
        self,
        select_action: Callable,
        folder="videos",
        episode_index=0,
        FPS=60,
        params=None,
        seed=None,
        format="mp4",
    ):
        return save_video(
            self, select_action, folder, episode_index, FPS, params, seed, format=format
        )


if __name__ == "__main__":
    env = Airplane2D(render_mode="rgb_array")
    seed = 42
    env_params = EnvParams(max_steps_in_episode=2_000)
    action = (0.8, 0.0)
    env.save_video(
        lambda o: action, seed=seed, folder="videos", episode_index=0, params=env_params
    )
