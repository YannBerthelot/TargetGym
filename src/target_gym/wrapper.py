from typing import Callable, Optional, Type

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from target_gym.base import EnvParams


def gym_wrapper_factory(jax_env_class: Type):
    """
    Factory function that returns a Gym-compatible wrapper class
    for any JAX-based environment.
    """

    class GymWrapperJaxEnv(gym.Env):
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

        def __init__(
            self,
            render_mode: Optional[str] = None,
            env_params: Optional[EnvParams] = None,
            **jax_env_kwargs,
        ):
            super().__init__()
            # Instantiate the JAX environment
            self.jax_env = jax_env_class(**jax_env_kwargs)
            self.render_mode = render_mode

            # Infer observation shape
            self.obs_shape = getattr(self.jax_env, "obs_shape", None)
            if self.obs_shape is None:
                obs, _ = self.jax_env.reset_env(
                    jax.random.PRNGKey(0), getattr(self.jax_env, "default_params", None)
                )
                self.obs_shape = np.array(obs).shape

            # Observation space
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
            )

            # Action space
            if hasattr(self.jax_env, "action_space"):
                action_space = self.jax_env.action_space()
                self.action_space = spaces.Box(
                    low=np.array(action_space.low),
                    high=np.array(action_space.high),
                    shape=action_space.shape,
                    dtype=np.float32,
                )
            else:
                # Default fallback
                self.action_space = spaces.Box(
                    low=np.array([0.0, -1.0]),
                    high=np.array([1.0, 1.0]),
                    shape=(2,),
                    dtype=np.float32,
                )

            # Internal state
            self.state = None
            self.env_params = (
                env_params
                if env_params is not None
                else getattr(self.jax_env, "default_params", None)
            )
            self.frames = []
            self.screen = None
            self.clock = None
            self.key = jax.random.PRNGKey(0)

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                self.key = jax.random.PRNGKey(seed)
            obs, self.state = self.jax_env.reset_env(self.key, self.env_params)
            self.frames = []
            self.screen = None
            self.clock = None
            return np.array(obs, dtype=np.float32), {}

        def is_terminal(self, *args, **kwargs):
            return self.jax_env.is_terminal(*args, **kwargs)

        def step(self, action: np.ndarray):
            obs, new_state, reward, done, info = self.jax_env.step_env(
                self.key,
                self.state,
                jnp.array(action, dtype=jnp.float32),
                self.env_params,
            )
            self.state = new_state
            return (
                np.array(obs, dtype=np.float32),
                float(reward),
                bool(done),
                False,
                info,
            )

        def render(self):
            frames, self.screen, self.clock = self.jax_env.render(
                self.screen, self.state, self.env_params, self.frames, self.clock
            )
            self.frames = frames

            if self.render_mode == "human":
                import pygame

                pygame.event.pump()
                if self.clock:
                    self.clock.tick(self.metadata["render_fps"])
                return None
            elif self.render_mode in ["rgb_array", "rgb_array_list"]:
                return self.frames[-1] if self.frames else None

        def save_video(
            self,
            select_action: Callable,
            folder: str = "videos",
            episode_index: int = 0,
            FPS: int = 60,
            format: str = "mp4",
            seed: Optional[int] = None,
        ):
            return self.jax_env.save_video(
                select_action,
                seed if seed is not None else 0,
                self.env_params,
                folder=folder,
                episode_index=episode_index,
                FPS=FPS,
                format=format,
            )

    return GymWrapperJaxEnv
