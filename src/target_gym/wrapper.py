import warnings
from typing import Callable, Optional, Type

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from target_gym.base import EnvParams


class GymnasiumEnv(gym.Env):
    """A JAX environment behind the stateful Gymnasium API.

    Build it with ``target_gym.make_gymnasium(name)``, which passes the scored
    parameters. ``env_params=None`` falls back to ``jax_env.default_params``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        jax_env,
        env_params: Optional[EnvParams] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.jax_env = jax_env
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
        # Seeds Gymnasium's ``np_random``, which the API promises after a
        # seeded reset. The dynamics draw from ``self.key`` alone.
        super().reset(seed=seed)
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        # A fresh subkey per call: reusing one key gave every unseeded
        # episode the same start and every trip restart the same draw.
        self.key, key = jax.random.split(self.key)
        obs, self.state = self.jax_env.reset_env(key, self.env_params)
        self.frames = []
        self.screen = None
        self.clock = None
        return np.array(obs, dtype=np.float32), {}

    def is_terminated(self, *args, **kwargs):
        return self.jax_env.is_terminated(*args, **kwargs)

    def is_truncated(self, *args, **kwargs):
        return self.jax_env.is_truncated(*args, **kwargs)

    def step(self, action: np.ndarray):
        self.key, key = jax.random.split(self.key)
        obs, new_state, reward, terminated, info = self.jax_env.step_env(
            key,
            self.state,
            jnp.array(action, dtype=jnp.float32),
            self.env_params,
        )
        self.state = new_state
        # `step_env` reports natural termination only; the time limit is
        # the env's `is_truncated`. Gymnasium wants the two apart, so
        # report the real flag instead of the hardcoded False this used
        # to return.
        truncated = self.jax_env.is_truncated(new_state, self.env_params)
        return (
            np.array(obs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            # NumPy like the rest of the step: Gymnasium compares info
            # leaves as NumPy or Python values and has no case for JAX's.
            jax.tree_util.tree_map(np.asarray, info),
        )

    def render(self):
        frames, self.screen, self.clock = self.jax_env.render(
            self.screen, self.state, self.env_params, self.frames, self.clock
        )
        self.frames = frames

        if self.render_mode == "human":
            # Only the renderers that open a pygame window hand a screen
            # back; the ones rebuilt on the shared instrument kit draw to
            # an image and return None. Pumping an uninitialised video
            # system raises, so an environment whose renderer never opened
            # a window is not asked about its events -- ``render()`` still
            # produces the frame, it just does not display it live.
            if self.screen is not None:
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


def _class_wrapper(jax_env_class: Type) -> Type[GymnasiumEnv]:
    class GymWrapperJaxEnv(GymnasiumEnv):
        def __init__(
            self,
            render_mode: Optional[str] = None,
            env_params: Optional[EnvParams] = None,
            **jax_env_kwargs,
        ):
            super().__init__(jax_env_class(**jax_env_kwargs), env_params, render_mode)

    return GymWrapperJaxEnv


def gym_wrapper_factory(jax_env_class: Type) -> Type[GymnasiumEnv]:
    """Deprecated: use ``target_gym.make_gymnasium(name)``, which also sets the
    scored parameters."""
    warnings.warn(
        "gym_wrapper_factory is deprecated; use target_gym.make_gymnasium(name)",
        DeprecationWarning,
        stacklevel=2,
    )
    return _class_wrapper(jax_env_class)
