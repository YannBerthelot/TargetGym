import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.pc_gym.cstr.env import (
    CSTRParams,
    CSTRState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.pc_gym.cstr.rendering import _render
from target_gym.utils import save_video


class CSTR(environment.Environment[CSTRState, CSTRParams]):
    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (3,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> CSTRParams:
        return CSTRParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CSTRState,
        action: jnp.ndarray,
        params: CSTRParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params
        T_c = action

        new_state, metrics = compute_next_state(
            T_c, state, params, integration_method=self.integration_method
        )

        reward = compute_reward(new_state, params, xp=jnp)
        terminated, truncated = check_is_terminal(new_state, params, xp=jnp)
        done = terminated | truncated

        obs = self.get_obs(new_state)
        return (
            obs,
            new_state,
            reward,
            done,
            {"last_state": new_state},
        )

    def get_obs(self, state: CSTRState, params: CSTRParams = None):
        """
        Observation vector
        """
        if params is None:
            params = (
                self.default_params
            )  # TODO : propagate this into the code sometime, as having params given to get_obs is not standard gymnax API
        return get_obs(state, params=params)

    def is_terminal(self, state: CSTRState, params: CSTRParams) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: CSTRParams = None
    ) -> Tuple[jnp.ndarray, CSTRState]:
        """
        Reset the environment using JAX random keys
        """
        if params is None:
            params = self.default_params

        key, target_key, C_a_key = jax.random.split(key, 3)

        initial_C_a = jax.random.uniform(
            target_key,
            minval=params.initial_CA_range[0],
            maxval=params.initial_CA_range[1],
        )

        initial_target_C_a = jax.random.uniform(
            target_key,
            minval=params.target_CA_range[0],
            maxval=params.target_CA_range[1],
        )

        state = CSTRState(
            t=0,
            C_a=initial_C_a,
            T=params.initial_T,
            target_CA=initial_target_C_a,
            T_c=0.0,
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: CSTRParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: CSTRParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: CSTRParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(CSTRState.__dataclass_fields__), dtype=jnp.float32
        )

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        key: chex.PRNGKey,
        params=None,
        folder="videos",
        episode_index=0,
        FPS=60,
        format="mp4",
    ):
        return save_video(
            self,
            select_action,
            folder,
            episode_index,
            FPS,
            params,
            seed=key,
            format=format,
        )

    def render(self, screen, state: CSTRState, params: CSTRParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = CSTR()
    seed = 42
    env_params = CSTRParams(max_steps_in_episode=1_000, delta_t=1e-3)
    os.makedirs("videos/cstr", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/cstr",
        episode_index=0,
        params=env_params,
        format="gif",
    )
