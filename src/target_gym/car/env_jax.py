from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.car.env import (
    CarParams,
    CarState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    road_profile,
)
from target_gym.car.rendering import _render
from target_gym.utils import save_video


class Car2D(environment.Environment[CarState, CarParams]):
    """
    JAX-compatible 2D car environment.
    """

    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (12,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> CarParams:
        return CarParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CarState,
        action: jnp.ndarray,
        params: CarParams = None,
    ):
        """
        Performs step transitions using JAX, returns observation, new state, reward, done, info
        """
        if params is None:
            params = self.default_params
        throttle = action

        new_state, metrics = compute_next_state(
            throttle, state, params, integration_method=self.integration_method
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

    def get_obs(self, state: CarState, params: CarParams = None):
        """
        Observation vector
        """
        if params is None:
            params = (
                self.default_params
            )  # TODO : propagate this into the code sometime, as having params given to get_obs is not standard gymnax API
        return get_obs(state, params=params, road_profile=road_profile, xp=jnp)

    def is_terminal(self, state: CarState, params: CarParams) -> jax.Array:
        return check_is_terminal(state, params, xp=jnp)

    def reset_env(
        self, key: chex.PRNGKey, params: CarParams = None
    ) -> Tuple[jnp.ndarray, CarState]:
        """
        Reset the environment using JAX random keys
        """
        if params is None:
            params = self.default_params

        key, velocity_key, target_key = jax.random.split(key, 3)

        initial_x = 0.0
        initial_velocity = jax.random.uniform(
            velocity_key,
            minval=params.initial_velocity_range[0],
            maxval=params.initial_velocity_range[1],
        )

        target_velocity = jax.random.uniform(
            target_key,
            minval=params.target_velocity_range[0],
            maxval=params.target_velocity_range[1],
        )

        state = CarState(
            t=0,
            x=initial_x,
            velocity=initial_velocity,
            target_velocity=target_velocity,
            throttle=params.initial_throttle,
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: CarParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: CarParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: CarParams) -> spaces.Box:
        """Observation space of the environment."""
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(CarState.__dataclass_fields__), dtype=jnp.float32
        )

    def save_video(
        self,
        select_action: Callable[[jnp.ndarray], jnp.ndarray],
        seed: int,
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
            seed=seed,
            format=format,
        )

    def render(self, screen, state: CarState, params: CarParams, frames, clock):
        """
        JAX-compatible rendering wrapper
        """
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = Car2D()
    seed = 42
    env_params = CarParams(max_steps_in_episode=1_000)
    action = 1.0
    env.save_video(
        lambda o: 1.0 if o[0] < 120 / 3.6 else -1,
        seed,
        folder="videos",
        episode_index=0,
        params=env_params,
        format="gif",
    )
