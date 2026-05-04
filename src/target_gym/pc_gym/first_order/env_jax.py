import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.pc_gym.first_order.env import (
    FirstOrderParams,
    FirstOrderState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.pc_gym.first_order.rendering import _render
from target_gym.utils import save_video


class FirstOrderSystem(environment.Environment[FirstOrderState, FirstOrderParams]):
    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [x, target_x]
    obs_value_index: int = 0  # x
    obs_target_index: int = 1  # target_x

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (2,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> FirstOrderParams:
        return FirstOrderParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FirstOrderState,
        action: jnp.ndarray,
        params: FirstOrderParams = None,
    ):
        if params is None:
            params = self.default_params

        u = action
        if not isinstance(action, float):
            u = action.reshape(())

        new_state, metrics = compute_next_state(
            u, state, params, integration_method=self.integration_method
        )

        reward = compute_reward(new_state, params)
        terminated, truncated = check_is_terminal(new_state, params)
        done = terminated | truncated

        obs = self.get_obs(new_state)
        return obs, new_state, reward, done, {"last_state": new_state}

    def get_obs(self, state: FirstOrderState, params: FirstOrderParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(
        self, state: FirstOrderState, params: FirstOrderParams
    ) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: FirstOrderParams = None
    ) -> Tuple[jnp.ndarray, FirstOrderState]:
        if params is None:
            params = self.default_params

        key, x_key, target_key = jax.random.split(key, 3)

        initial_x = jax.random.uniform(
            x_key,
            minval=params.initial_x_range[0],
            maxval=params.initial_x_range[1],
        )
        target_x = jax.random.uniform(
            target_key,
            minval=params.target_x_range[0],
            maxval=params.target_x_range[1],
        )

        state = FirstOrderState(time=0, x=initial_x, target_x=target_x, u=0.0)
        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: FirstOrderParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: FirstOrderParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: FirstOrderParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(FirstOrderState.__dataclass_fields__), dtype=jnp.float32
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

    def render(
        self, screen, state: FirstOrderState, params: FirstOrderParams, frames, clock
    ):
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock

    @property
    def expert_policy(self):
        from target_gym.experts.pid import FunctionalExpertPolicy, make_first_order_pid, pid_step
        params, zero_state = make_first_order_pid()
        return FunctionalExpertPolicy(params, zero_state, pid_step)

    def make_pid(self):
        """Return a ready-to-use StatefulPID for state tracking."""
        from target_gym.experts.pid import make_first_order_stateful_pid

        return make_first_order_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a GradientMPC oracle for state tracking."""
        from target_gym.experts.mpc import make_first_order_mpc

        if params is None:
            params = self.default_params
        return make_first_order_mpc(self, params, **kwargs)


if __name__ == "__main__":
    env = FirstOrderSystem()
    seed = 42
    env_params = FirstOrderParams(max_steps_in_episode=200)
    os.makedirs("videos/first_order", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/first_order",
        episode_index=0,
        params=env_params,
        format="gif",
    )
