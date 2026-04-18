import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.pc_gym.four_tank.env import (
    FourTankParams,
    FourTankState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.pc_gym.four_tank.rendering import _render
from target_gym.utils import save_video


class FourTank(environment.Environment[FourTankState, FourTankParams]):
    render_car = classmethod(_render)
    screen_width = 600
    screen_height = 400

    # obs = [h1, h2, h3, h4, target_h1, target_h2]
    obs_value_index: tuple = (0, 1)  # h1, h2
    obs_target_index: tuple = (4, 5)  # target_h1, target_h2

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (6,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> FourTankParams:
        return FourTankParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FourTankState,
        action: jnp.ndarray,
        params: FourTankParams = None,
    ):
        if params is None:
            params = self.default_params

        action = action.reshape((2,))

        new_state, metrics = compute_next_state(
            action, state, params, integration_method=self.integration_method
        )

        reward = compute_reward(new_state, params)
        terminated, truncated = check_is_terminal(new_state, params)
        done = terminated | truncated

        obs = self.get_obs(new_state)
        return obs, new_state, reward, done, {"last_state": new_state}

    def get_obs(self, state: FourTankState, params: FourTankParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(self, state: FourTankState, params: FourTankParams) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: FourTankParams = None
    ) -> Tuple[jnp.ndarray, FourTankState]:
        if params is None:
            params = self.default_params

        key, h1_key, h2_key, h3_key, h4_key, t1_key, t2_key = jax.random.split(key, 7)

        h1 = jax.random.uniform(
            h1_key, minval=params.initial_h1_range[0], maxval=params.initial_h1_range[1]
        )
        h2 = jax.random.uniform(
            h2_key, minval=params.initial_h2_range[0], maxval=params.initial_h2_range[1]
        )
        h3 = jax.random.uniform(
            h3_key, minval=params.initial_h3_range[0], maxval=params.initial_h3_range[1]
        )
        h4 = jax.random.uniform(
            h4_key, minval=params.initial_h4_range[0], maxval=params.initial_h4_range[1]
        )
        target_h1 = jax.random.uniform(
            t1_key, minval=params.target_h1_range[0], maxval=params.target_h1_range[1]
        )
        target_h2 = jax.random.uniform(
            t2_key, minval=params.target_h2_range[0], maxval=params.target_h2_range[1]
        )

        state = FourTankState(
            time=0,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            target_h1=target_h1,
            target_h2=target_h2,
            v1=0.0,
            v2=0.0,
        )
        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: FourTankParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: FourTankParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: FourTankParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(FourTankState.__dataclass_fields__), dtype=jnp.float32
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
        self, screen, state: FourTankState, params: FourTankParams, frames, clock
    ):
        frames, screen, clock = self.render_car(screen, state, params, frames, clock)
        return frames, screen, clock

    @property
    def expert_policy(self):
        """Tuned gain-scheduled MIMO PID controller for level tracking."""
        from target_gym.experts.pid import make_four_tank_stateful_gs_pid

        return make_four_tank_stateful_gs_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a GradientMPC oracle for level tracking."""
        from target_gym.experts.mpc import make_four_tank_mpc

        if params is None:
            params = self.default_params
        return make_four_tank_mpc(self, params, **kwargs)


if __name__ == "__main__":
    env = FourTank()
    seed = 42
    env_params = FourTankParams(max_steps_in_episode=500)
    os.makedirs("videos/four_tank", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1, size=(2,)),
        seed,
        folder="videos/four_tank",
        episode_index=0,
        params=env_params,
        format="gif",
    )
