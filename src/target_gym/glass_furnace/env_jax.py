import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.glass_furnace.env import (
    N_SETPOINTS,
    GlassFurnaceParams,
    GlassFurnaceState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    get_target_from_schedule,
)
from target_gym.glass_furnace.rendering import _render
from target_gym.utils import save_video


class GlassFurnace(environment.Environment[GlassFurnaceState, GlassFurnaceParams]):
    render_furnace = classmethod(_render)
    screen_width = 700
    screen_height = 900

    # obs = [T_crown, fuel_pct (% of fuel_max), target_T_crown]
    obs_value_index: int = 0  # T_crown
    obs_target_index: int = 2  # target_T_crown

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (3,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> GlassFurnaceParams:
        return GlassFurnaceParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: GlassFurnaceState,
        action: jnp.ndarray,
        params: GlassFurnaceParams = None,
    ):
        if params is None:
            params = self.default_params

        fuel_raw = action
        if not isinstance(action, float):
            fuel_raw = action.reshape(())

        new_state, metrics = compute_next_state(
            fuel_raw, state, params, key, integration_method=self.integration_method
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

    def get_obs(self, state: GlassFurnaceState, params: GlassFurnaceParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminal(
        self, state: GlassFurnaceState, params: GlassFurnaceParams
    ) -> jnp.ndarray:
        return check_is_terminal(state, params)

    def reset_env(
        self, key: chex.PRNGKey, params: GlassFurnaceParams = None
    ) -> Tuple[jnp.ndarray, GlassFurnaceState]:
        if params is None:
            params = self.default_params

        key, schedule_key, crown_key = jax.random.split(key, 3)

        initial_T_crown = jax.random.uniform(
            crown_key,
            minval=params.initial_T_crown_range[0],
            maxval=params.initial_T_crown_range[1],
        )
        target_schedule = jax.random.uniform(
            schedule_key,
            shape=(N_SETPOINTS,),
            minval=params.target_T_crown_range[0],
            maxval=params.target_T_crown_range[1],
        )
        initial_target = get_target_from_schedule(target_schedule, 0, params)

        state = GlassFurnaceState(
            time=0,
            T_crown=initial_T_crown,
            T_melt=params.initial_T_melt,
            T_work=params.initial_T_work,
            target_T_crown=initial_target,
            target_schedule=target_schedule,
            m_pull_disturbance=jnp.zeros(()),
            fuel_flow=0.5 * (params.fuel_min + params.fuel_max),
        )

        obs = self.get_obs(state)
        return obs, state

    def action_space(self, params: GlassFurnaceParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: GlassFurnaceParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: GlassFurnaceParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(GlassFurnaceState.__dataclass_fields__), dtype=jnp.float32
        )

    @property
    def expert_policy(self):
        from target_gym.experts.pid import FunctionalExpertPolicy, make_glass_furnace_pid, pid_step
        params, zero_state = make_glass_furnace_pid()
        return FunctionalExpertPolicy(params, zero_state, pid_step)

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
        self,
        screen,
        state: GlassFurnaceState,
        params: GlassFurnaceParams,
        frames,
        clock,
    ):
        frames, screen, clock = self.render_furnace(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = GlassFurnace()
    seed = 42
    env_params = GlassFurnaceParams(max_steps_in_episode=500)
    os.makedirs("videos/glass_furnace", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        seed,
        folder="videos/glass_furnace",
        episode_index=0,
        params=env_params,
        format="gif",
    )
