import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.cement_kiln.env import (
    CementKilnParams,
    CementKilnState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    gas_sweep,
    get_obs,
    steady_profile,
)
from target_gym.cement_kiln.rendering import _render
from target_gym.utils import save_video


class CementKiln(environment.Environment[CementKilnState, CementKilnParams]):
    """Cement rotary kiln with a half-hour transport delay.

    Observation (8): [lime_pct, T_burning_zone, T_exhaust, T_back_end,
                      feed_rate, fuel_pct, speed_pct, target_lime_pct]
    Action      (2): [fuel, kiln_speed], raw in [-1, 1]
    """

    render_kiln = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: int = 0  # discharge free lime
    tracked_names: tuple = ("discharge free lime (%)",)
    obs_target_index: int = 7  # target free lime

    def __init__(self, integration_method: str = "rk4_1"):
        self.obs_shape = (8,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> CementKilnParams:
        return CementKilnParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: CementKilnState,
        action: jnp.ndarray,
        params: CementKilnParams = None,
    ):
        if params is None:
            params = self.default_params

        new_state, _ = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )
        reward = compute_reward(new_state, params, xp=jnp)
        # gymnax >= 1.0 owns truncation; step_env reports natural termination only.
        terminated, _ = check_is_terminal(new_state, params, xp=jnp)
        return (
            self.get_obs(new_state),
            new_state,
            reward,
            terminated,
            {"last_state": new_state},
        )

    def get_obs(self, state: CementKilnState, params: CementKilnParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(
        self, state: CementKilnState, params: CementKilnParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    def reset_env(
        self, key: chex.PRNGKey, params: CementKilnParams = None
    ) -> Tuple[jnp.ndarray, CementKilnState]:
        if params is None:
            params = self.default_params

        key, target_key, feed_key = jax.random.split(key, 3)

        # Start from the lined-out axial profile. A kiln started from an
        # arbitrary profile would spend hours of episode time relaxing, which
        # is an artefact rather than a control problem.
        T_solid, T_wall, alpha, lime = steady_profile(params)
        T_gas, _, _, T_exhaust = gas_sweep(T_solid, T_wall, params.fuel_nominal, params)

        target_lime = jax.random.uniform(
            target_key,
            minval=params.target_lime_range[0],
            maxval=params.target_lime_range[1],
        )
        raw_meal = params.raw_meal_nominal + params.feed_noise_std * jax.random.normal(
            feed_key
        )

        state = CementKilnState(
            time=0,
            T_solid=T_solid,
            T_wall=T_wall,
            alpha=alpha,
            lime=lime,
            T_gas=T_gas,
            T_exhaust=T_exhaust,
            fuel=jnp.asarray(params.fuel_nominal),
            rpm=jnp.asarray(params.rpm_nominal),
            raw_meal=raw_meal,
            target_lime=target_lime,
        )
        return self.get_obs(state), state

    def action_space(self, params: CementKilnParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: CementKilnParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: CementKilnParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(CementKilnState.__dataclass_fields__), dtype=jnp.float32
        )

    def make_pid(self):
        """Return the cascade controller (free lime -> burning zone -> fuel)."""
        from target_gym.experts.pid import make_cement_kiln_stateful_pid

        return make_cement_kiln_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        from target_gym.experts.mpc import make_cement_kiln_mpc

        if params is None:
            params = self.default_params
        return make_cement_kiln_mpc(self, params, **kwargs)

    @property
    def expert_policy(self):
        from target_gym.experts.pid import (
            FunctionalExpertPolicy,
            make_cement_kiln_pid,
            mimo_pid_step,
        )

        pid_params, zero_state = make_cement_kiln_pid()
        return FunctionalExpertPolicy(pid_params, zero_state, mimo_pid_step)

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
        self, screen, state: CementKilnState, params: CementKilnParams, frames, clock
    ):
        frames, screen, clock = self.render_kiln(screen, state, params, frames, clock)
        return frames, screen, clock


if __name__ == "__main__":
    env = CementKiln()
    os.makedirs("videos/cement_kiln", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1, size=(2,)),
        42,
        folder="videos/cement_kiln",
        episode_index=0,
        params=CementKilnParams(max_steps_in_episode=240),
        format="gif",
    )
