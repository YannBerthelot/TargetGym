import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.energy.battery.env import (
    BatteryParams,
    BatteryState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)
from target_gym.energy.battery.rendering import _render
from target_gym.utils import save_video


class GridBattery(environment.Environment[BatteryState, BatteryParams]):
    """Grid battery storage tracking a dispatch signal.

    Observation (5,): [soc, V_cell, T_cell, P_MW, target_P_MW]
    Action      (1,): power, raw in [-1, 1] -> [-P_max, +P_max] (positive = discharge)
    """

    render_battery = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 3  # delivered power (MW)
    obs_target_index: int = 4  # dispatch (MW)

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (5,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BatteryParams:
        return BatteryParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: BatteryState,
        action: jnp.ndarray,
        params: BatteryParams = None,
    ):
        if params is None:
            params = self.default_params
        action_raw = action
        if not isinstance(action, float):
            action_raw = jnp.asarray(action).reshape(())

        new_state, _ = compute_next_state(
            action_raw, state, params, key, integration_method=self.integration_method
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

    def get_obs(self, state, params: BatteryParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    def reset_env(
        self, key: chex.PRNGKey, params: BatteryParams = None
    ) -> Tuple[jnp.ndarray, BatteryState]:
        if params is None:
            params = self.default_params

        key, soc_key, dispatch_key = jax.random.split(key, 3)

        soc = jax.random.uniform(
            soc_key,
            minval=params.initial_soc_range[0],
            maxval=params.initial_soc_range[1],
        )
        target = jnp.clip(
            params.dispatch_std * jax.random.normal(dispatch_key),
            -params.power_max,
            params.power_max,
        )

        # Rested pack: no diffusion voltage, at ambient, undegraded. A battery
        # entering a dispatch window has typically been idle, and starting with
        # a polarisation voltage would be an artefact.
        state = BatteryState(
            time=0,
            soc=soc,
            v_rc=jnp.zeros(()),
            T_cell=jnp.asarray(params.T_ambient),
            q_loss=jnp.zeros(()),
            current=jnp.zeros(()),
            power=jnp.zeros(()),
            target_power=target,
        )
        return self.get_obs(state), state

    def action_space(self, params: BatteryParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]), high=jnp.array([1.0]), shape=(1,), dtype=jnp.float32
        )

    def observation_space(self, params: BatteryParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: BatteryParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(BatteryState.__dataclass_fields__), dtype=jnp.float32
        )

    def make_pid(self):
        """Return the dispatch-following controller with state-of-charge guarding."""
        from target_gym.experts.pid import make_battery_stateful_pid

        return make_battery_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a gradient MPC oracle for dispatch tracking."""
        from target_gym.experts.mpc import make_battery_mpc

        if params is None:
            params = self.default_params
        return make_battery_mpc(self, params, **kwargs)

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

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_battery(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = GridBattery()
    os.makedirs("videos/battery", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1),
        42,
        folder="videos/battery",
        episode_index=0,
        params=BatteryParams(max_steps_in_episode=360),
        format="gif",
    )
