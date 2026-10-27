from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.energy.battery.env import (
    N_DISPATCH_BLOCKS,
    BatteryParams,
    BatteryState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
)
from target_gym.energy.battery.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo


class GridBattery(
    RegistryExperts, SavesVideo, environment.Environment[BatteryState, BatteryParams]
):
    """Grid battery storage tracking a dispatch signal.

    Observation (5,): [soc, V_cell, T_cell, P_MW, target_P_MW]
    Action      (1,): power, raw in [-1, 1] -> [-P_max, +P_max] (positive = discharge)
    """

    registry_name: str = "battery"

    render_battery = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 3  # delivered power (MW)
    tracked_names: tuple = ("delivered power (MW)",)
    obs_target_index: int = 4  # dispatch (MW)

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (5,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BatteryParams:
        return BatteryParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

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
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state), new_state, reward, terminated, info

    def get_obs(self, state, params: BatteryParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
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
        # A schedule of held setpoints, drawn once. The pack is told what it
        # will be asked for, and delivering it is the task; within a scored
        # episode, exact delivery stays inside the charge window and what it
        # costs is wear.
        schedule = jax.random.uniform(
            dispatch_key,
            shape=(N_DISPATCH_BLOCKS,),
            minval=-params.dispatch_range,
            maxval=params.dispatch_range,
        )
        target = jnp.clip(schedule[0], -params.power_max, params.power_max)

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
            dispatch_schedule=schedule,
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

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_battery(
            screen, state, params, frames, clock
        )
        return frames, screen, clock
