from typing import Tuple

import chex
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    ChannelReading,
    FieldReading,
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.energy.battery.env import (
    BatteryParams,
    BatteryState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    initial_state,
)
from target_gym.energy.battery.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo


class GridBattery(
    RegistryExperts, SavesVideo, environment.Environment[BatteryState, BatteryParams]
):
    """Grid battery selling Dynamic Regulation, steered by its baseline.

    Observation (13,): [reported_soc, V_cell, T_cell, P_MW, required_MW,
    signal_pu, baseline_now_MW, baseline_next_MW, baseline_after_MW,
    export_requirement_MWh, import_requirement_MWh, seconds_to_next_period,
    period_lost]
    Action      (1,): Operational Baseline, raw in [-1, 1] -> +-40 % of the
    contracted power (positive = export), for the first period past gate
    closure
    """

    registry_name: str = "battery"

    render_battery = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 3  # power at the meter (MW)
    tracked_names: tuple = ("delivered power (MW)",)
    obs_target_index: int = 4  # required power: baseline plus response (MW)

    # The BMS display, a cell monitor, an NTC and a revenue meter (PHYSICS.md,
    # Readings). The BMS's own bias is in the model (``reported_soc``); this
    # is the display's scatter on top. Cell voltage is a channel reading. The
    # NTC's step is 152.59 uV at the GPIO over the assumed 50 mV/K divider.
    readings: tuple = (
        FieldReading("soc_reported", sigma=4.7e-5),
        ChannelReading(1, sigma=5.5e-5),
        FieldReading("T_cell", sigma=0.03, step=152.59e-6 / 0.05),
        FieldReading("power", sigma=330.0),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (13,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> BatteryParams:
        return BatteryParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: BatteryState,
        action: jnp.ndarray,
        params: BatteryParams = None,
    ):
        if params is None:
            params = self.default_params
        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)

        new_state, _ = compute_next_state(
            action.reshape(()),
            state,
            params,
            key,
            integration_method=self.integration_method,
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state, params), new_state, reward, terminated, info

    @measured_obs
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
        state = initial_state(key, params)
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
