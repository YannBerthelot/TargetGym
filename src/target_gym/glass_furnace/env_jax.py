from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import (
    FieldReading,
    actuate,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.expert_access import RegistryExperts
from target_gym.glass_furnace.env import (
    FUEL_DEAD_TIME_STEPS,
    N_REGEN_NODES,
    N_SETPOINTS,
    GlassFurnaceParams,
    GlassFurnaceState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    get_target_from_schedule,
    regenerator_diagnostics,
    settled_point,
)
from target_gym.glass_furnace.rendering import _render
from target_gym.utils import convert_raw_action_to_range
from target_gym.video import SavesVideo


class GlassFurnace(
    RegistryExperts,
    SavesVideo,
    environment.Environment[GlassFurnaceState, GlassFurnaceParams],
):
    """Regenerative glass furnace; six of nine states are hidden.

    Action (1,): [fuel rate], raw in [-1, 1] -> [fuel_min, fuel_max] kg/s
    """

    registry_name: str = "glass_furnace"

    render_furnace = classmethod(_render)
    screen_width = 700
    screen_height = 900

    # obs = [T_crown, T_air_preheat, fuel_pct, reversal_phase, target_T_crown, pull]
    obs_value_index: int = 0  # T_crown
    tracked_names: tuple = ("crown temperature (K)",)
    obs_target_index: int = 4  # target_T_crown

    # Two thermocouples and a vortex meter (PHYSICS.md, Readings). The crown's
    # noise goes after the modelled lag, on the measured copy: the state keeps
    # the lag clean.
    readings: tuple = (
        FieldReading("T_crown_meas", sigma=0.36),
        FieldReading("T_air_preheat", sigma=0.32),
        FieldReading("fuel_flow", sigma=3e-4),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (6,)
        self.positions_history = []
        self.integration_method = integration_method

    @property
    def default_params(self) -> GlassFurnaceParams:
        return GlassFurnaceParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: GlassFurnaceState,
        action: jnp.ndarray,
        params: GlassFurnaceParams = None,
    ):
        if params is None:
            params = self.default_params

        action, state = actuate(self, state, jnp.reshape(action, (1,)), params)

        new_state, metrics = compute_next_state(
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
    def get_obs(self, state: GlassFurnaceState, params: GlassFurnaceParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(
        self, state: GlassFurnaceState, params: GlassFurnaceParams
    ) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: GlassFurnaceParams = None
    ) -> Tuple[jnp.ndarray, GlassFurnaceState]:
        if params is None:
            params = self.default_params

        key, schedule_key, crown_key, gas_key = jax.random.split(key, 4)

        # A trim walk, not independent draws: the first slot is the level the
        # furnace is already being held at, and every later slot is the
        # previous one plus a small increment, clipped to the operating band.
        start_key, trim_key = jax.random.split(schedule_key)
        start = jax.random.uniform(
            start_key,
            minval=params.target_T_crown_start_range[0],
            maxval=params.target_T_crown_start_range[1],
        )
        trims = jax.random.uniform(
            trim_key,
            shape=(N_SETPOINTS,),
            minval=-params.target_T_crown_trim,
            maxval=params.target_T_crown_trim,
        )
        target_schedule = jnp.clip(
            start + jnp.cumsum(trims.at[0].set(0.0)),
            params.target_T_crown_range[0],
            params.target_T_crown_range[1],
        )
        initial_target = get_target_from_schedule(target_schedule, 0, params)
        # Near the level it is being held at, not somewhere else in the band.
        initial_T_crown = jnp.clip(
            initial_target
            + jax.random.uniform(
                crown_key,
                minval=-params.initial_T_crown_offset,
                maxval=params.initial_T_crown_offset,
            ),
            params.initial_T_crown_range[0],
            params.initial_T_crown_range[1],
        )

        # The gas of the day is stationary, and the furnace is found running on
        # it: the fuel flow burns the heat the orbit needs.
        lhv_deviation = params.lhv_noise_std * jax.random.normal(gas_key)
        raw, positions = settled_point(initial_T_crown)
        fuel = convert_raw_action_to_range(
            raw, min_action=params.fuel_min, max_action=params.fuel_max
        ) / (1.0 + lhv_deviation)
        T_gas = initial_T_crown + params.initial_T_gas_offset
        T_air, T_stack = regenerator_diagnostics(positions, T_gas, fuel, 0.0, params)
        state = GlassFurnaceState(
            time=0,
            T_gas=T_gas,
            T_crown=initial_T_crown,
            T_melt=positions[1],
            T_work=positions[2],
            T_rA=positions[4 : 4 + N_REGEN_NODES],
            T_rB=positions[4 + N_REGEN_NODES :],
            m_batch=positions[3],
            target_T_crown=initial_target,
            target_schedule=target_schedule,
            m_pull_disturbance=jnp.zeros(()),
            lhv_deviation=lhv_deviation,
            # The instrument is settled on the plant, which has just come
            # through a reversal, and the fuel pipeline starts full.
            T_crown_meas=initial_T_crown + params.initial_T_meas_offset,
            fuel_pipeline=jnp.full((FUEL_DEAD_TIME_STEPS,), fuel),
            fuel_flow=fuel,
            T_air_preheat=T_air,
            T_stack=T_stack,
        )

        obs = self.get_obs(state)
        return obs, state

    def initial_action(self, state: GlassFurnaceState, params: GlassFurnaceParams):
        """The raw action that keeps the reset's fuel flow."""
        span = params.fuel_max - params.fuel_min
        fuel = state.fuel_pipeline[-1:]
        return 2.0 * (fuel - params.fuel_min) / span - 1.0

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
