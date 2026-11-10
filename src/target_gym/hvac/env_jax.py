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
from target_gym.hvac.env import (
    DAY_STEPS,
    HVACParams,
    HVACState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    floor_surface_temperature,
    get_obs,
    internal_gain,
    outdoor_temperature,
    return_temperature,
    slab_heat,
    solar_gain,
    solve_air_and_surface,
    split_gains,
    steady_state,
)
from target_gym.hvac.rendering import _render
from target_gym.video import SavesVideo


class BuildingHVAC(
    RegistryExperts, SavesVideo, environment.Environment[HVACState, HVACParams]
):
    """Single-zone building heated by a TABS slab (ISO 13790 5R1C plus [9]'s slab).

    Observation (12,): [T_air, T_out, T_out_24h, T_core, T_return, T_supply,
                        sin_h, cos_h, band_centre, T_out_forecast_24h,
                        price_now, h_to_price_change]
    Action      (1,): supply water temperature, raw in [-1, 1] ->
                      [T_supply_min, T_supply_max]
    """

    #: What a trip's restart carries over (``base.failure_kernel``): the outside
    #: world and the records, never the plant (docs/proposals/long-strata.md, S2).
    OUTSIDE_WORLD = (
        "T_out",
        "T_out_hist",
        "weather_dev",
    )
    #: Runs as a continuing plant (``target_gym.stream``): its day wraps on a step
    #: count, and its scenarios are regimes with no event of their own.
    STREAM_READY = True

    registry_name: str = "hvac"

    render_hvac = classmethod(_render)
    screen_width = 700
    screen_height = 900

    obs_value_index: int = 0  # T_air
    tracked_names: tuple = ("zone air temperature (deg C)",)
    obs_target_index: int = 8  # the band's centre; comfort_tolerance is its half

    # Four temperature sensors of one class (PHYSICS.md, Readings).
    readings: tuple = (
        FieldReading("T_air", sigma=0.023, step=0.1),
        FieldReading("T_out", sigma=0.023, step=0.1),
        FieldReading("T_core", sigma=0.023, step=0.1),
        FieldReading("T_return", sigma=0.023, step=0.1),
    )

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (12,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> HVACParams:
        return HVACParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params, tripped=None):
        return compute_reward_terms(state, params, tripped=tripped)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: HVACState,
        action: jnp.ndarray,
        params: HVACParams = None,
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
    def get_obs(self, state: HVACState, params: HVACParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state: HVACState, params: HVACParams) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: HVACParams = None
    ) -> Tuple[jnp.ndarray, HVACState]:
        if params is None:
            params = self.default_params

        # Established operation (rule 17): the building has run on its slab
        # for days, settled at the band's heating edge [9, p. 5] for the last
        # day's mean weather and gains, as a heating curve holds it [7, p. 7].
        weather_dev = params.T_out_noise_std * jax.random.normal(key)
        past = jnp.arange(-DAY_STEPS + 1, 1)
        history = outdoor_temperature(past, weather_dev, params)
        day_int = jnp.mean(internal_gain(past, params))
        day_sol = jnp.mean(solar_gain(past, params))
        T_mass, T_core, _, T_supply = steady_state(
            params, jnp.mean(history), params.band_low, day_int, day_sol
        )
        T_out = history[-1]
        phi_ia, phi_st, _ = split_gains(
            internal_gain(0, params), solar_gain(0, params), params
        )
        T_air, T_surface = solve_air_and_surface(
            T_mass, T_core, T_out, phi_st, phi_ia, params
        )
        T_supply = jnp.clip(T_supply, params.T_supply_min, params.T_supply_max)
        state = HVACState(
            time=0,
            T_mass=T_mass,
            T_core=T_core,
            T_air=T_air,
            T_surface=T_surface,
            T_out=T_out,
            T_out_hist=history,
            weather_dev=weather_dev,
            T_supply=T_supply,
            Q_slab=slab_heat(T_supply, T_core, params),
            T_return=return_temperature(T_supply, T_core, params),
            T_floor=floor_surface_temperature(T_core, T_air, params),
            frost=jnp.zeros(()),
        )
        return self.get_obs(state), state

    def start_event(self, state: HVACState, key, params: HVACParams) -> HVACState:
        """The stream's event hook (docs/proposals/long-strata.md, S1). The
        building's scenarios are regimes of weather and gains with no event of
        their own, so nothing starts: the state is returned as it is."""
        del key, params
        return state

    def initial_action(self, state: HVACState, params: HVACParams):
        """The raw action that keeps the reset's supply water."""
        span = params.T_supply_max - params.T_supply_min
        return jnp.atleast_1d(2.0 * (state.T_supply - params.T_supply_min) / span - 1.0)

    def action_space(self, params: HVACParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0]),
            high=jnp.array([1.0]),
            shape=(1,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: HVACParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: HVACParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(HVACState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state: HVACState, params: HVACParams, frames, clock):
        frames, screen, clock = self.render_hvac(screen, state, params, frames, clock)
        return frames, screen, clock
