from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.energy.wind_turbine.env import (
    WindTurbineParams,
    WindTurbineState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    get_obs,
    omega_rated,
)
from target_gym.energy.wind_turbine.rendering import _render
from target_gym.expert_access import RegistryExperts
from target_gym.video import SavesVideo


class WindTurbine(
    RegistryExperts,
    SavesVideo,
    environment.Environment[WindTurbineState, WindTurbineParams],
):
    """NREL 5 MW reference turbine, collective-pitch power regulation.

    Observation (5,): [omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]
    Action      (2,): [pitch_raw, torque_raw] in [-1, 1]
    """

    registry_name: str = "wind_turbine"

    render_turbine = classmethod(_render)
    screen_width = 700
    screen_height = 800

    obs_value_index: int = 3  # electrical power (MW)
    tracked_names: tuple = ("electrical power (MW)",)
    obs_target_index: int = 4  # target power (MW)

    def __init__(self, integration_method: str = "rk4_2"):
        self.obs_shape = (5,)
        self.integration_method = integration_method

    @property
    def default_params(self) -> WindTurbineParams:
        return WindTurbineParams()

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: WindTurbineState,
        action: jnp.ndarray,
        params: WindTurbineParams = None,
    ):
        if params is None:
            params = self.default_params
        action = jnp.atleast_1d(jnp.asarray(action)).reshape(-1)

        new_state, _ = compute_next_state(
            action, state, params, key, integration_method=self.integration_method
        )
        # A trip is part of the kernel: the step is charged the trip cost and
        # the plant restarts at once, or is lost (``terminated``) with
        # ``params.continuing = False`` (``base.finish_step``).
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params
        )
        return self.get_obs(new_state), new_state, reward, terminated, info

    def get_obs(self, state, params: WindTurbineParams = None):
        if params is None:
            params = self.default_params
        return get_obs(state, params=params)

    def is_terminated(self, state, params) -> jnp.ndarray:
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(
        self, key: chex.PRNGKey, params: WindTurbineParams = None
    ) -> Tuple[jnp.ndarray, WindTurbineState]:
        if params is None:
            params = self.default_params

        key, wind_key, target_key, gust_key = jax.random.split(key, 4)

        v_mean = jax.random.uniform(
            wind_key,
            minval=params.v_mean_range[0],
            maxval=params.v_mean_range[1],
        )
        v_wind = jnp.clip(
            v_mean + params.turbulence_std * jax.random.normal(gust_key),
            params.v_min,
            params.v_max,
        )
        target_power = jax.random.uniform(
            target_key,
            minval=params.target_power_range[0],
            maxval=params.target_power_range[1],
        )

        # Start at rated speed producing the target power: a turbine in
        # above-rated wind is already regulating, so starting spun-down would
        # make every episode begin with a run-up rather than a control problem.
        w0 = omega_rated(params)
        torque0 = jnp.clip(
            target_power / (params.eta_gen * params.N_gear * w0),
            0.0,
            params.torque_max,
        )
        # Pitch that roughly balances the rotor at this wind, from the same
        # steady-state relation the controller has to find.
        pitch0 = jnp.clip(2.0 * (v_wind - params.v_rated), params.pitch_min, 20.0)

        state = WindTurbineState(
            time=0,
            omega=w0,
            pitch=pitch0,
            torque=torque0,
            v_wind=v_wind,
            v_mean=v_mean,
            pitch_cmd=pitch0,
            torque_cmd=torque0,
            target_power=target_power,
        )
        return self.get_obs(state), state

    def action_space(self, params: WindTurbineParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: WindTurbineParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: WindTurbineParams) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(
            -inf, inf, len(WindTurbineState.__dataclass_fields__), dtype=jnp.float32
        )

    def render(self, screen, state, params, frames, clock):
        frames, screen, clock = self.render_turbine(
            screen, state, params, frames, clock
        )
        return frames, screen, clock
