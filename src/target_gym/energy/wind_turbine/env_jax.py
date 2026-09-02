import os
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset
from target_gym.energy.wind_turbine.env import (
    WindTurbineParams,
    WindTurbineState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
    omega_rated,
)
from target_gym.energy.wind_turbine.rendering import _render
from target_gym.utils import save_video


class WindTurbine(environment.Environment[WindTurbineState, WindTurbineParams]):
    """NREL 5 MW reference turbine, collective-pitch power regulation.

    Observation (5,): [omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]
    Action      (2,): [pitch_raw, torque_raw] in [-1, 1]
    """

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

    def make_pid(self):
        """Return the standard above-rated turbine controller."""
        from target_gym.experts.pid import make_wind_turbine_stateful_pid

        return make_wind_turbine_stateful_pid()

    def make_mpc(self, params=None, **kwargs):
        """Return a gradient MPC oracle for power regulation."""
        from target_gym.experts.mpc import make_wind_turbine_mpc

        if params is None:
            params = self.default_params
        return make_wind_turbine_mpc(self, params, **kwargs)

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
        frames, screen, clock = self.render_turbine(
            screen, state, params, frames, clock
        )
        return frames, screen, clock


if __name__ == "__main__":
    env = WindTurbine()
    os.makedirs("videos/wind_turbine", exist_ok=True)
    env.save_video(
        lambda o: np.random.uniform(-1, 1, size=(2,)),
        42,
        folder="videos/wind_turbine",
        episode_index=0,
        params=WindTurbineParams(max_steps_in_episode=400),
        format="gif",
    )
